"""
token_optimizer
----------------
Compresses long context before it's used, verifies the compression didn't
change its meaning, caches recent LLM calls to avoid repeat spend, and
converts every token saved into a real dollar figure.

Deliberately reuses existing SDK infrastructure instead of adding new
dependencies:
  - local_extractor's filler-stripping regexes (the same ones compress_essence()
    uses for memory writes) power compress_context() below, minus that
    function's 140-char per-item cap — a whole context blob isn't a single
    telegraphic memory bullet, and unlike compress_essence() this never
    discards code/technical content, only filler prose around it.
  - local_embeddings.embed()/cosine_similarity() (the SDK's one embedding
    model) power both the compression safety check and the semantic cache —
    no second similarity mechanism.
  - litellm.token_counter for real per-model token counts, since litellm is
    already a hard dependency (classifier.py/reflector.py/mcp_server.py's
    recap all call litellm directly).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from typing import Any

from magnet.local_extractor import _CODE_FENCE, _INLINE_CODE, _FILLER_PATTERNS
from magnet.local_embeddings import embed, cosine_similarity

logger = logging.getLogger(__name__)

# Below a threshold this low, the compression is assumed to have changed
# what the text actually says — discard it and ship the original rather
# than risk losing meaning. Configurable because "close enough" depends on
# how much the caller's use case tolerates paraphrase drift.
MIN_SIMILARITY = float(os.environ.get("MAGNET_COMPRESSION_MIN_SIMILARITY", "0.75"))

# Compressing a short string isn't worth an embedding call on each side —
# the token savings wouldn't cover the overhead.
MIN_COMPRESS_CHARS = int(os.environ.get("MAGNET_COMPRESSION_MIN_CHARS", "400"))

# How long a compressed-text / semantic-cache entry stays valid. Short-lived
# on purpose — this is about not re-doing near-duplicate work within the
# same working session, not a durable cache that could go stale.
CACHE_TTL_SECONDS = int(os.environ.get("MAGNET_CACHE_TTL_SECONDS", "600"))

# How similar a new query has to be to a cached one to reuse its cached
# response outright — much stricter than MIN_SIMILARITY, since serving a
# stale answer to a not-quite-matching question is worse than a cache miss.
CACHE_MIN_SIMILARITY = float(os.environ.get("MAGNET_CACHE_MIN_SIMILARITY", "0.92"))

# The embedding model (all-MiniLM-L6-v2) silently truncates input past this
# many tokens — anything embedded whole beyond this length only has its
# FIRST _EMBEDDING_MAX_TOKENS actually compared, regardless of how long the
# real document is. Every verification window built below is kept under
# this on purpose; see compress_and_verify()'s chunking.
_EMBEDDING_MAX_TOKENS = 256

# Target size (in tokens) of each chunk when a document is too long to
# verify as one whole-text embedding. Kept well under _EMBEDDING_MAX_TOKENS
# once padding is added on both sides (see CHUNK_OVERLAP_TOKENS).
CHUNK_TARGET_TOKENS = int(os.environ.get("MAGNET_COMPRESSION_CHUNK_TOKENS", "120"))

# Extra ORIGINAL-text context stitched onto each side of a chunk before
# embedding it, so the similarity check judges a chunk's compression
# in-context rather than as an isolated, context-free fragment.
CHUNK_OVERLAP_TOKENS = int(os.environ.get("MAGNET_COMPRESSION_CHUNK_OVERLAP_TOKENS", "20"))

# USD per 1,000,000 INPUT tokens. Mirrors the pricing already shown to
# prospects (AgentMagnet/client/src/pages/Home.tsx's modelSavings table) so
# the dashboard's dollar figures are consistent with the marketing copy.
PRICE_PER_MILLION_TOKENS: dict[str, float] = {
    "claude-sonnet-5": 2.00,
    "claude-opus-5": 5.00,
    "claude-haiku-4-5": 1.00,
    "gpt-4o-mini": 0.15,
    "gpt-4o": 5.00,
    "gemini-2.0-flash": 0.075,
}
DEFAULT_PRICE_MODEL = os.environ.get("MAGNET_DEFAULT_PRICE_MODEL", "claude-sonnet-5")


def _price_per_million(model: str | None) -> float:
    return PRICE_PER_MILLION_TOKENS.get(model or DEFAULT_PRICE_MODEL, PRICE_PER_MILLION_TOKENS[DEFAULT_PRICE_MODEL])


def _litellm_model_name(model: str | None) -> str:
    """litellm needs a provider-recognized name to pick a tokenizer. This
    only ever feeds a dollar *estimate*, not billing, so a close-enough
    same-family tokenizer (BPE vocabularies are similar in size) is fine —
    it never needs to exactly match the model the dashboard prices against."""
    if not model:
        return "gpt-4o-mini"
    if "claude" in model:
        return "claude-3-5-sonnet-20241022"
    if "gemini" in model:
        return "gemini/gemini-2.0-flash"
    return model


def count_tokens(text: str, model: str | None = None) -> int:
    """Real tokenizer, not len(text)//4 — litellm ships one per provider."""
    if not text:
        return 0
    try:
        import litellm
        return litellm.token_counter(model=_litellm_model_name(model), text=text)
    except Exception:
        return max(1, len(text) // 4)  # last resort only if litellm itself is unavailable


def usd_saved(tokens_before: int, tokens_after: int, model: str | None = None) -> float:
    saved_tokens = max(0, tokens_before - tokens_after)
    return round(saved_tokens / 1_000_000 * _price_per_million(model), 6)


# ── Compression ─────────────────────────────────────────────────────────────

# A line is "structured" — an ASCII diagram border/row, a table row, or a
# markdown table separator — when whitespace inside it is part of the
# content, not incidental. Collapsing "|  Data (*)  |" to "| Data (*) |"
# doesn't change what it SAYS, but it does change how it LOOKS, and for a
# byte-layout diagram or a column-aligned table, how it looks is exactly
# the information it's there to convey. These lines are left untouched.
_BORDER_LINE = re.compile(r"^[\s+\-=|:]{3,}$")
_BOX_RUN = re.compile(r"[+\-]{3,}")


def _is_structured_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    # A single "|" is enough — diagram content rows for a field that
    # continues past the frame boundary (RFC-style "...") only ever have
    # the OPENING pipe, no closing one, so requiring a pair would miss them.
    return bool(_BORDER_LINE.match(stripped) or _BOX_RUN.search(stripped) or "|" in stripped)


def compress_context(text: str) -> str:
    """Essence-style compression for a whole context blob — same
    filler/hedging regexes compress_essence() uses for memory items, but
    without that function's 140-char cap (a multi-paragraph injection body
    isn't one telegraphic bullet) and without discarding code — memory
    items can drop code bodies since they're short summaries, but context
    being injected into a real conversation often carries file paths and
    identifiers the model still needs, so fenced/inline code is stashed and
    restored untouched instead of replaced with a placeholder.

    Whitespace collapsing runs line-by-line and skips any line that looks
    like an ASCII diagram border, a diagram content row, or a table row
    (see _is_structured_line) — a blanket regex across the whole text
    would flatten the deliberate column alignment in those lines into
    something that still contains the same words but no longer looks like
    the diagram/table it was."""
    if not text:
        return text

    protected: list[str] = []

    def _stash(m: re.Match) -> str:
        protected.append(m.group(0))
        return f"\x00{len(protected) - 1}\x00"

    working = _CODE_FENCE.sub(_stash, text)
    working = _INLINE_CODE.sub(_stash, working)

    for pattern in _FILLER_PATTERNS:
        working = pattern.sub("", working)
        # Collapse doubled whitespace after EACH pattern, not just once at
        # the end — back-to-back filler phrases (e.g. "the user said that"
        # right before "I think") leave a double space that breaks the
        # next pattern's \b word-boundary check, letting that filler
        # phrase survive untouched. (Line-aware, same as the pass below —
        # a filler phrase can't span the structured lines this protects
        # anyway, so this is just the same collapse applied early.)
        working = "\n".join(
            line if _is_structured_line(line) else re.sub(r"[ \t]{2,}", " ", line)
            for line in working.split("\n")
        )
    working = re.sub(r"\n{3,}", "\n\n", working)
    # Deliberately no .strip() here — this function runs per-CHUNK when the
    # caller is compress_and_verify()'s chunked path (see token_optimizer's
    # module docstring), and chunks are rejoined with a plain "".join().
    # Stripping a chunk's leading/trailing newline would glue it directly
    # onto its neighbor, silently merging two lines that were never meant
    # to be on the same line (this is exactly how an early version of this
    # function corrupted ASCII diagrams that happened to fall across a
    # chunk boundary).

    for i, original in enumerate(protected):
        working = working.replace(f"\x00{i}\x00", original)
    return working


def verify_compression(original: str, compressed: str, threshold: float | None = None) -> tuple[bool, float]:
    """Embeds both versions and compares. Fails closed: if the embedder is
    unavailable, meaning-preservation can't be verified at all, so the
    compression is treated as failed rather than trusted blind.

    Callers are responsible for keeping `original`/`compressed` under
    _EMBEDDING_MAX_TOKENS — this function does not chunk or truncate
    itself; see compress_and_verify() for the chunked caller that does."""
    threshold = MIN_SIMILARITY if threshold is None else threshold
    orig_vec = embed(original)
    comp_vec = embed(compressed)
    if orig_vec is None or comp_vec is None:
        return False, 0.0
    similarity = cosine_similarity(orig_vec, comp_vec)
    return similarity >= threshold, similarity


# content-hash -> (compressed_text, expires_at). Deliberately keyed on
# content only, NOT model — the same compressed text is valid input
# regardless of which model reads it next, so escalating from a cheap model
# to an expensive one on the same context reuses this cache instead of
# paying the compression+verification cost twice.
_compression_cache: dict[str, tuple[str, float]] = {}


def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _split_into_chunks(text: str, target_tokens: int) -> list[str]:
    """Greedy, line-based chunking: never splits a line in half (so an
    ASCII-diagram row or table row is never divided across two chunks),
    sized with a cheap chars-per-token proxy rather than a real tokenizer
    call per line — this only needs to be roughly the right size, the
    embedding-budget fitting in _build_verification_windows() is what
    actually guarantees correctness."""
    lines = text.splitlines(keepends=True)
    chunks: list[str] = []
    current: list[str] = []
    current_chars = 0
    target_chars = max(target_tokens, 1) * 4
    for line in lines:
        if current and current_chars + len(line) > target_chars:
            chunks.append("".join(current))
            current = []
            current_chars = 0
        current.append(line)
        current_chars += len(line)
    if current:
        chunks.append("".join(current))
    return chunks


def _tail(text: str, approx_tokens: int) -> str:
    chars = approx_tokens * 4
    return text[-chars:] if len(text) > chars else text


def _head(text: str, approx_tokens: int) -> str:
    chars = approx_tokens * 4
    return text[:chars] if len(text) > chars else text


def _build_verification_windows(original_core: str, compressed_core: str, before: str, after: str) -> tuple[str, str]:
    """Pads both sides of a chunk with neighboring ORIGINAL-text context for
    a fairer, in-context similarity comparison — but never past
    _EMBEDDING_MAX_TOKENS, since the embedder would just silently truncate
    there anyway (the exact blind spot this chunking exists to close). The
    fit is decided once, from the longer (original) side, then applied
    identically to the compressed side so both windows get the same
    padding and stay comparable."""
    budget = _EMBEDDING_MAX_TOKENS - 4
    for candidate_before, candidate_after in ((before, after), ("", after), ("", "")):
        if count_tokens(candidate_before + original_core + candidate_after) <= budget:
            return candidate_before + original_core + candidate_after, candidate_before + compressed_core + candidate_after

    # Even the bare core exceeds the embedding budget (shouldn't happen at
    # the default chunk size) — truncate both consistently rather than
    # silently rely on the embedder's own truncation to bail us out.
    core = original_core
    while count_tokens(core) > budget and len(core) > 1:
        core = core[: int(len(core) * 0.9)]
    return core, compressed_core[: len(core)]


def _compress_and_verify_whole(text: str, model: str | None) -> dict[str, Any]:
    """Single whole-text compress+verify, cached by content hash — used
    when `text` already fits under the embedder's token budget, so
    chunking would add complexity without adding any accuracy."""
    empty_result = {
        "text": text, "compressed": False, "similarity": 1.0,
        "tokens_before": 0, "tokens_after": 0, "usd_saved": 0.0,
    }
    key = _cache_key(text)
    now = time.time()
    cached = _compression_cache.get(key)
    if cached and cached[1] > now:
        compressed = cached[0]
    else:
        compressed = compress_context(text)
        if compressed != text:
            _compression_cache[key] = (compressed, now + CACHE_TTL_SECONDS)

    if compressed == text:
        return empty_result

    passed, similarity = verify_compression(text, compressed)
    if not passed:
        logger.info(f"[token_optimizer] compression discarded (similarity {similarity:.2f} < {MIN_SIMILARITY})")
        _compression_cache.pop(key, None)
        return {**empty_result, "similarity": similarity}

    tokens_before = count_tokens(text, model)
    tokens_after = count_tokens(compressed, model)
    return {
        "text": compressed,
        "compressed": True,
        "similarity": similarity,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "usd_saved": usd_saved(tokens_before, tokens_after, model),
    }


def _compress_and_verify_chunked(text: str, model: str | None) -> dict[str, Any]:
    """For text too long to embed in one shot: verify each chunk against
    its OWN compression independently (padded with neighboring original
    context — see _build_verification_windows), instead of embedding the
    whole document and only ever actually checking the first
    _EMBEDDING_MAX_TOKENS of it. A chunk that fails its own check keeps
    its ORIGINAL text — one bad chunk no longer discards every chunk's
    compression, it just doesn't compress that one."""
    original_chunks = _split_into_chunks(text, CHUNK_TARGET_TOKENS)
    final_parts: list[str] = []
    chunk_results: list[dict[str, Any]] = []

    for i, chunk in enumerate(original_chunks):
        compressed_chunk = compress_context(chunk)
        if compressed_chunk == chunk:
            final_parts.append(chunk)
            chunk_results.append({"index": i, "compressed": False, "similarity": 1.0})
            continue

        before = _tail(original_chunks[i - 1], CHUNK_OVERLAP_TOKENS) if i > 0 else ""
        after = _head(original_chunks[i + 1], CHUNK_OVERLAP_TOKENS) if i < len(original_chunks) - 1 else ""
        original_window, compressed_window = _build_verification_windows(chunk, compressed_chunk, before, after)

        passed, similarity = verify_compression(original_window, compressed_window)
        final_parts.append(compressed_chunk if passed else chunk)
        chunk_results.append({"index": i, "compressed": passed, "similarity": similarity})
        if not passed:
            logger.info(f"[token_optimizer] chunk {i} compression discarded (similarity {similarity:.2f} < {MIN_SIMILARITY}), kept original")

    final_text = "".join(final_parts)
    any_compressed = any(c["compressed"] for c in chunk_results)
    if not any_compressed:
        return {
            "text": text, "compressed": False,
            "similarity": min((c["similarity"] for c in chunk_results), default=1.0),
            "chunk_results": chunk_results, "tokens_before": 0, "tokens_after": 0, "usd_saved": 0.0,
        }

    tokens_before = count_tokens(text, model)
    tokens_after = count_tokens(final_text, model)
    return {
        "text": final_text,
        "compressed": True,
        # The headline similarity is the WORST chunk, not an average — one
        # weak chunk shouldn't be hidden by every other chunk being a
        # clean, easy 1.0.
        "similarity": min(c["similarity"] for c in chunk_results),
        "chunk_results": chunk_results,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "usd_saved": usd_saved(tokens_before, tokens_after, model),
    }


def compress_and_verify(text: str, model: str | None = None) -> dict[str, Any]:
    """The one entry point everything else should call: compress, verify
    with embeddings, price it in dollars. Never ships a compression that
    changed meaning — on a failed similarity check (or on text too short to
    bother with), `text` comes back untouched and `usd_saved` is 0.

    Texts that fit within the embedder's token budget are verified whole
    (and cached by content hash). Longer texts are chunked so every part
    of the document actually gets checked — see _compress_and_verify_chunked
    for why a single whole-document embedding can't do that on its own."""
    if not text or len(text) < MIN_COMPRESS_CHARS:
        return {"text": text, "compressed": False, "similarity": 1.0, "tokens_before": 0, "tokens_after": 0, "usd_saved": 0.0}

    if count_tokens(text, model) <= _EMBEDDING_MAX_TOKENS - 4:
        return _compress_and_verify_whole(text, model)
    return _compress_and_verify_chunked(text, model)


# ── Semantic cache for repeated LLM calls ──────────────────────────────────

class SemanticCache:
    """Short-lived cache of (query embedding -> response) pairs, scoped per
    caller-chosen namespace — e.g. "recap:{project}:{model}" — so a cached
    answer is never served to a different project/user, and a model
    upgrade naturally invalidates itself by landing in a different
    namespace instead of silently serving a cheaper model's old output."""

    def __init__(self, ttl_seconds: int | None = None, min_similarity: float | None = None):
        self._ttl = CACHE_TTL_SECONDS if ttl_seconds is None else ttl_seconds
        self._min_similarity = CACHE_MIN_SIMILARITY if min_similarity is None else min_similarity
        self._entries: dict[str, list[dict[str, Any]]] = {}

    def lookup(self, namespace: str, query: str) -> str | None:
        entries = self._entries.get(namespace)
        if not entries:
            return None
        now = time.time()
        entries[:] = [e for e in entries if e["expires_at"] > now]
        if not entries:
            return None
        vec = embed(query)
        if vec is None:
            return None
        best = max(entries, key=lambda e: cosine_similarity(vec, e["vec"]))
        if cosine_similarity(vec, best["vec"]) >= self._min_similarity:
            return best["response"]
        return None

    def store(self, namespace: str, query: str, response: str) -> None:
        vec = embed(query)
        if vec is None:
            return
        entries = self._entries.setdefault(namespace, [])
        entries.append({"vec": vec, "expires_at": time.time() + self._ttl, "response": response})
        # Bound memory use — this is a short-lived working-session cache,
        # not a durable store; drop the oldest half once a namespace grows
        # past a small cap instead of tracking access recency.
        if len(entries) > 50:
            del entries[: len(entries) - 50]
