"""
LocalEmbedder
-------------
On-device semantic embeddings using sentence-transformers (all-MiniLM-L6-v2).
No API key required. Model is downloaded once to ~/.agent-magnet/models/ on first use.

Falls back to keyword-overlap matching if sentence-transformers is not installed
or the model cannot be downloaded (offline first run).
"""

from __future__ import annotations

import logging
import math
import re
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"
_MODEL_DIR = Path.home() / ".agent-magnet" / "models"

# Normalized MiniLM cosine scores below this point are too weak to justify
# injecting a past episode into the current prompt. The keyword fallback uses
# a different scale and therefore only requires at least one overlapping term.
_MIN_EMBEDDING_SIMILARITY = 0.25

_embedder: Any = None
_embedder_available: bool | None = None  # None = not yet tried


_EMBEDDER_INIT_TIMEOUT_S = 15.0


def _get_embedder() -> Any:
    global _embedder, _embedder_available
    if _embedder_available is not None:
        return _embedder  # already resolved (None if unavailable)

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        if not (_MODEL_DIR / _MODEL_NAME).exists():
            sys.stderr.write(
                "[agent-magnet] Downloading local embedding model (one-time, ~90 MB)...\n"
            )
            sys.stderr.flush()
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # Constructing SentenceTransformer downloads the model from
        # HuggingFace Hub when it isn't already cached in _MODEL_DIR, with
        # no timeout of its own — on a host where that download stalls
        # (blocked/unreliable egress, slow DNS), this call can hang for
        # minutes. Since this runs inline in live request paths
        # (memory_requests' conflict-check, is_semantic_duplicate on every
        # personal-memory write, rank_by_similarity on forget/mark_done
        # fuzzy search) and even in a startup warm-up thread, an unbounded
        # hang here doesn't just make ONE call slow — sustained CPU/network
        # contention while it hangs is enough to make a small host miss its
        # own health checks and get killed as "unhealthy" by Render.
        #
        # Bound it with a plain daemon thread + join(timeout=...), NOT
        # ThreadPoolExecutor — a `with ThreadPoolExecutor(...)` block calls
        # shutdown(wait=True) on exit unconditionally, which blocks until
        # the submitted work finishes regardless of any timeout passed to
        # future.result(), completely defeating the purpose (this shipped
        # once already and made the hang worse, not bounded — a thread pool
        # someone can't walk away from isn't a timeout). A daemon thread's
        # join(timeout=...) returns after the timeout no matter what the
        # thread is still doing, and being daemonic means it never blocks
        # process exit either.
        import threading

        _result: dict[str, Any] = {}

        def _build_embedder() -> None:
            try:
                # show_progress_bar is an encode()-time argument in current
                # sentence-transformers, not a constructor kwarg — passing it
                # here raised a TypeError on newer versions, which silently
                # downgraded every caller to the keyword-overlap fallback.
                _result["model"] = SentenceTransformer(_MODEL_NAME, cache_folder=str(_MODEL_DIR))
            except Exception as e:
                _result["error"] = e

        thread = threading.Thread(target=_build_embedder, daemon=True)
        thread.start()
        thread.join(timeout=_EMBEDDER_INIT_TIMEOUT_S)

        if thread.is_alive():
            logger.warning(
                f"[magnet] Local embedder init exceeded {_EMBEDDER_INIT_TIMEOUT_S}s "
                "(likely a stalled model download) — keyword fallback active. "
                "The download thread is abandoned (daemon), not cancelled; it may "
                "still finish in the background, but this process won't wait on it again."
            )
            _embedder_available = False
        elif "error" in _result:
            raise _result["error"]
        else:
            _embedder = _result["model"]
            _embedder_available = True
            logger.info(f"[magnet] Local embedder ready: {_MODEL_NAME}")
    except ImportError:
        logger.info(
            "[magnet] sentence-transformers not installed — "
            "using keyword fallback. Install with: pip install sentence-transformers"
        )
        _embedder_available = False
    except Exception as e:
        logger.warning(
            f"[magnet] Local embedder failed ({e}) — keyword fallback active."
        )
        _embedder_available = False

    return _embedder


def embed(text: str) -> list[float] | None:
    """Return a normalized 384-dim vector, or None if unavailable."""
    model = _get_embedder()
    if model is None:
        return None
    try:
        return model.encode(text, normalize_embeddings=True, show_progress_bar=False).tolist()
    except Exception as e:
        logger.debug(f"[embedder] encode failed: {e}")
        return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two pre-normalized vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def is_semantic_duplicate(text: str, candidates: list[str], threshold: float = 0.90) -> bool:
    """Return True if `text` is semantically near-identical to any candidate."""
    if not candidates:
        return False

    model = _get_embedder()
    if model is None:
        # Keyword fallback: Jaccard similarity on words
        text_words = set(re.findall(r"\w+", text.lower()))
        for c in candidates:
            c_words = set(re.findall(r"\w+", c.lower()))
            union = text_words | c_words
            if not union:
                continue
            if len(text_words & c_words) / len(union) > 0.65:
                return True
        return False

    try:
        vec = embed(text)
        if vec is None:
            return False
        for c in candidates:
            c_vec = embed(c)
            if c_vec and cosine_similarity(vec, c_vec) >= threshold:
                return True
    except Exception:
        pass
    return False


def _rank_by_keyword(
    query: str,
    documents: list[dict],
    text_key: str,
    top_k: int,
) -> list[dict]:
    """Rank by word overlap and exclude documents with no shared terms."""
    q_words = set(re.findall(r"\w+", query.lower()))
    scored: list[tuple[float, dict]] = []
    for doc in documents:
        doc_text = doc.get(text_key, "").lower()
        doc_words = set(re.findall(r"\w+", doc_text))
        hits = sum(1 for word in q_words if word in doc_words)
        score = hits / (math.log(max(len(doc_words), 1)) + 1)
        scored.append((score, doc))
    scored.sort(key=lambda item: -item[0])
    return [doc for score, doc in scored if score > 0][:top_k]


def rank_by_similarity(
    query: str,
    documents: list[dict],
    text_key: str = "text",
    top_k: int = 5,
) -> list[dict]:
    """
    Rank documents by similarity to query.
    Uses embedding vectors when available; keyword overlap when not.
    """
    if not documents:
        return []

    model = _get_embedder()

    if model is None:
        return _rank_by_keyword(query, documents, text_key, top_k)

    try:
        q_vec = embed(query)
        if q_vec is None:
            return _rank_by_keyword(query, documents, text_key, top_k)
        scored_vecs: list[tuple[float, dict]] = []
        for doc in documents:
            d_vec = doc.get("_embedding") or embed(doc.get(text_key, ""))
            if d_vec:
                scored_vecs.append((cosine_similarity(q_vec, d_vec), doc))
            else:
                scored_vecs.append((0.0, doc))
        scored_vecs.sort(key=lambda x: -x[0])
        return [
            d for score, d in scored_vecs
            if score >= _MIN_EMBEDDING_SIMILARITY
        ][:top_k]
    except Exception as e:
        logger.debug(f"[embedder] rank_by_similarity failed: {e}")
        return _rank_by_keyword(query, documents, text_key, top_k)
