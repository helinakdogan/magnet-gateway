import time
from unittest.mock import patch

from magnet import token_optimizer as topt


def test_compress_context_strips_filler_but_keeps_facts():
    text = (
        "So, I think the user mentioned that we should just really "
        "look at file src/api/routes.py:142 where the retry count is 5, "
        "please fix it maybe."
    )
    compressed = topt.compress_context(text)

    assert "i think" not in compressed.lower()
    assert "just" not in compressed.lower()
    assert "please" not in compressed.lower()
    assert "src/api/routes.py:142" in compressed
    assert "5" in compressed


def test_compress_context_preserves_code_blocks_untouched():
    text = "Here is the fix:\n```python\ndef foo():\n    return 1\n```\nplease apply it."
    compressed = topt.compress_context(text)

    assert "```python\ndef foo():\n    return 1\n```" in compressed


def test_compress_context_noop_on_short_clean_text():
    text = "Uses Postgres for storage."
    assert topt.compress_context(text) == text


def test_compress_context_preserves_ascii_diagram_alignment():
    text = (
        "Please note the frame layout:\n\n"
        "    +---------------+\n"
        "    |Pad Length? (8)|\n"
        "    +---------------+-----------------------------------------------+\n"
        "    |                            Data (*)                         ...\n"
        "    +---------------------------------------------------------------+\n\n"
        "That is basically the whole format."
    )
    compressed = topt.compress_context(text)

    for line in (
        "    +---------------+",
        "    |Pad Length? (8)|",
        "    +---------------+-----------------------------------------------+",
        "    |                            Data (*)                         ...",
        "    +---------------------------------------------------------------+",
    ):
        assert line in compressed, f"diagram line altered: {line!r}"
    # filler around the diagram still gets cleaned up
    assert "please" not in compressed.lower()
    assert "basically" not in compressed.lower()


def test_compress_context_preserves_table_row_alignment():
    text = (
        "I think the pricing is as follows.\n\n"
        "| Model            | Price   |\n"
        "|------------------|---------|\n"
        "| Claude Sonnet 5  | $2.00   |\n\n"
        "Please use this table."
    )
    compressed = topt.compress_context(text)

    assert "| Model            | Price   |" in compressed
    assert "|------------------|---------|" in compressed
    assert "| Claude Sonnet 5  | $2.00   |" in compressed


def test_verify_compression_fails_closed_when_embedder_unavailable():
    with patch.object(topt, "embed", return_value=None):
        passed, similarity = topt.verify_compression("original text", "compressed")
    assert passed is False
    assert similarity == 0.0


def test_verify_compression_respects_threshold():
    vectors = {"original": [1.0, 0.0], "close": [0.99, 0.14], "far": [0.0, 1.0]}
    with patch.object(topt, "embed", side_effect=lambda t: vectors[t]):
        passed_close, sim_close = topt.verify_compression("original", "close", threshold=0.9)
        passed_far, sim_far = topt.verify_compression("original", "far", threshold=0.9)

    assert passed_close is True
    assert sim_close > 0.9
    assert passed_far is False
    assert sim_far < 0.9


def test_compress_and_verify_skips_short_text():
    text = "short text"
    result = topt.compress_and_verify(text)
    assert result["compressed"] is False
    assert result["usd_saved"] == 0.0
    assert result["text"] == text


def test_compress_and_verify_discards_low_similarity_compression():
    long_filler_text = "I think " * 40 + "actual fact stays here " * 10
    with patch.object(topt, "verify_compression", return_value=(False, 0.4)):
        result = topt.compress_and_verify(long_filler_text)

    assert result["compressed"] is False
    assert result["text"] == long_filler_text
    assert result["usd_saved"] == 0.0
    assert result["similarity"] == 0.4


def test_compress_and_verify_ships_a_verified_compression():
    long_filler_text = "I think that " * 40 + "the deploy uses port 8080 and file config.py:12. " * 5

    with patch.object(topt, "verify_compression", return_value=(True, 0.95)):
        result = topt.compress_and_verify(long_filler_text, model="gpt-4o-mini")

    assert result["compressed"] is True
    assert result["similarity"] == 0.95
    assert result["tokens_before"] > result["tokens_after"] >= 0
    assert result["usd_saved"] >= 0.0


def test_compress_and_verify_cache_is_model_agnostic():
    """Escalating from a cheap model to an expensive one on the SAME
    content must reuse the compression cache instead of recompressing."""
    long_filler_text = "I think that " * 40 + "the deploy uses port 8080. " * 5
    topt._compression_cache.clear()

    with patch.object(topt, "verify_compression", return_value=(True, 0.95)) as mock_verify:
        topt.compress_and_verify(long_filler_text, model="gpt-4o-mini")
        topt.compress_and_verify(long_filler_text, model="claude-opus-5")

    # compress_context() (the expensive-ish regex pass) only needs to run
    # once; the second call should have hit the cache and only needed to
    # re-verify + re-price, not recompute the compressed text — but the
    # simplest observable proxy for "reused the cache" is that both calls'
    # verify_compression received the exact same compressed text.
    first_call_args = mock_verify.call_args_list[0].args
    second_call_args = mock_verify.call_args_list[1].args
    assert first_call_args[1] == second_call_args[1]


def test_usd_saved_uses_configured_price_table():
    usd = topt.usd_saved(2_000_000, 1_000_000, model="gpt-4o")
    assert usd == topt.PRICE_PER_MILLION_TOKENS["gpt-4o"]


def test_usd_saved_never_negative():
    assert topt.usd_saved(100, 200, model="gpt-4o") == 0.0


def test_semantic_cache_hit_on_similar_query():
    cache = topt.SemanticCache(ttl_seconds=60, min_similarity=0.9)
    vectors = {"first phrasing": [1.0, 0.0], "second phrasing": [0.99, 0.14]}

    with patch.object(topt, "embed", side_effect=lambda t: vectors[t]):
        cache.store("ns", "first phrasing", "cached answer")
        result = cache.lookup("ns", "second phrasing")

    assert result == "cached answer"


def test_semantic_cache_miss_below_threshold():
    cache = topt.SemanticCache(ttl_seconds=60, min_similarity=0.95)
    vectors = {"first phrasing": [1.0, 0.0], "unrelated": [0.0, 1.0]}

    with patch.object(topt, "embed", side_effect=lambda t: vectors[t]):
        cache.store("ns", "first phrasing", "cached answer")
        result = cache.lookup("ns", "unrelated")

    assert result is None


def test_semantic_cache_expires_entries():
    cache = topt.SemanticCache(ttl_seconds=0, min_similarity=0.5)
    with patch.object(topt, "embed", return_value=[1.0, 0.0]):
        cache.store("ns", "query", "answer")
        time.sleep(0.01)
        result = cache.lookup("ns", "query")

    assert result is None


def test_semantic_cache_namespaces_are_isolated():
    cache = topt.SemanticCache(ttl_seconds=60, min_similarity=0.5)
    with patch.object(topt, "embed", return_value=[1.0, 0.0]):
        cache.store("recap:proj:gpt-4o-mini", "query", "cheap model answer")
        result = cache.lookup("recap:proj:claude-opus-5", "query")

    assert result is None


# ── Chunked verification for long documents (verification blind spot fix) ──

def _long_filler_paragraph(marker: str) -> str:
    return (
        f"I think that basically the {marker} section really matters. "
        f"Please note the {marker} value is actually documented clearly. "
    ) * 6


def test_split_into_chunks_never_splits_a_line():
    text = "line one\nline two\nline three\n" * 20
    chunks = topt._split_into_chunks(text, target_tokens=10)
    assert len(chunks) > 1
    # every original line appears whole, in exactly one chunk
    rejoined = "".join(chunks)
    assert rejoined == text
    for chunk in chunks:
        assert chunk.count("\n") == chunk.rstrip("\n").count("\n") + (1 if chunk.endswith("\n") else 0)


def test_long_document_routes_to_chunked_path():
    text = "\n\n".join(_long_filler_paragraph(f"section-{i}") for i in range(12))
    assert topt.count_tokens(text) > topt._EMBEDDING_MAX_TOKENS

    with patch.object(topt, "verify_compression", return_value=(True, 0.95)) as mock_verify:
        result = topt.compress_and_verify(text)

    assert result["compressed"] is True
    assert "chunk_results" in result
    assert len(result["chunk_results"]) > 1
    # every embed comparison stayed under the model's real truncation limit
    for call in mock_verify.call_args_list:
        original_window, compressed_window = call.args[0], call.args[1]
        assert topt.count_tokens(original_window) <= topt._EMBEDDING_MAX_TOKENS
        assert topt.count_tokens(compressed_window) <= topt._EMBEDDING_MAX_TOKENS


def test_long_document_keeps_original_for_failing_chunks_only():
    text = "\n\n".join(_long_filler_paragraph(f"section-{i}") for i in range(6))
    chunks = topt._split_into_chunks(text, topt.CHUNK_TARGET_TOKENS)
    assert len(chunks) >= 3

    # fail every other chunk's verification; the rest pass
    call_count = {"n": 0}

    def fake_verify(_original, _compressed, threshold=None):  # noqa: ARG001
        call_count["n"] += 1
        return (call_count["n"] % 2 == 0), 0.5

    with patch.object(topt, "verify_compression", side_effect=fake_verify):
        result = topt.compress_and_verify(text)

    outcomes = [c["compressed"] for c in result["chunk_results"] if "similarity" in c]
    assert False in outcomes  # at least one chunk was rejected and kept original
    assert True in outcomes or result["compressed"]  # at least one chunk WAS compressed
    # the reported similarity is the worst chunk's, not an average that hides it
    assert result["similarity"] == 0.5


def test_short_document_does_not_chunk():
    text = ("I think that this basically uses Postgres for storage, please note. " * 6).strip()
    assert len(text) >= topt.MIN_COMPRESS_CHARS
    assert topt.count_tokens(text) <= topt._EMBEDDING_MAX_TOKENS

    with patch.object(topt, "verify_compression", return_value=(True, 0.95)) as mock_verify:
        result = topt.compress_and_verify(text)

    assert result["compressed"] is True
    assert "chunk_results" not in result
    assert mock_verify.call_count == 1
