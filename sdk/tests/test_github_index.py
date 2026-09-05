import base64
import os
from unittest.mock import patch

import pytest

from magnet import github_index as gi


def test_parse_repo_url_full_https():
    assert gi.parse_repo_url("https://github.com/helinakdogan/agentmagnet") == ("helinakdogan", "agentmagnet")


def test_parse_repo_url_with_git_suffix_and_trailing_slash():
    assert gi.parse_repo_url("https://github.com/helinakdogan/agentmagnet.git/") == ("helinakdogan", "agentmagnet")


def test_parse_repo_url_bare_owner_repo():
    assert gi.parse_repo_url("helinakdogan/agentmagnet") == ("helinakdogan", "agentmagnet")


def test_parse_repo_url_rejects_garbage():
    with pytest.raises(gi.GithubFetchError):
        gi.parse_repo_url("not a repo at all")


def test_build_index_fetches_tree_once_and_labels_markdown_with_heading():
    tree_response = {
        "tree": [
            {"path": "README.md", "type": "blob"},
            {"path": "magnet/mcp_server.py", "type": "blob"},
            {"path": "docs", "type": "tree"},  # directories are skipped
            {"path": "docs/architecture.md", "type": "blob"},
        ],
        "truncated": False,
    }

    def fake_get_json(url, token=None):  # noqa: ARG001
        if url.endswith("/repos/owner/repo"):
            return {"default_branch": "main"}
        return tree_response

    def fake_get_raw_text(url, max_bytes, token=None):  # noqa: ARG001
        if url.endswith("README.md"):
            return "# Agent Magnet\n\nSome intro text."
        if url.endswith("architecture.md"):
            return "no heading here, just prose"
        return None

    with (
        patch.object(gi, "_get_json", side_effect=fake_get_json),
        patch.object(gi, "_get_raw_text", side_effect=fake_get_raw_text),
    ):
        index = gi.build_index("owner", "repo")

    assert index["owner"] == "owner"
    assert index["repo"] == "repo"
    assert index["branch"] == "main"
    assert index["file_count"] == 3  # the "docs" tree entry is excluded
    assert index["has_token"] is False
    assert index["token_enc"] is None

    by_path = {f["path"]: f for f in index["files"]}
    assert by_path["README.md"]["label"] == "Agent Magnet"  # real heading
    assert by_path["README.md"]["type"] == "md"
    assert by_path["docs/architecture.md"]["label"] == "architecture.md"  # no heading -> filename fallback
    assert by_path["magnet/mcp_server.py"]["label"] == "mcp_server.py"  # code file, no fetch attempted
    assert all(f["fetched"] is False for f in index["files"])


def test_build_index_caps_markdown_heading_fetches():
    tree_response = {
        "tree": [{"path": f"doc{i}.md", "type": "blob"} for i in range(5)],
        "truncated": False,
    }
    fetch_calls = {"n": 0}

    def fake_get_raw_text(url, max_bytes, token=None):  # noqa: ARG001
        fetch_calls["n"] += 1
        return "# Heading"

    with (
        patch.object(gi, "MD_HEADING_FETCH_LIMIT", 2),
        patch.object(gi, "_get_json", side_effect=lambda url, token=None: {"default_branch": "main"} if url.endswith("/repos/o/r") else tree_response),
        patch.object(gi, "_get_raw_text", side_effect=fake_get_raw_text),
    ):
        index = gi.build_index("o", "r")

    assert fetch_calls["n"] == 2  # capped, even though there are 5 markdown files
    assert index["file_count"] == 5


def test_build_index_with_token_uses_contents_api_for_headings_not_raw_cdn():
    """Private repos can't reliably use the unauthenticated raw CDN even
    with a token, so when a token is given, heading reads must go through
    fetch_file() (Contents API), not _get_raw_text()."""
    tree_response = {"tree": [{"path": "README.md", "type": "blob"}], "truncated": False}
    raw_text_called = {"n": 0}

    def fake_get_json(url, token=None):
        if url.endswith("/repos/o/r"):
            return {"default_branch": "main"}
        if "/git/trees/" in url:
            return tree_response
        if "/contents/" in url:
            assert token == "ghp_secret"
            return {"encoding": "base64", "content": base64.b64encode(b"# Private Heading").decode()}
        raise AssertionError(f"unexpected url {url}")

    def fake_get_raw_text(*a, **k):
        raw_text_called["n"] += 1
        return None

    with (
        patch.object(gi, "_get_json", side_effect=fake_get_json),
        patch.object(gi, "_get_raw_text", side_effect=fake_get_raw_text),
    ):
        index = gi.build_index("o", "r", token="ghp_secret")

    assert raw_text_called["n"] == 0
    assert index["files"][0]["label"] == "Private Heading"
    assert index["has_token"] is True


def test_find_relevant_files_ranks_by_label_with_real_embeddings():
    index = {
        "files": [
            {"path": "magnet/team_store.py", "label": "Team memory storage", "fetched": False},
            {"path": "magnet/local_embeddings.py", "label": "Local embeddings", "fetched": False},
            {"path": "README.md", "label": "Agent Magnet", "fetched": False},
        ]
    }
    matches = gi.find_relevant_files(index, "how does team memory work", top_k=2)

    paths = [m["path"] for m in matches]
    assert "magnet/team_store.py" in paths


def test_find_relevant_files_keyword_fallback_when_embedder_unavailable():
    index = {
        "files": [
            {"path": "magnet/team_store.py", "label": "Team memory storage", "fetched": False},
            {"path": "magnet/local_embeddings.py", "label": "Local embeddings module", "fetched": False},
        ]
    }
    with patch("magnet.local_embeddings._get_embedder", return_value=None):
        matches = gi.find_relevant_files(index, "team memory", top_k=1)

    assert [m["path"] for m in matches] == ["magnet/team_store.py"]


def test_fetch_file_uses_contents_api_and_decodes_base64():
    captured = {}

    def fake_get_json(url, token=None):
        captured["url"] = url
        captured["token"] = token
        return {"encoding": "base64", "content": base64.b64encode(b"file contents").decode()}

    with patch.object(gi, "_get_json", side_effect=fake_get_json):
        result = gi.fetch_file("owner", "repo", "main", "src/app.py", token="ghp_abc")

    assert result == "file contents"
    assert captured["url"] == "https://api.github.com/repos/owner/repo/contents/src/app.py?ref=main"
    assert captured["token"] == "ghp_abc"


def test_fetch_file_returns_none_on_github_error():
    with patch.object(gi, "_get_json", side_effect=gi.GithubFetchError("404")):
        assert gi.fetch_file("owner", "repo", "main", "missing.py") is None


def test_fetch_file_truncates_to_max_bytes():
    big_content = b"x" * 100
    with patch.object(gi, "_get_json", return_value={"encoding": "base64", "content": base64.b64encode(big_content).decode()}):
        result = gi.fetch_file("o", "r", "main", "f.txt", max_bytes=10)

    assert result == "x" * 10


class _FakeBackend:
    def __init__(self):
        self._data: dict[str, str] = {}

    def get(self, key):
        return self._data.get(key)

    def set(self, key, value, ex=None):  # noqa: ARG002
        self._data[key] = value

    def delete(self, key):
        self._data.pop(key, None)


def test_save_and_load_index_round_trip():
    backend = _FakeBackend()
    index = {"owner": "o", "repo": "r", "branch": "main", "files": []}

    gi.save_index(backend, "user1", index)
    loaded = gi.load_index(backend, "user1", "o", "r")

    assert loaded == index
    assert gi.load_index(backend, "user1", "o", "other-repo") is None


def test_active_repo_round_trip():
    backend = _FakeBackend()
    assert gi.get_active_repo(backend, "user1", "myproject") is None

    gi.set_active_repo(backend, "user1", "myproject", "owner", "repo")

    assert gi.get_active_repo(backend, "user1", "myproject") == ("owner", "repo")
    assert gi.get_active_repo(backend, "user1", "other-project") is None


def test_clear_active_repo():
    backend = _FakeBackend()
    gi.set_active_repo(backend, "user1", "myproject", "owner", "repo")
    gi.clear_active_repo(backend, "user1", "myproject")
    assert gi.get_active_repo(backend, "user1", "myproject") is None


def test_mark_fetched_sets_flag_and_timestamp():
    index = {"files": [{"path": "a.py", "fetched": False, "fetched_at": None}, {"path": "b.py", "fetched": False, "fetched_at": None}]}
    gi.mark_fetched(index, "a.py")

    a, b = index["files"]
    assert a["fetched"] is True
    assert a["fetched_at"] is not None
    assert b["fetched"] is False


# ── Token encryption (fail-closed, mirrors team_permissions' pattern) ──────

@pytest.fixture(autouse=True)
def _reset_fernet_singleton():
    gi._fernet = None
    yield
    gi._fernet = None


def test_encrypt_token_returns_none_without_encryption_key():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("MAGNET_ENCRYPTION_KEY", None)
        assert gi.encrypt_token("ghp_secret") is None


def test_encrypt_decrypt_round_trip():
    from cryptography.fernet import Fernet
    key = Fernet.generate_key().decode()
    with patch.dict(os.environ, {"MAGNET_ENCRYPTION_KEY": key}):
        enc = gi.encrypt_token("ghp_secret_token")
        assert enc is not None
        assert enc != "ghp_secret_token"
        assert gi.decrypt_token(enc) == "ghp_secret_token"


def test_decrypt_token_returns_none_on_garbage():
    from cryptography.fernet import Fernet
    key = Fernet.generate_key().decode()
    with patch.dict(os.environ, {"MAGNET_ENCRYPTION_KEY": key}):
        assert gi.decrypt_token("not-a-valid-fernet-token") is None


# ── Refresh (rebuild tree, keep fetched status for surviving files) ────────

def test_merge_fetched_status_preserves_existing_fetched_files():
    old_index = {"files": [
        {"path": "a.py", "fetched": True, "fetched_at": 123.0},
        {"path": "b.py", "fetched": False, "fetched_at": None},
    ]}
    new_index = {"files": [
        {"path": "a.py", "fetched": False, "fetched_at": None},
        {"path": "b.py", "fetched": False, "fetched_at": None},
        {"path": "c.py", "fetched": False, "fetched_at": None},  # new file
    ]}

    merged = gi.merge_fetched_status(new_index, old_index)

    by_path = {f["path"]: f for f in merged["files"]}
    assert by_path["a.py"]["fetched"] is True
    assert by_path["a.py"]["fetched_at"] == 123.0
    assert by_path["b.py"]["fetched"] is False
    assert by_path["c.py"]["fetched"] is False


def test_merge_fetched_status_noop_without_old_index():
    new_index = {"files": [{"path": "a.py", "fetched": False}]}
    assert gi.merge_fetched_status(new_index, None) == new_index


def test_refresh_index_reuses_stored_token_and_preserves_fetched():
    backend = _FakeBackend()
    from cryptography.fernet import Fernet
    key = Fernet.generate_key().decode()

    with patch.dict(os.environ, {"MAGNET_ENCRYPTION_KEY": key}):
        old_index = {
            "owner": "o", "repo": "r", "branch": "main", "file_count": 1,
            "files": [{"path": "a.py", "label": "a.py", "type": "code", "fetched": True, "fetched_at": 999.0}],
            "token_enc": gi.encrypt_token("ghp_secret"), "has_token": True,
        }
        gi.save_index(backend, "user1", old_index)

        rebuilt = {
            "owner": "o", "repo": "r", "branch": "main", "file_count": 1,
            "files": [{"path": "a.py", "label": "a.py", "type": "code", "fetched": False, "fetched_at": None}],
            "token_enc": None, "has_token": False,
        }

        captured_token = {}

        def fake_build_index(owner, repo, token=None):  # noqa: ARG001
            captured_token["value"] = token
            return rebuilt

        with patch.object(gi, "build_index", side_effect=fake_build_index):
            result = gi.refresh_index(backend, "user1", "o", "r")

    assert captured_token["value"] == "ghp_secret"  # reused without re-asking
    assert result["files"][0]["fetched"] is True  # preserved across refresh
    assert result["has_token"] is True

    reloaded = gi.load_index(backend, "user1", "o", "r")
    assert reloaded["files"][0]["fetched"] is True
