"""
github_index
------------
Context7-style GitHub repo connection: index lightly upfront, fetch content
on demand — never a bulk upload of a whole repo into memory at once.

  connect  -> build_index()        one tree API call + a capped number of
                                    raw fetches for markdown headings only.
                                    Stores path + short label per file, not
                                    file content. A Personal Access Token,
                                    if given, is encrypted before storage
                                    and used for both the tree call and
                                    every file fetch (required for private
                                    repos, optional for public ones).
  ask      -> find_relevant_files() ranks the index against a question
                                    (embedding similarity, reusing
                                    local_embeddings.rank_by_similarity — no
                                    second relevance mechanism), then only
                                    the matched file(s) get fetch_file()'d
                                    and run through
                                    token_optimizer.compress_and_verify()
                                    before becoming a memory item.
  cache    -> each index entry tracks fetched/fetched_at; mcp_server.py's
              _handle_github_recall skips re-fetching a file that's already
              fetched, unless refresh=True is explicitly requested.

Uses stdlib urllib only, matching team_permissions.py's own convention for
one-off HTTP calls (no new HTTP client dependency). Token encryption mirrors
team_permissions.py's encrypt_redis_url/decrypt_redis_url exactly (same
MAGNET_ENCRYPTION_KEY env var, same Fernet library, same fail-closed
behavior) — kept here rather than imported from there because
team_permissions.py is hosted/private-repo-only, and a locally-run stdio
user connecting their own private repo needs this exactly as much as a
hosted team does.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"

# Building the light index still fetches one raw request per markdown file
# (just to read its heading) — capped so "connect" stays fast/cheap even on
# a repo with hundreds of markdown files. Every non-markdown file costs
# zero extra requests (its label is just its filename).
MD_HEADING_FETCH_LIMIT = int(os.environ.get("MAGNET_GITHUB_MD_LIMIT", "40"))

# A fetched file's raw content is capped too — this is a context/memory
# item, not a file browser; a multi-megabyte generated file wouldn't fit a
# model's context regardless, and compress_and_verify() only needs the text
# to be readable, not complete to the last byte.
MAX_FILE_BYTES = int(os.environ.get("MAGNET_GITHUB_MAX_FILE_BYTES", "200000"))

_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
_REPO_URL_RE = re.compile(r"github\.com[:/]([^/]+)/([^/.\s]+?)(?:\.git)?/?$")


class GithubFetchError(Exception):
    pass


def parse_repo_url(text: str) -> tuple[str, str]:
    """Accepts a full GitHub URL (https://github.com/owner/repo, with or
    without .git / a trailing slash) or a bare "owner/repo" string."""
    text = text.strip()
    if "github.com" not in text and re.fullmatch(r"[\w.-]+/[\w.-]+", text):
        owner, repo = text.split("/", 1)
        return owner, repo
    m = _REPO_URL_RE.search(text)
    if not m:
        raise GithubFetchError(f"Could not parse a GitHub owner/repo from '{text}'.")
    return m.group(1), m.group(2)


# ── Token encryption (mirrors team_permissions.encrypt_redis_url exactly) ──

_fernet = None


def _get_fernet():
    """Lazily built, process-wide Fernet instance from MAGNET_ENCRYPTION_KEY.
    Returns None if the env var isn't set — callers must fail closed, never
    store a token in plaintext."""
    global _fernet
    if _fernet is None:
        key = os.environ.get("MAGNET_ENCRYPTION_KEY", "")
        if not key:
            return None
        from cryptography.fernet import Fernet
        _fernet = Fernet(key.encode("utf-8") if isinstance(key, str) else key)
    return _fernet


def encrypt_token(token: str) -> str | None:
    f = _get_fernet()
    if f is None:
        return None
    return f.encrypt(token.encode("utf-8")).decode("utf-8")


def decrypt_token(enc: str) -> str | None:
    f = _get_fernet()
    if f is None:
        return None
    try:
        return f.decrypt(enc.encode("utf-8")).decode("utf-8")
    except Exception as e:
        logger.error(f"[github] failed to decrypt token: {type(e).__name__}: {e}")
        return None


# ── GitHub HTTP ──────────────────────────────────────────────────────────────

def _request(url: str, headers: dict[str, str], token: str | None) -> Any:
    request = urllib.request.Request(url, headers=headers)
    tok = token or os.environ.get("MAGNET_GITHUB_TOKEN")
    if tok:
        request.add_header("Authorization", f"Bearer {tok}")
    return urllib.request.urlopen(request, timeout=15)  # noqa: S310 — fixed https hosts only


def _get_json(url: str, token: str | None = None) -> Any:
    try:
        with _request(url, {"Accept": "application/vnd.github+json", "User-Agent": "agent-magnet"}, token) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 404):
            raise GithubFetchError(
                f"GitHub returned {e.code} for {url} — for a private repo, check the "
                "token is valid and has 'repo' (or fine-grained Contents: Read) access."
            ) from e
        raise GithubFetchError(f"GitHub API request failed ({e.code}) for {url}") from e
    except urllib.error.URLError as e:
        raise GithubFetchError(f"GitHub API unreachable: {e}") from e


def _get_raw_text(url: str, max_bytes: int, token: str | None = None) -> str | None:
    try:
        with _request(url, {"User-Agent": "agent-magnet"}, token) as response:
            return response.read(max_bytes).decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        logger.debug(f"[github] raw fetch failed for {url}: {e}")
        return None


def _default_branch(owner: str, repo: str, token: str | None) -> str:
    data = _get_json(f"{GITHUB_API}/repos/{owner}/{repo}", token=token)
    return data.get("default_branch") or "main"


def _extract_heading(text: str, fallback: str) -> str:
    m = _HEADING_RE.search(text)
    return m.group(1).strip() if m else fallback


def _raw_url(owner: str, repo: str, branch: str, path: str) -> str:
    # Public convenience path only — see fetch_file()'s docstring for why
    # actual file fetches go through the Contents API instead.
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"


def build_index(owner: str, repo: str, token: str | None = None) -> dict:
    """The LIGHT index: file tree (path + short label) only — never file
    content. One tree API call regardless of repo size (recursive=1
    returns everything at once); markdown files each cost one small raw
    fetch (capped at MD_HEADING_FETCH_LIMIT) just to read a heading, code
    files cost nothing extra. `token` (plaintext, already decrypted by the
    caller if it came from storage) is required for a private repo — same
    token used for every call in this function and passed to fetch_file()
    later for the on-demand step."""
    branch = _default_branch(owner, repo, token)
    tree = _get_json(f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{branch}?recursive=1", token=token)
    blobs = [e for e in tree.get("tree", []) if e.get("type") == "blob"]

    md_fetched = 0
    files: list[dict] = []
    for entry in blobs:
        path = entry["path"]
        is_md = path.lower().endswith(".md")
        label = path.rsplit("/", 1)[-1]
        if is_md and md_fetched < MD_HEADING_FETCH_LIMIT:
            # Private repos can't use the unauthenticated raw CDN even with
            # a token in hand reliably, so headings go through fetch_file()
            # (Contents API) too when a token is set; public repos keep
            # using the cheaper raw URL directly.
            raw = fetch_file(owner, repo, branch, path, token=token, max_bytes=2000) if token else \
                _get_raw_text(_raw_url(owner, repo, branch, path), max_bytes=2000)
            if raw:
                label = _extract_heading(raw, fallback=label)
            md_fetched += 1
        files.append({
            "path": path,
            "label": label,
            "type": "md" if is_md else "code",
            "fetched": False,
            "fetched_at": None,
        })

    logger.info(f"[github] indexed {owner}/{repo}@{branch}: {len(files)} files, {md_fetched} headings read")
    return {
        "owner": owner,
        "repo": repo,
        "branch": branch,
        "indexed_at": time.time(),
        "truncated": bool(tree.get("truncated")),
        "file_count": len(files),
        "files": files,
        "token_enc": encrypt_token(token) if token else None,
        "has_token": bool(token),
    }


def merge_fetched_status(new_index: dict, old_index: dict | None) -> dict:
    """Used on refresh: rebuilding the tree gives every file a fresh
    fetched=False, which would make _handle_github_recall re-fetch (and
    re-spend tokens re-compressing) content that's still valid in memory.
    Carries fetched/fetched_at over from the previous index for any path
    that still exists — new/renamed files correctly start unfetched."""
    if not old_index:
        return new_index
    old_by_path = {f["path"]: f for f in old_index.get("files", [])}
    for f in new_index["files"]:
        old = old_by_path.get(f["path"])
        if old and old.get("fetched"):
            f["fetched"] = True
            f["fetched_at"] = old.get("fetched_at")
    return new_index


def find_relevant_files(index: dict, question: str, top_k: int = 3) -> list[dict]:
    """Ranks the index's (path + label) entries against a question — same
    embedding-similarity-with-keyword-fallback ranking already used for
    forget/mark_done fuzzy search, not a second relevance mechanism."""
    from magnet.local_embeddings import rank_by_similarity
    return rank_by_similarity(question, index["files"], text_key="label", top_k=top_k)


def fetch_file(owner: str, repo: str, branch: str, path: str, token: str | None = None, max_bytes: int = MAX_FILE_BYTES) -> str | None:
    """The on-demand fetch — only ever called for a file find_relevant_files()
    already picked out, never for the whole tree.

    Goes through the Contents API (base64-decoded) rather than
    raw.githubusercontent.com: the Contents API is the officially
    documented, reliably-authenticated way to read a private file with a
    PAT, and it works identically for public repos too (with or without a
    token), so there's one code path instead of two."""
    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    try:
        data = _get_json(url, token=token)
    except GithubFetchError as e:
        logger.debug(f"[github] contents fetch failed for {owner}/{repo}:{path}: {e}")
        return None
    if not isinstance(data, dict) or data.get("encoding") != "base64":
        return None
    try:
        raw_bytes = base64.b64decode(data["content"])
    except Exception as e:
        logger.debug(f"[github] base64 decode failed for {owner}/{repo}:{path}: {e}")
        return None
    return raw_bytes[:max_bytes].decode("utf-8", errors="replace")


# ── Storage (index + connected-repo pointer) ───────────────────────────────

def _index_key(user_id: str, owner: str, repo: str) -> str:
    return f"github:index:{user_id}:{owner}/{repo}"


def _active_key(user_id: str, project: str) -> str:
    return f"github:active:{user_id}:{project}"


def save_index(backend: Any, user_id: str, index: dict) -> None:
    backend.set(_index_key(user_id, index["owner"], index["repo"]), json.dumps(index, ensure_ascii=False))


def load_index(backend: Any, user_id: str, owner: str, repo: str) -> dict | None:
    raw = backend.get(_index_key(user_id, owner, repo))
    return json.loads(raw) if raw else None


def delete_index(backend: Any, user_id: str, owner: str, repo: str) -> None:
    if hasattr(backend, "delete"):
        backend.delete(_index_key(user_id, owner, repo))
    else:
        backend.set(_index_key(user_id, owner, repo), "")


def set_active_repo(backend: Any, user_id: str, project: str, owner: str, repo: str) -> None:
    backend.set(_active_key(user_id, project), f"{owner}/{repo}")


def get_active_repo(backend: Any, user_id: str, project: str) -> tuple[str, str] | None:
    raw = backend.get(_active_key(user_id, project))
    if not raw or "/" not in raw:
        return None
    owner, repo = raw.split("/", 1)
    return owner, repo


def clear_active_repo(backend: Any, user_id: str, project: str) -> None:
    if hasattr(backend, "delete"):
        backend.delete(_active_key(user_id, project))
    else:
        backend.set(_active_key(user_id, project), "")


def refresh_index(backend: Any, user_id: str, owner: str, repo: str) -> dict:
    """Rebuilds the tree (picks up new/renamed/deleted files) while
    preserving fetched/fetched_at for files that already made it into
    memory, and reuses the previously-stored (encrypted) token if any —
    the dashboard's "Refresh" action, and also usable by a future
    '*connect github <repo> --refresh' without needing the token retyped."""
    old_index = load_index(backend, user_id, owner, repo)
    token = decrypt_token(old_index["token_enc"]) if old_index and old_index.get("token_enc") else None

    new_index = build_index(owner, repo, token)
    new_index["token_enc"] = old_index.get("token_enc") if old_index else None
    new_index["has_token"] = bool(new_index["token_enc"])
    merge_fetched_status(new_index, old_index)

    save_index(backend, user_id, new_index)
    return new_index


def mark_fetched(index: dict, path: str) -> None:
    for f in index["files"]:
        if f["path"] == path:
            f["fetched"] = True
            f["fetched_at"] = time.time()
            return
