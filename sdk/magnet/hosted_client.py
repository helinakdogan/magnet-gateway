"""
hosted_client — HostedRelayTeamBackend, the public thin-client implementation
--------------------------------------------------------------------------------
Local/stdio processes never hold Postgres credentials (see
team_permissions.py's module docstring — that module is deliberately
Postgres-only and unusable outside our hosted deployment). This module is
the ONLY way a stdio process can perform team operations: over HTTPS,
authenticated by the user's own MAGNET_API_KEY, against the hosted server's
/api/team/op endpoint (http_server.py), which re-validates the key and
re-checks the plan/membership server-side on every single call.

HostedRelayTeamBackend implements the TeamBackend protocol (team_backend.py)
by doing exactly one thing per method: POST an op, return whatever JSON body
came back. No plan/role/membership branching happens here — that would
defeat the point. FAIL CLOSED, always: any network error, timeout, or
unparsable response is treated as a denial, via the same {"error","message"}
shape a real server denial would use — never an implicit "allowed", and
never a crash into the calling tool handler.

MAGNET_REDIS_URL, if set, only ever affects WHERE team data is stored once
permission has already been granted server-side — it has zero bearing on
any decision made in this file.

Uses stdlib urllib only — no new dependency, since this must work in the
plain `pip install agent-magnet` (no [postgres] extra) case that's the
normal stdio/local install.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

_DEFAULT_HOSTED_URL = "https://mcp.agentmagnet.app"
_TIMEOUT = 8.0

_UNREACHABLE = {
    "error": "unreachable",
    "message": "Could not reach the hosted server. Try again shortly.",
}
_NO_KEY = {
    "error": "no_key",
    "message": "Team memory requires an Agent Magnet key. Set MAGNET_API_KEY. Get one at agentmagnet.app.",
}


def _hosted_base_url() -> str:
    return os.environ.get("MAGNET_HOSTED_URL", _DEFAULT_HOSTED_URL).rstrip("/")


def _post(path: str, payload: dict) -> dict | None:
    """POST JSON to the hosted server. Returns the parsed JSON body on ANY
    response we can parse (2xx or not — our endpoints return a JSON body
    even on 401/402/404/503, and callers need that body to pick the right
    deny message), or None if the server is unreachable / times out / sends
    something we can't parse at all."""
    url = f"{_hosted_base_url()}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read())
        except Exception:
            logger.warning(f"[hosted_client] {path} failed: HTTP {e.code}")
            return None
    except Exception as e:
        logger.warning(f"[hosted_client] {path} unreachable: {type(e).__name__}: {e}")
        return None


class HostedRelayTeamBackend:
    """The default TeamBackend for every stdio/local process. Reads
    MAGNET_API_KEY once at construction — matching stdio's "one user per
    process" model (mcp_server.py's module docstring) — and sends it with
    every op. If no key is configured, every method fails closed without
    making a network call at all."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = (api_key if api_key is not None else os.environ.get("MAGNET_API_KEY", "")).strip()

    def _op(self, op: str, **fields: object) -> dict:
        if not self._api_key:
            return dict(_NO_KEY)
        result = _post("/api/team/op", {"key": self._api_key, "op": op, **fields})
        if result is None:
            return dict(_UNREACHABLE)
        return result

    # ── Team coordination ───────────────────────────────────────────────

    def create_team(self, user_id: str, team_name: str) -> dict:  # noqa: ARG002 — identity comes from the key server-side
        return self._op("create_team", name=team_name)

    def join_team(self, user_id: str, team_id: str) -> dict:  # noqa: ARG002
        return self._op("join_team", team_id=team_id)

    def add_member(self, actor_user_id: str, team_id: str, new_user_id: str) -> dict:  # noqa: ARG002
        return self._op("add_member", team_id=team_id, new_user_id=new_user_id)

    def list_members(self, user_id: str, team_id: str) -> dict:  # noqa: ARG002
        return self._op("list_team_members", team_id=team_id)

    # ── Shared project data ──────────────────────────────────────────────

    def load_team_items(self, user_id: str, team_id: str, project: str) -> list[dict]:  # noqa: ARG002
        result = self._op("load_team_items", team_id=team_id, project=project)
        items = result.get("items")
        return items if isinstance(items, list) else []

    def list_shared_projects(self, user_id: str, team_id: str) -> dict:  # noqa: ARG002
        return self._op("list_team_projects", team_id=team_id)

    def share_project(self, user_id: str, team_id: str, project: str, items: list[dict]) -> dict:  # noqa: ARG002
        return self._op("share_project_to_team", team_id=team_id, project=project, items=items)

    def share_item(self, user_id: str, team_id: str, project: str, item_id: str, item: dict) -> dict:  # noqa: ARG002
        return self._op("share_item_to_team", team_id=team_id, project=project, item_id=item_id, item=item)

    def get_team_memory(self, user_id: str, team_id: str, project: str, explicit_project: bool) -> dict:  # noqa: ARG002
        return self._op(
            "get_team_memory", team_id=team_id, project=project, explicit_project=explicit_project
        )

    def get_history(self, user_id: str, team_id: str, item_id: str | None) -> dict:  # noqa: ARG002
        return self._op("history", team_id=team_id, item_id=item_id)

    def get_team_status(self, user_id: str, team_id: str, project: str) -> dict:  # noqa: ARG002
        return self._op("get_team_status", team_id=team_id, project=project)

    # ── Fire-and-forget checks ──────────────────────────────────────────

    def check_auto_promote(self, user_id: str, team_id: str, project: str, item: dict) -> bool:  # noqa: ARG002
        result = self._op("check_auto_promote", team_id=team_id, project=project, item=item)
        return bool(result.get("promoted"))

    def check_memory_cap(self, user_id: str) -> str | None:  # noqa: ARG002
        # Personal storage cost is the user's own disk in stdio/local mode —
        # never capped, and never worth a network round trip on every write.
        # Only a registered DirectPostgresTeamBackend (hosted process)
        # enforces a real cap; see mcp_server._memory_cap_check.
        return None

    def record_memory_delta(self, user_id: str, delta: int) -> None:  # noqa: ARG002
        # No-op for the same reason as check_memory_cap above — nothing to
        # meter locally.
        return None

    # ── Write-approval flow ──────────────────────────────────────────────

    def request_team_write(self, user_id: str, team_id: str, project: str, item: dict) -> dict:  # noqa: ARG002
        return self._op("request_team_write", team_id=team_id, project=project, item=item)

    def list_pending_requests(self, user_id: str, team_id: str, project: str | None = None) -> dict:  # noqa: ARG002
        return self._op("list_pending_requests", team_id=team_id, project=project)

    def approve_request(self, user_id: str, team_id: str, request_id: str) -> dict:  # noqa: ARG002
        return self._op("approve_request", team_id=team_id, request_id=request_id)

    def reject_request(self, user_id: str, team_id: str, request_id: str) -> dict:  # noqa: ARG002
        return self._op("reject_request", team_id=team_id, request_id=request_id)

    def list_my_requests(self, user_id: str, team_id: str) -> dict:  # noqa: ARG002
        return self._op("list_my_requests", team_id=team_id)
