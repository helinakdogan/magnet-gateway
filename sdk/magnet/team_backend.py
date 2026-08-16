"""
team_backend — the seam between mcp_server.py and team/plan enforcement
--------------------------------------------------------------------------
mcp_server.py's team-tool handlers (and the personal-memory cap check) call
ONLY through TeamBackend — never `import magnet.team_permissions`,
`magnet.team_store`, `magnet.postgres_store`, or `magnet.auth` directly.
Those four modules hold the paid moat (team coordination, plan/key
validation, hosted storage) and are private-repo-only; mcp_server.py must
still build and run without them installed at all.

Two implementations:
  - HostedRelayTeamBackend (hosted_client.py, public, the default) — every
    stdio/local process relays every op over HTTPS to the hosted server,
    authenticated by MAGNET_API_KEY. This is the real thin client: it POSTs
    and renders whatever comes back, with no plan/role/membership branching
    of its own.
  - DirectPostgresTeamBackend (private repo) — the hosted server registers
    this at startup via set_team_backend(), so ITS OWN process talks to
    Postgres directly instead of looping back over HTTP to itself.

Every method must fail closed: on any doubt (server unreachable, no
Postgres, expired plan, not a member) return the "denied" shape below —
never raise into a tool response, never guess "allowed". check_auto_promote
and check_memory_cap in particular must never surface an error to the
user — a denial there means "silently skip", not "fail the calling tool".

Return shapes (every op except the two check_* ones returns a plain dict):
  success   -> op-specific keys (see each method's docstring)
  failure   -> {"error": <slug>, "message": <user-facing text>}
The handler's job is only ever `if "error" in result: return result["message"]`
— it never re-derives *why* something was denied.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TeamBackend(Protocol):
    # ── Team coordination ───────────────────────────────────────────────
    def create_team(self, user_id: str, team_name: str) -> dict:
        """{"team": {"id", "name", "plan", ...}} or {"error", "message"}."""
        ...

    def join_team(self, user_id: str, team_id: str) -> dict:
        """{"team": {...}} or {"error", "message"}."""
        ...

    def add_member(self, actor_user_id: str, team_id: str, new_user_id: str) -> dict:
        """{"ok": bool, "message": str}."""
        ...

    def list_members(self, user_id: str, team_id: str) -> dict:
        """{"team": {...}, "members": [{"user_id","role","joined_at"}]} or {"error","message"}."""
        ...

    # ── Shared project data ──────────────────────────────────────────────
    def load_team_items(self, user_id: str, team_id: str, project: str) -> list[dict]:
        """The raw shared items for `project`, to merge with personal items
        for display (recall/recap/show_project_memory). Runs on every one of
        those calls when a team_id is active, so unlike every other method
        here this NEVER returns an error shape — any denial, unreachable
        server, or "not shared" case returns [] silently, falling back to
        personal-only memory exactly as if no team were configured."""
        ...

    def list_shared_projects(self, user_id: str, team_id: str) -> dict:
        """{"shared_projects": [{"project","item_count","shared_by"}]} or {"error","message"}."""
        ...

    def share_project(self, user_id: str, team_id: str, project: str, items: list[dict]) -> dict:
        """{"shared": int, "team_id", "project"} or {"error", "message"}."""
        ...

    def share_item(self, user_id: str, team_id: str, project: str, item_id: str, item: dict) -> dict:
        """{"shared": 1, "item", "category"} or {"already_shared": True, "text"} or {"error","message"}."""
        ...

    def get_team_memory(self, user_id: str, team_id: str, project: str, explicit_project: bool) -> dict:
        """If explicit_project is True, always returns {"display_text": str}
        (or {"error","message"}) for exactly the requested project — an
        unshared project just displays as empty, matching format_team_display's
        own "no shared memory yet" text; never triggers auto-discovery.

        If explicit_project is False, one of:
        {"display_text": str}                                    — `project` (the local active one) is shared
        {"ambiguous": True, "shared_projects": [...]}             — caller should show a menu
        {"auto_selected_project": str, "display_text": str}       — one shared project, different from asked
        {"none_shared": True}                                     — nothing shared yet in this team
        {"error", "message"}                                      — denied/unreachable
        """
        ...

    def get_history(self, user_id: str, team_id: str, item_id: str | None) -> dict:
        """{"history": [...]} or {"error", "message"}."""
        ...

    def get_team_status(self, user_id: str, team_id: str, project: str) -> dict:
        """{"name","plan","member_count","project_shared": bool} or {"error","message"}."""
        ...

    # ── Fire-and-forget checks — never surface an error, only a boolean/None ──
    def check_auto_promote(self, user_id: str, team_id: str, project: str, item: dict) -> bool:
        """True if the item was auto-promoted to team memory. Any denial
        (not a live paid member, unreachable server, nothing to agree with)
        returns False silently — never raises, never blocks the personal
        `remember` save this rides alongside."""
        ...

    def check_memory_cap(self, user_id: str) -> str | None:
        """Deny message if `user_id` is at/over their plan's personal memory
        cap, else None (allowed). Local/stdio is always unlimited — only a
        registered hosted backend enforces this."""
        ...

    def record_memory_delta(self, user_id: str, delta: int) -> None:
        """Adjust the maintained per-user memory-item counter. Best-effort,
        fire-and-forget — must never raise."""
        ...

    # ── Write-approval flow ────────────────────────────────────────────
    #
    # A lead can restrict a specific (member, project, category) combination
    # so matching writes queue for review instead of landing directly in
    # team memory — opt-in per restriction, not a per-team toggle. With no
    # restriction covering a given write, share_item/share_project/
    # check_auto_promote behave exactly as they did before this existed.

    def request_team_write(self, user_id: str, team_id: str, project: str, item: dict) -> dict:
        """Explicit member-facing request for review — the same outcome
        share_item/share_project fall into automatically when a restriction
        matches, but callable directly even without one.
        {"request_id", "status": "pending"} or {"error", "message"}."""
        ...

    def list_pending_requests(self, user_id: str, team_id: str, project: str | None = None) -> dict:
        """Owner/lead-only. {"requests": [{"id","project","requested_by",
        "category","text","created_at","conflict": {...}|None}, ...]} or
        {"error","message"}. Each request's `conflict` — the most similar
        existing team item in the same category/project, if any — is
        surfaced so a lead never approves blind; it never blocks the call."""
        ...

    def approve_request(self, user_id: str, team_id: str, request_id: str) -> dict:
        """Owner/lead-only. Writes the item into team memory (reusing
        share_item's path, attributed to the ORIGINAL requester) and marks
        the request approved. {"ok": True, "item", "conflict"} or
        {"error","message"}."""
        ...

    def reject_request(self, user_id: str, team_id: str, request_id: str) -> dict:
        """Owner/lead-only. The item never enters team memory.
        {"ok": True} or {"error","message"}."""
        ...

    def list_my_requests(self, user_id: str, team_id: str) -> dict:
        """Any member — their own requests only (any status), never another
        member's pending requests. {"requests": [...]}."""
        ...


_backend: TeamBackend | None = None


def set_team_backend(backend: TeamBackend) -> None:
    """Called once, at process startup, by whichever server registers a
    non-default backend (today: only the hosted HTTP server, before it
    starts serving requests). Never called by stdio, which always uses the
    lazily-constructed default below."""
    global _backend
    _backend = backend


def get_team_backend() -> TeamBackend:
    """Returns the registered backend, or lazily constructs and caches the
    default HostedRelayTeamBackend (stdio/local processes never call
    set_team_backend, so this default is what makes team commands work
    there at all — by relaying to the hosted server, never locally)."""
    global _backend
    if _backend is None:
        from magnet.hosted_client import HostedRelayTeamBackend

        _backend = HostedRelayTeamBackend()
    return _backend
