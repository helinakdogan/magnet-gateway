"""
UsageCounter
------------
Billing groundwork: meters memory writes and retrieval calls.
Does NOT enforce limits — metering only.

Storage:
  - Local (no Redis): ~/.agent-magnet/usage.json
  - Redis: hash at  magnet:usage:{user_id}

For hosted HTTP mode, module-level check_usage_limit()/record_usage_event()/
get_hosted_usage_summary() below read/write the usage_events table in
Postgres (see postgres_store.py) — a per-tool-call event log, distinct from
this class's older hincrby counters (which keep working unchanged for
stdio mode). The TODO — future enforcement point — lives on
check_usage_limit(): it currently always returns True (allowed). When
billing goes live, plug a Stripe subscription/quota check there. Local
stdio mode never calls it at all and is always unlimited.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LOCAL_FILE = Path.home() / ".agent-magnet" / "usage.json"
_REDIS_PREFIX = "magnet:usage:"
_SAVINGS_REDIS_PREFIX = "magnet:savings:"


class UsageCounter:
    def __init__(self, redis_client: Any = None, user_id: str = "default"):
        self._redis = redis_client
        self._user_id = user_id

    # ── Write ─────────────────────────────────────────────────────────────────

    def record_write(self, project_id: str = "default") -> None:
        self._inc(f"writes:{project_id}")
        self._inc("writes:total")

    def record_retrieval(self, project_id: str = "default") -> None:
        self._inc(f"retrievals:{project_id}")
        self._inc("retrievals:total")

    def record_team_write(self, team_id: str, project_id: str = "default") -> None:
        """Meter team memory writes — for future Pro tier billing."""
        self._inc(f"team_writes:{team_id}:{project_id}")
        self._inc("team_writes:total")

    def record_team_recall(self, team_id: str, project_id: str = "default") -> None:
        """Meter team memory reads — for future Pro tier billing."""
        self._inc(f"team_recalls:{team_id}:{project_id}")
        self._inc("team_recalls:total")

    def _inc(self, metric: str) -> None:
        if self._redis:
            try:
                self._redis.hincrby(f"{_REDIS_PREFIX}{self._user_id}", metric, 1)
            except Exception:
                self._inc_local(metric)
        else:
            self._inc_local(metric)

    def _inc_local(self, metric: str) -> None:
        try:
            data: dict = {}
            if _LOCAL_FILE.exists():
                data = json.loads(_LOCAL_FILE.read_text(encoding="utf-8"))
            user = data.setdefault(self._user_id, {})
            user[metric] = user.get(metric, 0) + 1
            _LOCAL_FILE.parent.mkdir(parents=True, exist_ok=True)
            _LOCAL_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.debug(f"[usage] local increment failed: {e}")

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        if self._redis:
            try:
                raw = self._redis.hgetall(f"{_REDIS_PREFIX}{self._user_id}")
                if raw:
                    return {k: int(v) for k, v in raw.items()}
            except Exception:
                pass
        try:
            if _LOCAL_FILE.exists():
                data = json.loads(_LOCAL_FILE.read_text(encoding="utf-8"))
                return data.get(self._user_id, {})
        except Exception:
            pass
        return {}

    # ── Token-optimization savings — separate storage from the plain
    # metrics above (get_stats() casts every value to int, which would
    # truncate a fractional dollar amount to 0) ────────────────────────────

    def record_savings(self, tokens_saved: int, usd: float) -> None:
        if tokens_saved <= 0 and usd <= 0:
            return
        if self._redis:
            try:
                self._redis.hincrby(f"{_SAVINGS_REDIS_PREFIX}{self._user_id}", "tokens_saved", tokens_saved)
                self._redis.hincrbyfloat(f"{_SAVINGS_REDIS_PREFIX}{self._user_id}", "usd_saved", usd)
                return
            except Exception:
                pass
        self._record_savings_local(tokens_saved, usd)

    def _record_savings_local(self, tokens_saved: int, usd: float) -> None:
        try:
            data: dict = {}
            if _LOCAL_FILE.exists():
                data = json.loads(_LOCAL_FILE.read_text(encoding="utf-8"))
            key = f"{self._user_id}:savings"
            savings = data.setdefault(key, {"tokens_saved": 0, "usd_saved": 0.0})
            savings["tokens_saved"] = savings.get("tokens_saved", 0) + tokens_saved
            savings["usd_saved"] = savings.get("usd_saved", 0.0) + usd
            _LOCAL_FILE.parent.mkdir(parents=True, exist_ok=True)
            _LOCAL_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.debug(f"[usage] local savings record failed: {e}")

    def get_savings(self) -> dict:
        if self._redis:
            try:
                raw = self._redis.hgetall(f"{_SAVINGS_REDIS_PREFIX}{self._user_id}")
                if raw:
                    return {
                        "tokens_saved": int(float(raw.get("tokens_saved", 0))),
                        "usd_saved": round(float(raw.get("usd_saved", 0.0)), 6),
                    }
            except Exception:
                pass
        try:
            if _LOCAL_FILE.exists():
                data = json.loads(_LOCAL_FILE.read_text(encoding="utf-8"))
                saved = data.get(f"{self._user_id}:savings")
                if saved:
                    return {"tokens_saved": int(saved.get("tokens_saved", 0)), "usd_saved": round(float(saved.get("usd_saved", 0.0)), 6)}
        except Exception:
            pass
        return {"tokens_saved": 0, "usd_saved": 0.0}

    # ── Enforcement hook (TODO: plug tier limits here) ────────────────────────

    def check_usage_limit(self, metric: str = "writes:total") -> bool:  # noqa: ARG002
        """
        Always returns True (allowed) — enforcement not yet active.

        FUTURE: if MAGNET_API_KEY is set (hosted mode), fetch quota from Magnet API
        and return False when exceeded. Local mode (no API key) is always unlimited.
        """
        return True


# ── Hosted mode (HTTP) — event log + enforcement seam ─────────────────────────
#
# These are module-level, not UsageCounter methods, because they're keyed by
# (user_id, team_id) resolved fresh per HTTP request from the validated API
# key — not bound to a single cached instance the way UsageCounter used to be.
#
# postgres_store.py is private-repo-only (it moves out with team_permissions.py/
# auth.py/http_server.py — see the repo-split plan); a plain `pip install
# agent-magnet` never has it on disk at all. _pool_if_configured() below is
# the one place that boundary is crossed: it treats "module not installed"
# exactly like "not configured" (both are legitimate "hosted mode isn't this
# process" signals), so every function here stays a no-op in the public
# package instead of raising ModuleNotFoundError.


def _pool_if_configured() -> Any:
    try:
        from magnet.postgres_store import get_pool_if_configured
    except ImportError:
        return None
    return get_pool_if_configured()


def check_usage_limit(user_id: str, team_id: str = "") -> bool:  # noqa: ARG001
    """
    ALWAYS returns True today — metering only, no enforcement.

    TODO: BILLING HOOK — plug a Stripe subscription/quota check here. This is
    the single seam where future enforcement reads the user's/team's plan
    (api_keys.plan, or a live Stripe lookup) plus the aggregated usage_events
    counts for the current billing period (see get_hosted_usage_summary
    below) and returns False once a paid quota is exceeded. Called once per
    authenticated HTTP request from http_server.py's auth middleware, before
    the request is allowed to reach the MCP transport. Never called in
    stdio mode (local/free tier is always unlimited).
    """
    return True


def record_usage_event(
    user_id: str,
    team_id: str,
    event_type: str,
    key_id: str | None = None,
    tokens_saved: int | None = None,
    usd_saved: float | None = None,
) -> None:
    """Best-effort INSERT into usage_events. No-op outside hosted/Postgres
    mode. Metering must never break a tool call — all failures are swallowed
    (logged at debug level only). key_id (the mg_sk_... key that made this
    call, if any) is optional and used only for per-key usage breakdowns —
    never for identity/authorization. tokens_saved/usd_saved are optional
    and only set by token-optimization events (compression, cache hits) —
    every other event type leaves them NULL."""
    pool = _pool_if_configured()
    if pool is None:
        return
    try:
        with pool.connection() as conn:
            conn.execute(
                "INSERT INTO usage_events (user_id, team_id, event_type, key_id, tokens_saved, usd_saved) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (user_id, team_id or None, event_type, key_id, tokens_saved, usd_saved),
            )
    except Exception as e:
        logger.debug(f"[usage] record_usage_event failed: {e}")


def get_usd_saved_this_month(user_id: str) -> float:
    """SUM(usd_saved) across every token-optimization event this calendar
    month — the headline number for *usage and the dashboard's Usage tab.
    Returns 0.0 (not None) outside hosted/Postgres mode so callers can
    always display a number rather than branch on availability."""
    pool = _pool_if_configured()
    if pool is None:
        return 0.0
    try:
        with pool.connection() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(usd_saved), 0) FROM usage_events "
                "WHERE user_id = %s AND created_at >= date_trunc('month', now())",
                (user_id,),
            ).fetchone()
        return round(float(row[0]), 6) if row else 0.0
    except Exception as e:
        logger.debug(f"[usage] get_usd_saved_this_month failed: {e}")
        return 0.0


def get_hosted_usage_summary(user_id: str, team_id: str = "") -> dict | None:  # noqa: ARG001
    """
    Returns {event_type: count} for the current calendar month, or None if
    not running in hosted/Postgres mode. Feeds get_status's HTTP response
    (plan, memories stored, reads/writes this period).

    Excludes "token_optimization:*" rows — those are metadata about an
    ALREADY-counted tool call (e.g. a "recall" row plus a
    "token_optimization:recall_injection" row for the same request), not a
    second billable request. Counting them here would double-count sync
    usage against the plan's request cap.
    """
    pool = _pool_if_configured()
    if pool is None:
        return None
    try:
        with pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT event_type, COUNT(*) FROM usage_events
                WHERE user_id = %s AND created_at >= date_trunc('month', now())
                  AND event_type NOT LIKE 'token_optimization:%%'
                GROUP BY event_type
                """,
                (user_id,),
            ).fetchall()
        return {event_type: count for event_type, count in rows}
    except Exception as e:
        logger.debug(f"[usage] get_hosted_usage_summary failed: {e}")
        return None


def get_usage_by_key(user_id: str) -> dict[str, int]:
    """Returns {key_id: count} of validated calls this calendar month, for
    every one of this user's keys that has made at least one call — feeds
    the dashboard's per-key usage breakdown (API Keys tab / Usage tab).
    Rows recorded before key_id existed, or from dashboard actions that
    aren't tied to a specific mg_sk_... key, have key_id IS NULL and are
    excluded here (there's no key to attribute them to)."""
    pool = _pool_if_configured()
    if pool is None:
        return {}
    try:
        with pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT key_id, COUNT(*) FROM usage_events
                WHERE user_id = %s AND key_id IS NOT NULL
                  AND created_at >= date_trunc('month', now())
                GROUP BY key_id
                """,
                (user_id,),
            ).fetchall()
        return {str(key_id): count for key_id, count in rows}
    except Exception as e:
        logger.debug(f"[usage] get_usage_by_key failed: {e}")
        return {}
