"""
MemoryStore
-----------
Stores project memory under a clean three-level hierarchy:

  helin (user)
  ├── personal (profile)
  │   ├── general (project)  →  key: vmm:helin:personal:general
  │   └── kuika   (project)  →  key: vmm:helin:personal:kuika
  └── hobby (profile)
      └── side-thing         →  key: vmm:helin:hobby:side-thing

Key format:   vmm:{user}:{profile}:{project}
Value:        JSON {"items": [{id, category, text, status, confidence, stored_at,
              source_tool, source_transport, created_by, updated_by, updated_at}]}

Index key:    vmm:{user}:__index__
Value:        JSON {"personal": ["general", "kuika"], "hobby": ["side-thing"]}

ProjectStore is an alias for backward compat with existing imports.
"""

from __future__ import annotations

import json
import logging
import random
import string
import time
from typing import Any

logger = logging.getLogger(__name__)

_MAX_ENTRIES = 200
_TTL = 60 * 60 * 24 * 90  # 90 days

CATEGORIES = frozenset({"action", "decision", "convention", "watch_out", "tried_failed", "goal", "preference"})

_LABELS: dict[str, str] = {
    "action":       "Actions",
    "decision":     "Decisions",
    "convention":   "Conventions",
    "watch_out":    "Watch out",
    "tried_failed": "Tried & failed",
    "goal":         "Goals",
    "preference":   "Preferences",
}

# Display order matches the spec example (decisions first, preferences last).
# Actions lead — what was actually DONE is the highest-value context when
# resuming work, more reliable than statements of intent.
_DISPLAY_ORDER = ["action", "decision", "convention", "watch_out", "tried_failed", "goal", "preference"]
# Injection order: high-value context first (actions + decisions + goals + watch-outs), preferences last
_INJECT_ORDER  = ["action", "decision", "goal", "watch_out", "tried_failed", "convention", "preference"]

_ID_CHARS = string.ascii_lowercase + string.digits


def _n(s: str) -> str:
    """Normalize a name: lowercase + strip."""
    return s.strip().lower()


def _gen_id() -> str:
    """Generate a short 6-char alphanumeric ID."""
    return "".join(random.choices(_ID_CHARS, k=6))


def _add_missing_fields(items: list[dict]) -> bool:
    """Migrate old items that predate id/status, or provenance/attribution
    fields (source_tool, source_transport, created_by, updated_by,
    updated_at). Legacy items get "unknown" rather than a guessed value —
    same fail-closed convention used everywhere else in this codebase.
    Returns True if any fields were added."""
    changed = False
    for item in items:
        if "id" not in item:
            item["id"] = _gen_id()
            changed = True
        if "status" not in item:
            item["status"] = "active"
            changed = True
        if "source_tool" not in item:
            item["source_tool"] = "unknown"
            changed = True
        if "source_transport" not in item:
            item["source_transport"] = "unknown"
            changed = True
        if "created_by" not in item:
            item["created_by"] = "unknown"
            changed = True
        if "updated_by" not in item:
            item["updated_by"] = item.get("created_by", "unknown")
            changed = True
        if "updated_at" not in item:
            item["updated_at"] = item.get("stored_at")
            changed = True
    return changed


class MemoryStore:
    def __init__(self, redis_client: Any = None):
        self._redis = redis_client
        self._mem: dict[str, Any] = {}

    # ── Key helpers ──────────────────────────────────────────────────────────────

    def _key(self, user: str, profile: str, project: str) -> str:
        return f"vmm:{_n(user)}:{_n(profile)}:{_n(project)}"

    def _index_key(self, user: str) -> str:
        return f"vmm:{_n(user)}:__index__"

    # ── Profile / project index ───────────────────────────────────────────────────

    def _load_index(self, user: str) -> dict[str, list[str]]:
        key = self._index_key(user)
        if self._redis:
            raw = self._redis.get(key)
            return json.loads(raw) if raw else {}
        return dict(self._mem.get(key, {}))

    def _save_index(self, user: str, index: dict[str, list[str]]) -> None:
        key = self._index_key(user)
        data = json.dumps(index, ensure_ascii=False)
        if self._redis:
            self._redis.set(key, data)
        else:
            self._mem[key] = dict(index)

    def create_profile(self, user: str, name: str) -> bool:
        """Add a profile. Returns True if newly created."""
        p = _n(name)
        index = self._load_index(user)
        if p in index:
            return False
        index[p] = []
        self._save_index(user, index)
        logger.info(f"[memory_store] created profile '{p}' for {user}")
        return True

    def create_project(self, user: str, profile: str, name: str) -> bool:
        """Add a project under a profile. Creates profile if absent. Returns True if newly created."""
        p, pr = _n(profile), _n(name)
        index = self._load_index(user)
        projects = index.setdefault(p, [])
        if pr in projects:
            return False
        projects.append(pr)
        self._save_index(user, index)
        logger.info(f"[memory_store] created project '{p}/{pr}' for {user}")
        return True

    def list_profiles(self, user: str) -> list[tuple[str, int]]:
        """Return [(profile_name, project_count)] sorted alphabetically."""
        index = self._load_index(user)
        return sorted((p, len(pjs)) for p, pjs in index.items())

    def list_projects(self, user: str, profile: str) -> list[str]:
        """Return project names in a profile, in insertion order."""
        index = self._load_index(user)
        return list(index.get(_n(profile), []))

    # ── Memory entries ───────────────────────────────────────────────────────────

    def load(self, user: str, profile: str, project: str) -> list[dict]:
        key = self._key(user, profile, project)
        if self._redis:
            raw = self._redis.get(key)
            if not raw:
                return []
            data = json.loads(raw)
            items = data.get("items", []) if isinstance(data, dict) else data
        else:
            stored = self._mem.get(key, {})
            items = list(stored.get("items", []))

        # Lazily migrate old items that predate id/status fields
        if _add_missing_fields(items):
            self._save(user, profile, project, items)
        return items

    def _save(self, user: str, profile: str, project: str, items: list[dict]) -> None:
        key = self._key(user, profile, project)
        trimmed = items[-_MAX_ENTRIES:]
        payload = json.dumps({"items": trimmed}, ensure_ascii=False)
        if self._redis:
            self._redis.setex(key, _TTL, payload)
        else:
            self._mem[key] = {"items": trimmed}

    def add_entry(
        self,
        user: str,
        profile: str,
        project: str,
        category: str,
        text: str,
        confidence: float = 0.8,
        dedup: bool = True,
        source_tool: str = "unknown",
        source_transport: str = "unknown",
    ) -> bool:
        """Add a memory item. Returns True if saved (False if empty or duplicate).

        source_tool/source_transport are provenance — which MCP client and
        which transport this item came from ("claude"/"cursor"/"codex"/
        "unknown", "stdio"/"http"/"unknown"). Callers should pass "unknown"
        rather than guess when the actual source isn't determinable.
        """
        if not text or not text.strip():
            return False
        if category not in CATEGORIES:
            category = "preference"

        # Ensure profile+project exist in the index
        self.create_project(user, profile, project)

        items = self.load(user, profile, project)

        if dedup:
            try:
                from magnet.local_embeddings import is_semantic_duplicate
                existing_texts = [e["text"] for e in items if e.get("category") == category]
                if is_semantic_duplicate(text, existing_texts):
                    logger.debug(f"[memory_store] dedup skip: {text[:60]!r}")
                    return False
            except Exception:
                pass

        now = time.time()
        items.append({
            "id": _gen_id(),
            "category": category,
            "text": text.strip(),
            "status": "active",
            "confidence": confidence,
            "stored_at": now,
            "source_tool": source_tool,
            "source_transport": source_transport,
            "created_by": user,
            "updated_by": user,
            "updated_at": now,
        })
        self._save(user, profile, project, items)
        logger.info(f"[memory_store] [{category}] saved for {user}/{profile}/{project}")
        return True

    def delete_entry(self, user: str, profile: str, project: str, item_id: str) -> dict | None:
        """Delete a memory item by id. Returns the removed item, or None if not found."""
        items = self.load(user, profile, project)
        for i, item in enumerate(items):
            if item.get("id") == item_id:
                removed = items.pop(i)
                self._save(user, profile, project, items)
                return removed
        return None

    def edit_entry(
        self,
        user: str,
        profile: str,
        project: str,
        item_id: str,
        text: str | None = None,
        category: str | None = None,
    ) -> dict | None:
        """Update an existing item's text/category in place (id, confidence,
        created_by, etc. untouched). Returns the updated item, or None if
        not found. A blank/whitespace-only text or an unrecognized category
        is ignored rather than rejected — the other field can still apply."""
        items = self.load(user, profile, project)
        for item in items:
            if item.get("id") == item_id:
                if text is not None and text.strip():
                    item["text"] = text.strip()
                if category is not None and category in CATEGORIES:
                    item["category"] = category
                item["updated_by"] = user
                item["updated_at"] = time.time()
                self._save(user, profile, project, items)
                return item
        return None

    def delete_project(self, user: str, profile: str, project: str) -> bool:
        """Remove a project entirely: its stored items key AND its entry in
        the profile's project index. Returns True if the project existed."""
        p, pr = _n(profile), _n(project)
        index = self._load_index(user)
        projects = index.get(p, [])
        if pr not in projects:
            return False
        index[p] = [x for x in projects if x != pr]
        self._save_index(user, index)

        key = self._key(user, profile, project)
        if self._redis:
            self._redis.delete(key)
        else:
            self._mem.pop(key, None)
        logger.info(f"[memory_store] deleted project '{p}/{pr}' for {user}")
        return True

    def mark_goal_done(self, user: str, profile: str, project: str, item_id: str) -> dict | None:
        """Mark a goal item as done. Returns updated item, or None if not found or not a goal."""
        items = self.load(user, profile, project)
        for item in items:
            if item.get("id") == item_id:
                if item.get("category") != "goal":
                    return None
                item["status"] = "done"
                item["updated_by"] = user
                item["updated_at"] = time.time()
                self._save(user, profile, project, items)
                return item
        return None

    # ── Group helpers ─────────────────────────────────────────────────────────────

    def _group_by_category(self, items: list[dict]) -> dict[str, list[str]]:
        """Return {category: [text, ...]} — used for compact injection."""
        by_cat: dict[str, list[str]] = {c: [] for c in CATEGORIES}
        for e in items:
            c = e.get("category", "preference")
            if c in by_cat:
                by_cat[c].append(e["text"])
        return by_cat

    def _group_by_items(self, items: list[dict]) -> dict[str, list[dict]]:
        """Return {category: [item_dict, ...]} — used for display with IDs."""
        by_cat: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
        for e in items:
            c = e.get("category", "preference")
            if c in by_cat:
                by_cat[c].append(e)
        return by_cat

    # ── Format helpers ───────────────────────────────────────────────────────────

    def format_for_injection(self, user: str, profile: str, project: str) -> str:
        """Compact string for system-prompt injection. Omits done goals."""
        items = self.load(user, profile, project)
        active = [i for i in items if i.get("status", "active") != "done"]
        if not active:
            return ""
        by_cat = self._group_by_category(active)
        lines: list[str] = []
        for cat in _INJECT_ORDER:
            xs = by_cat.get(cat, [])
            if xs:
                lines.append(f"{_LABELS[cat]}:")
                for x in xs[-10:]:
                    lines.append(f"  - {x}")
        return "\n".join(lines)

    def format_merged_for_injection(
        self, user: str, profile: str, project: str, team_items: list[dict]
    ) -> str:
        """
        Merge personal + team items for system-prompt injection.
        Personal items take precedence; team items are deduplicated and labeled [team].
        Done goals are omitted.
        """
        personal = self.load(user, profile, project)
        active_personal = [i for i in personal if i.get("status", "active") != "done"]
        personal_texts = {i.get("text", "").lower() for i in active_personal}

        merged = list(active_personal)
        for ti in team_items:
            if ti.get("status", "active") == "done":
                continue
            if ti.get("text", "").lower() not in personal_texts:
                merged.append({**ti, "_team": True})

        if not merged:
            return ""

        by_cat: dict[str, list[str]] = {c: [] for c in CATEGORIES}
        for item in merged:
            c = item.get("category", "preference")
            if c in by_cat:
                text = item["text"]
                if item.get("_team"):
                    text = f"[team] {text}"
                by_cat[c].append(text)

        lines: list[str] = []
        for cat in _INJECT_ORDER:
            xs = by_cat.get(cat, [])
            if xs:
                lines.append(f"{_LABELS[cat]}:")
                for x in xs[-10:]:
                    lines.append(f"  - {x}")
        return "\n".join(lines)

    def format_merged_for_display(
        self, user: str, profile: str, project: str, team_items: list[dict]
    ) -> str:
        """
        Human-readable merged display. Team items labeled [team].
        Personal items take precedence; team-only items added below.
        """
        personal = self.load(user, profile, project)
        personal_texts = {i.get("text", "").lower() for i in personal}

        all_items = list(personal)
        for ti in team_items:
            if ti.get("text", "").lower() not in personal_texts:
                all_items.append({**ti, "_team": True})

        if not all_items:
            return f"No memory yet in {profile} / {project}."

        by_cat: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
        for e in all_items:
            c = e.get("category", "preference")
            if c in by_cat:
                by_cat[c].append(e)

        lines = [f"Memory — {profile} / {project} (with team):"]
        for cat in _DISPLAY_ORDER:
            lines.append(f"\n  {_LABELS[cat]}:")
            xs = by_cat.get(cat, [])
            if xs:
                for item in xs[-15:]:
                    item_id = item.get("id", "??????")
                    text = item["text"]
                    team_tag = " [team]" if item.get("_team") else ""
                    source = item.get("source_tool")
                    source_tag = f" ({source})" if source and source != "unknown" else ""
                    if cat == "goal":
                        status = item.get("status", "active")
                        lines.append(f"    [{item_id}]{team_tag} {text}  ({status}){source_tag}")
                    else:
                        lines.append(f"    [{item_id}]{team_tag} {text}{source_tag}")
            else:
                lines.append("    (none)")
        return "\n".join(lines)

    def format_for_display(self, user: str, profile: str, project: str) -> str:
        """Human-readable display with item IDs. Goals show status."""
        items = self.load(user, profile, project)
        if not items:
            return f"No memory yet in {profile} / {project}."
        by_cat = self._group_by_items(items)
        lines = [f"Memory — {profile} / {project}:"]
        for cat in _DISPLAY_ORDER:
            lines.append(f"\n  {_LABELS[cat]}:")
            xs = by_cat.get(cat, [])
            if xs:
                for item in xs[-15:]:
                    item_id = item.get("id", "??????")
                    text = item["text"]
                    source = item.get("source_tool")
                    source_tag = f" ({source})" if source and source != "unknown" else ""
                    if cat == "goal":
                        status = item.get("status", "active")
                        lines.append(f"    [{item_id}] {text}  ({status}){source_tag}")
                    else:
                        lines.append(f"    [{item_id}] {text}{source_tag}")
            else:
                lines.append("    (none)")
        return "\n".join(lines)


# Backward-compat alias
ProjectStore = MemoryStore
