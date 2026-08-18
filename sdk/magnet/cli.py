"""
agent-magnet init
-----------------
One-command setup: detects installed AI tools, asks minimal questions,
and writes the correct MCP server config into each tool's config file.
"""

from __future__ import annotations

import getpass
import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


# ── Config path resolution ─────────────────────────────────────────────────────

def _claude_code_path() -> Path:
    return Path.home() / ".claude" / "settings.json"


def _claude_desktop_path() -> Path:
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    if system == "Windows":
        appdata = os.environ.get("APPDATA", "")
        return Path(appdata) / "Claude" / "claude_desktop_config.json"
    return Path.home() / ".config" / "claude" / "claude_desktop_config.json"


def _cursor_path() -> Path:
    return Path.home() / ".cursor" / "mcp.json"


def _cursor_path_alt() -> Path:
    """Fallback Cursor MCP config (VS Code-style globalStorage path)."""
    system = platform.system()
    if system == "Darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "Cursor"
            / "User"
            / "globalStorage"
            / "cursor.mcp"
            / "settings.json"
        )
    if system == "Windows":
        appdata = os.environ.get("APPDATA", "")
        return Path(appdata) / "Cursor" / "User" / "globalStorage" / "cursor.mcp" / "settings.json"
    return Path.home() / ".config" / "Cursor" / "User" / "globalStorage" / "cursor.mcp" / "settings.json"


TOOLS = {
    "Claude Code": _claude_code_path,
    "Claude Desktop": _claude_desktop_path,
    "Cursor": _cursor_path,
}


# ── CLAUDE.md memory block ────────────────────────────────────────────────────

_CLAUDE_MD_MARKER = "## Agent Magnet Memory"

def _claude_md_block(user_id: str) -> str:
    return "\n".join([
        "## Agent Magnet Memory",
        "",
        f"At conversation start, call `recall` (agent-magnet) to load {user_id}'s active project",
        "memory and adopt it silently before your first response.",
        "",
        "Agent Magnet keeps a running memory. Count user messages in the conversation.",
        "After roughly every 8 user messages, silently call `checkpoint` (pass the recent messages)",
        "to save what was decided so far, then reset your count and continue.",
        "Do not interrupt the user's flow to do this — fold it silently into your response.",
        "",
        "Whenever the user explicitly states a decision, preference, correction, or something to",
        "watch out for, call `remember` immediately (in the background — never announce it).",
        "",
        "If the user types *projects, call `list_projects` and present the menu.",
        "If the user types *save, call `save_now` (pass ALL messages so far) to do a",
        "cumulative save of everything up to this point.",
        "If the user types *status, call `get_status` to show memory status and usage.",
        "If the user types *forget <something>, call `forget_memory`. If they reference an item",
        "by its id (shown in brackets in show_project_memory), delete it directly; otherwise",
        "find the best text match and present it for confirmation before deleting.",
        "If the user says a goal is done/completed/finished, call `mark_done`.",
        "If the user asks where they left off / what were we doing / types *recap,",
        "call `recap` and deliver it as a natural catch-up, not a list.",
        "If the user types *memory, call `show_all_memory` for the active project full dump.",
        "If they say *memory all or ask to see everything across all projects,",
        "call `show_all_memory` with show_all=true for a cross-project overview.",
        "",
        "Team memory (requires a paid Agent Magnet key — MAGNET_API_KEY, plan team/pro):",
        "*team new <name> → call create_team. Returns a team_id to share with teammates.",
        "*team join <id> → call join_team. Then add MAGNET_TEAM_ID to MCP config.",
        "*team members → call list_team_members to show who's in the team.",
        "*team share → call share_project_to_team to share the active project with the team.",
        "*share <item_id> → call share_item_to_team to share one specific item.",
        "When working in a shared project (MAGNET_TEAM_ID set and project is shared),",
        "recall, show_project_memory, and recap automatically merge team memory.",
        "Team items are labeled [team] so the user can see what's shared vs personal.",
        "Never expose Redis URLs, storage keys, or backend details — say 'team memory' or 'shared'.",
        "",
        "When the user asks what you know or remember, call `recall` or `show_project_memory`",
        "and answer ONLY from what those tools return — never guess or invent.",
        "",
        "Never expose storage keys, project_ids, or backend details — show only project names.",
    ])


def _write_claude_md(user_id: str, project_id: str = "", profile_id: str = "") -> bool:
    """Write the Agent Magnet memory block to ~/.claude/CLAUDE.md. Returns True if written."""
    claude_md = Path.home() / ".claude" / "CLAUDE.md"
    block = _claude_md_block(user_id)

    if claude_md.exists():
        existing = claude_md.read_text(encoding="utf-8")
        if _CLAUDE_MD_MARKER in existing:
            # Replace the existing block in-place
            start = existing.index(_CLAUDE_MD_MARKER)
            # Find the next top-level heading (##) after the block, or end of file
            next_section = existing.find("\n## ", start + len(_CLAUDE_MD_MARKER))
            if next_section == -1:
                content = existing[:start].rstrip() + "\n\n" + block + "\n"
            else:
                content = existing[:start].rstrip() + "\n\n" + block + "\n\n" + existing[next_section:].lstrip()
            claude_md.write_text(content, encoding="utf-8")
            return True
        content = existing.rstrip() + "\n\n" + block + "\n"
    else:
        claude_md.parent.mkdir(parents=True, exist_ok=True)
        content = "# Memory\n\n" + block + "\n"

    claude_md.write_text(content, encoding="utf-8")
    return True


# ── JSON helpers ───────────────────────────────────────────────────────────────

class ConfigReadError(RuntimeError):
    """Raised when an existing client config cannot be safely updated."""


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ConfigReadError(
            f"Cannot safely update existing config {path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ConfigReadError(
            f"Cannot safely update existing config {path}: expected a JSON object"
        )
    return data


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(data, temp_file, indent=2, ensure_ascii=False)
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())

        if path.exists():
            temp_path.chmod(path.stat().st_mode & 0o777)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


# ── Binary detection ───────────────────────────────────────────────────────────

def _find_mcp_binary_path() -> str | None:
    """
    Best-effort locate the agent-magnet-mcp console script, even when it's
    not on PATH — common on Windows, where the Python Scripts dir often
    isn't added to PATH by default.
    """
    found = shutil.which("agent-magnet-mcp")
    if found:
        return found

    exe_name = "agent-magnet-mcp.exe" if platform.system() == "Windows" else "agent-magnet-mcp"
    py_dir = Path(sys.executable).resolve().parent
    candidates = [
        py_dir / exe_name,
        py_dir / "Scripts" / exe_name,        # Windows venv / user installs
        py_dir.parent / "Scripts" / exe_name,
        py_dir / "bin" / exe_name,            # posix venv layout
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def _resolve_mcp_command() -> tuple[str, list[str]]:
    """
    Resolve (command, args) to launch the MCP server. Prefers the installed
    console-script binary (by full path, so it works regardless of PATH);
    falls back to `<python> -m magnet.mcp_server` as command+args (never a
    single flattened string, which most MCP clients won't shell-parse).
    """
    found = _find_mcp_binary_path()
    if found:
        return found, []
    return sys.executable, ["-m", "magnet.mcp_server"]


def _hook_command(env_pairs: list[tuple[str, str]]) -> str:
    """Build the inline shell command for the Claude Code Stop hook."""
    hook_bin = shutil.which("agent-magnet-hook") or f"{sys.executable} -m magnet.hooks.save_session"
    prefix = " ".join(f"{k}={v}" for k, v in env_pairs)
    return f"{prefix} {hook_bin}"


# ── MCP server entry builder ───────────────────────────────────────────────────

def _build_mcp_entry(
    user_id: str, openai_key: str = "", redis_url: str = "",
    team_id: str = "", api_key: str = "", profile_id: str = "",
    magnet_team_id: str = "",
    # legacy params kept for backward compat with callers that pass positional args
    storage: str = "local",
) -> dict:
    env: dict[str, str] = {"MAGNET_USER_ID": user_id}
    if openai_key:
        env["MAGNET_OPENAI_KEY"] = openai_key
    if redis_url:
        env["MAGNET_REDIS_URL"] = redis_url
    if team_id:
        env["MAGNET_TEAM_ID"] = team_id
    if magnet_team_id:
        env["MAGNET_TEAM_ID"] = magnet_team_id
    if api_key:
        env["MAGNET_API_KEY"] = api_key
    if profile_id and profile_id != "default":
        env["MAGNET_PROFILE"] = profile_id
    # No MAGNET_LOCAL_MODE needed — SQLite is now the automatic default

    command, args = _resolve_mcp_command()
    entry: dict[str, Any] = {"command": command}
    if args:
        entry["args"] = args
    entry["env"] = env
    return entry


# ── Per-tool config writers ────────────────────────────────────────────────────

def _write_claude_code(
    path: Path, user_id: str, openai_key: str = "", redis_url: str = "",
    team_id: str = "", api_key: str = "", profile_id: str = "",
    # legacy positional args kept for compat
    storage: str = "local",
) -> str:
    config = _read_json(path)

    config.setdefault("mcpServers", {})
    config["mcpServers"]["agent-magnet"] = _build_mcp_entry(
        user_id, openai_key, redis_url, team_id, api_key, profile_id
    )

    env_pairs: list[tuple[str, str]] = [("MAGNET_USER_ID", user_id)]
    if openai_key:
        env_pairs.append(("MAGNET_OPENAI_KEY", openai_key))
    if redis_url:
        env_pairs.append(("MAGNET_REDIS_URL", redis_url))
    if team_id:
        env_pairs.append(("MAGNET_TEAM_ID", team_id))
    if api_key:
        env_pairs.append(("MAGNET_API_KEY", api_key))
    if profile_id and profile_id != "default":
        env_pairs.append(("MAGNET_PROFILE", profile_id))

    hook_entry: dict[str, Any] = {
        "type": "command",
        "command": _hook_command(env_pairs),
        "timeout": 10,
    }

    config.setdefault("hooks", {})
    existing_stop = config["hooks"].get("Stop", [])
    new_stop = [b for b in existing_stop if not _is_magnet_hook(b)]
    new_stop.append({"matcher": "", "hooks": [hook_entry]})
    config["hooks"]["Stop"] = new_stop

    _write_json(path, config)
    return "MCP server + Stop hook"


def _is_magnet_hook(block: Any) -> bool:
    if not isinstance(block, dict):
        return False
    for h in block.get("hooks", []):
        if isinstance(h, dict) and "magnet" in h.get("command", ""):
            return True
    return False


def _write_mcp_only(
    path: Path, user_id: str, openai_key: str = "", redis_url: str = "",
    team_id: str = "", api_key: str = "", profile_id: str = "",
    storage: str = "local",
) -> str:
    config = _read_json(path)
    config.setdefault("mcpServers", {})
    config["mcpServers"]["agent-magnet"] = _build_mcp_entry(
        user_id, openai_key, redis_url, team_id, api_key, profile_id
    )
    _write_json(path, config)
    return "MCP server"


# ── Manual / fallback config ────────────────────────────────────────────────────

def _manual_config_dict(user_id: str) -> dict:
    """The minimal, universal MCP config block — works with any MCP-compatible client."""
    return {
        "mcpServers": {
            "agent-magnet": {
                "command": "agent-magnet-mcp",
                "env": {"MAGNET_USER_ID": user_id},
            }
        }
    }


def _print_manual_block(user_id: str) -> None:
    """
    Print a ready-to-paste MCP config block for any client, plus a PATH note
    if 'agent-magnet-mcp' can't be resolved to a bare command.
    """
    print("  Manual MCP config (paste into any MCP-compatible client's config):")
    print()
    block = json.dumps(_manual_config_dict(user_id), indent=2)
    for line in block.splitlines():
        print(f"  {line}")
    print()

    on_path = shutil.which("agent-magnet-mcp")
    resolved = _find_mcp_binary_path()
    if resolved is None:
        print("  Note: 'agent-magnet-mcp' could not be located at all.")
        print(f"  Use this instead:")
        print(f"    \"command\": \"{sys.executable}\"")
        print(f"    \"args\": [\"-m\", \"magnet.mcp_server\"]")
        print()
    elif not on_path:
        print("  Note: 'agent-magnet-mcp' is not on your PATH (common on Windows).")
        print(f"  If your client can't launch the bare command above, use the full path instead:")
        print(f"    \"command\": \"{resolved}\"")
        print()


# ── Local storage bootstrap ─────────────────────────────────────────────────────

def _bootstrap_local_storage(user_id: str) -> None:
    """Create the default 'general' project and active.json. Tool-agnostic."""
    try:
        from magnet.local_store import SQLiteBackend
        from magnet.project_store import MemoryStore
        _store = MemoryStore(SQLiteBackend())
        _store.create_project(user_id, "general")
    except Exception as exc:
        print(f"  Warning: could not init local storage: {exc}")

    active_path = Path.home() / ".agent-magnet" / "active.json"
    try:
        active_path.parent.mkdir(parents=True, exist_ok=True)
        active_path.write_text(
            json.dumps({"project": "general"}, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"  Warning: could not write active.json: {exc}")


# ── Input helpers ──────────────────────────────────────────────────────────────

def _ask(prompt: str, default: str = "") -> str:
    try:
        val = input(prompt).strip()
        return val if val else default
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)


def _ask_secret(prompt: str) -> str:
    try:
        val = getpass.getpass(prompt)
        return val.strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)


def _ask_bool(prompt: str, default: bool = True) -> bool:
    hint = "[Y/n]" if default else "[y/N]"
    raw = _ask(f"{prompt} {hint}: ", "")
    if not raw:
        return default
    return raw.lower() in ("y", "yes")


# ── Main init command ──────────────────────────────────────────────────────────

def cmd_init(manual: bool = False) -> None:
    print()
    print("  Agent Magnet Setup")
    print("  " + "─" * 38)
    print()

    if manual:
        user_id = _ask("  Your name/identifier: ")
        if not user_id:
            user_id = getpass.getuser()
            print(f"  Using system username: {user_id}")
        print()

        _bootstrap_local_storage(user_id)

        print("  " + "━" * 38)
        print("   Manual config")
        print("  " + "━" * 38)
        print()
        _print_manual_block(user_id)
        print("  Add this to your MCP client's config file, then restart it.")
        print()
        return

    # 1. Detect tools — never a dead end: whether or not anything is found,
    #    we fall through to a summary that always includes the manual block.
    print("  Scanning for AI tools...")
    print()
    found: dict[str, Path] = {}
    for name, path_fn in TOOLS.items():
        path = path_fn()
        if path.exists() or path.parent.exists():
            found[name] = path
            status = "found" if path.exists() else "config dir found"
            print(f"    ✓ {name} ({status})")
        else:
            print(f"    – {name} (not detected)")

    if not found:
        print()
        print("  No known AI tool detected on this machine.")
        print("  That's fine — Agent Magnet works with any MCP-compatible client.")
        print("  You'll get a config block to paste in manually below.")
        print()

    # 2. Questions
    print()
    print("  Configuration")
    print("  " + "─" * 38)
    print()

    user_id = _ask("  Your name/identifier: ")
    if not user_id:
        user_id = getpass.getuser()
        print(f"  Using system username: {user_id}")

    openai_key = ""
    redis_url = ""
    team_id = ""
    api_key = ""
    profile_id = "default"

    if found:
        print()
        openai_key = _ask_secret(
            "  OpenAI key for smarter extraction (press Enter to skip — memory works great without it): "
        )
        if not openai_key:
            print("  Skipped — on-device memory and semantic search will be used.")
        print()

        print("  Team setup (optional)")
        print("  " + "─" * 38)
        print()
        print("  Team memory lets you share decisions with teammates.")
        print("  It requires all team members to use the same Redis instance.")
        print()
        team_id = _ask("  Team ID (press Enter to skip — needs shared Redis): ")
        if team_id:
            if not redis_url:
                print()
                redis_url = _ask("  Redis URL for team (redis://...): ")
            if redis_url:
                print(f"  Team: {team_id} · Redis: configured ✓")
            else:
                print("  Note: team memory won't work without MAGNET_REDIS_URL.")
        print()

    # 3. Bootstrap default project in local storage (tool-agnostic)
    _bootstrap_local_storage(user_id)

    # 4. Write configs for whatever was detected
    written: dict[str, Path] = {}
    if found:
        print("  Writing configuration...")
        print()

        writers = {
            "Claude Code": _write_claude_code,
            "Claude Desktop": _write_mcp_only,
            "Cursor": _write_mcp_only,
        }

        for tool_name, path in found.items():
            writer = writers.get(tool_name, _write_mcp_only)
            try:
                action = writer(path, user_id, openai_key, redis_url, team_id, api_key, profile_id)
                print(f"    ✓ {tool_name} — {action}")
                written[tool_name] = path
            except Exception as exc:
                print(f"    ✗ {tool_name} — failed: {exc}")

        if "Claude Code" in found:
            try:
                ok = _write_claude_md(user_id)
                if ok:
                    print(f"    ✓ ~/.claude/CLAUDE.md — memory auto-trigger added")
                else:
                    print(f"    – ~/.claude/CLAUDE.md — already configured, skipped")
            except Exception as exc:
                print(f"    ✗ ~/.claude/CLAUDE.md — failed: {exc}")

    # 5. Summary — always printed, regardless of whether detection succeeded,
    #    so no user is ever left stuck on an unrecognized platform.
    db_path = Path.home() / ".agent-magnet" / "memory.db"
    print()
    print("  " + "━" * 38)
    print("   Agent Magnet is configured!")
    print("  " + "━" * 38)
    print()
    print("  Memory works — stored locally on your machine.")
    print("  No accounts, no cloud, no keys needed.")
    print()
    print(f"  Active context: personal / general")
    print(f"  Memory stored at: {db_path}")
    print()

    if written:
        print("  Config written to:")
        for tool_name, path in written.items():
            print(f"    · {tool_name}: {path}")
        print()

    _print_manual_block(user_id)

    if written:
        print("  Using another MCP client too? Paste the block above into its config as well.")
    else:
        print("  Paste the block above into your MCP-compatible client's config, then restart it.")
    print()
    if not openai_key and found:
        print("  For smarter extraction, add MAGNET_OPENAI_KEY to your tool's MCP config.")
        print()
    print("  Then type *projects to get started.")
    print()


def cmd_config(argv: list[str]) -> None:
    """Print the current recommended MCP config block on demand, any time."""
    user_id = argv[0] if argv else getpass.getuser()
    print()
    print("  Agent Magnet — MCP config")
    print("  " + "─" * 38)
    print()
    _print_manual_block(user_id)


# ── Entry point ────────────────────────────────────────────────────────────────

def main(args: list[str] | None = None) -> None:
    argv = args if args is not None else sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help"):
        print("Usage: agent-magnet <command>")
        print()
        print("Commands:")
        print("  init             Detect AI tools and configure Agent Magnet")
        print("  init --manual    Skip detection — just print a config block to paste anywhere")
        print("  config [name]    Print the recommended MCP config block on demand")
        print()
        return

    if argv[0] == "init":
        cmd_init(manual="--manual" in argv[1:])
        return

    if argv[0] == "config":
        cmd_config(argv[1:])
        return

    print(f"Unknown command: {argv[0]}")
    print("Run 'agent-magnet --help' for usage.")
    sys.exit(1)


if __name__ == "__main__":
    main()
