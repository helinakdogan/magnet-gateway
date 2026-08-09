<p align="center">
  <img src="assets/logo.png" alt="Agent Magnet" width="600">
</p>

<p align="center">
  <img src="https://img.shields.io/pypi/v/agent-magnet?label=PyPI&labelColor=111827&color=8B5CF6" alt="PyPI">
  <a href="https://github.com/helinakdogan/agentmagnet/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-A855F7"></a>
</p>

> Your AI forgets your project every new session. This fixes that.

## What it does

- Learns automatically as you work — no "remember this" required
- Shares one memory across Claude, Cursor, and Codex
- Lets you forget or mark things done — memory doesn't just pile up

## Quick start

```bash
pip install agent-magnet
agent-magnet init
```

That's the whole install. No Redis, no keys, local by default.

## What it remembers

Decisions, conventions, watch-outs, failed attempts, goals, preferences, and actions taken — one memory, sorted automatically.

## Team memory

Shared memory for a team — one person's decision becomes everyone's context. Requires a paid Agent Magnet key from [agentmagnet.app](https://agentmagnet.app).

## Works everywhere

Claude, Cursor, Codex — same memory, local by default. `agent-magnet init` detects and configures Claude Code, Claude Desktop, and Cursor for you.

<details>
<summary>Manual MCP config (Codex, or any other MCP client)</summary>

```json
{
  "mcpServers": {
    "agent-magnet": {
      "command": "agent-magnet-mcp",
      "env": { "MAGNET_USER_ID": "your_name" }
    }
  }
}
```
</details>

---

MIT licensed. [Contribute](.github/CONTRIBUTING.md) — and a ⭐ if this saved your context window.
