# MCP Server

Expose Agent Memory as MCP tools for Cursor, VS Code, Claude Code, and any other MCP-compatible client. The server speaks both MCP 1.x and 2.x.

## Quick start

```bash
# Install
pip install agent-memory-sdk

# Run the MCP server
AGENT_MEMORY_DIR=~/.agent_memory agent-memory-mcp

# Or via uvx (no install needed)
uvx agent-memory-sdk
```

## Configure Cursor

Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "agent-memory": {
      "command": "agent-memory-mcp",
      "env": {
        "AGENT_MEMORY_DIR": "~/.agent_memory"
      }
    }
  }
}
```

## Configure Claude Code

```json
{
  "mcpServers": {
    "agent-memory": {
      "command": "uvx",
      "args": ["agent-memory-sdk"],
      "env": {
        "AGENT_MEMORY_DIR": "~/.agent_memory"
      }
    }
  }
}
```

## Docker-based MCP (recommended for isolation)

```json
{
  "mcpServers": {
    "agent-memory": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "-v", "agent_memory_data:/home/appuser/.agent_memory",
        "ghcr.io/theprodsde/agent-memory-sdk:latest",
        "agent-memory-mcp"
      ]
    }
  }
}
```

## Available tools

| Tool | What it does |
|------|-------------|
| `remember_memory` | Store a query/response pair with type, scope, tags, confidence, TTL |
| `resolve_memory` | Retrieve and decide — returns action + payload (response for REPLAY, context for RESTORE/VERIFY) |
| `list_memories` | Paginated list with optional scope filter |
| `get_memory_by_id` | Fetch a single memory by ID |
| `forget_memory` | Delete by ID |
| `archive_memory` | Archive by ID |
| `consolidate_memories` | Merge near-duplicate memories |

## resolve_memory response shape

```json
{
  "action": "replay",
  "instruction": "Replay this stored answer verbatim.",
  "matched_query": "How do I reset my password?",
  "stored_at": "2024-01-15T10:30:00Z",
  "times_reused": 3,
  "response": "Go to Settings → Security → Reset Password."
}
```

For `restore` / `verify`: the payload includes `context` (array of matching entries) and `prompt_context` (pre-formatted Markdown for the LLM system prompt).

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_MEMORY_DIR` | `.agent_memory` | Persist directory |
| `AGENT_MEMORY_COLLECTION` | `agent_memories` | Collection / table name |
| `AGENT_MEMORY_BACKEND` | `sqlite` | `sqlite` · `chromadb` · `redis` · `postgres` · `qdrant` |
