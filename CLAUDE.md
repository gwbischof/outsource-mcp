# MCP Server Development

## Required Context
- context7 gwbischof/python-mcp
- @https://github.com/modelcontextprotocol/python-sdk/blob/main/tests/test_examples.py
- @https://github.com/gwbischof/bluesky-social-mcp

## Reminders
- **Package Naming**: Use underscores not hyphens (`outsource_mcp` not `outsource-mcp`)
- **Tool Descriptions**: Write tool docstrings for AI audience, not human developers
- **Transport**: Default to stdio transport unless you specifically need HTTP
- **Error Messages**: Return descriptive errors that help the AI understand what went wrong
- **FastMCP**: Use FastMCP framework for simpler server setup
- **Installation**: Support `uvx` installation pattern for easy distribution
