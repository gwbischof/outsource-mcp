"""
Integration test for outsource-mcp server
"""

import asyncio
import os
import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


@pytest.mark.asyncio
async def test_server_integration():
    """Test the MCP server integration"""
    # Set up server parameters to run our server
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "server.py"]
    )

    # Connect to the server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()

            # List available tools
            tools = await session.list_tools()
            print("\nAvailable tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")

            # Verify we have the expected tools
            tool_names = [tool.name for tool in tools.tools]
            assert "get_models" in tool_names
            assert "outsource_text" in tool_names
            assert "outsource_image" in tool_names

            # Test get_models tool
            print("\n\nTesting get_models tool...")
            result = await session.call_tool("get_models", {})
            print(f"Result: {result.content}")

            # Test outsource_text if OpenAI key is available
            if os.getenv("OPENAI_API_KEY"):
                print("\n\nTesting outsource_text tool...")
                result = await session.call_tool(
                    "outsource_text",
                    {
                        "model": "gpt-4o-mini",
                        "prompt": "Write a haiku about MCP servers"
                    }
                )
                print(f"Result: {result.content}")
            else:
                print("\n\nTesting outsource_text tool...")
                result = await session.call_tool(
                    "outsource_text",
                    {
                        "model": "gpt-4o-mini",
                        "prompt": "Write a haiku about MCP servers"
                    }
                )
                print(f"Result: {result.content}")
                # Verify it returns an error when model is not available
                assert "Error" in str(result.content)
