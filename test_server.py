"""
Integration test for outsource-mcp server
"""

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
        args=["run", "python", "server.py"],
        env={**os.environ}  # Pass through all environment variables
    )

    # Connect to the server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()

            # List available tools
            tools = await session.list_tools()
            #for tool in tools.tools:
            #    print(f"  - {tool.name}: {tool.description}")

            # Verify we have the expected tools
            tool_names = [tool.name for tool in tools.tools]
            assert "outsource_text" in tool_names
            assert "outsource_image" in tool_names

            # Test outsource_text
            print("\n\nTesting outsource_text tool...")
            result = await session.call_tool(
                "outsource_text",
                {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "prompt": "Write a haiku about MCP servers",
                },
            )
            
            # Check the result content
            result_text = str(result.content)
            print(f"Result: {result_text}")
            
            # Extract the actual text from the TextContent representation
            if "text='" in result_text:
                # Parse the text content from the string representation
                start = result_text.find("text='") + 6
                end = result_text.find("'", start)
                actual_text = result_text[start:end]
            else:
                actual_text = result_text
            
            if os.getenv("OPENAI_API_KEY"):
                # If API key is set, response should NOT start with "Error"
                assert not actual_text.startswith("Error"), f"Got error when API key was set: {actual_text}"
            else:
                # If no API key, response should start with "Error"
                assert actual_text.startswith("Error"), f"Expected error when no API key set, got: {actual_text}"
