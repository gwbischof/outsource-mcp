# outsource-mcp

An MCP (Model Context Protocol) server that allows you to outsource tasks to various AI agents. This server provides tools to detect available AI models and generate text or images using Agno agents.

## Features

- **Model Detection**: Automatically detects available AI models based on environment variables
- **Text Generation**: Create text using any available language model
- **Image Generation**: Generate images using DALL-E models
- **Multi-Provider Support**: Works with OpenAI, Anthropic, and Google AI models

## Installation

### Via uvx (recommended)

```bash
uvx outsource-mcp
```

### From source

```bash
git clone https://github.com/yourusername/outsource-mcp.git
cd outsource-mcp
uv pip install -e .
```

## Configuration

Set up your API keys as environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
```

## Claude Desktop Integration

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "outsource-mcp": {
      "command": "uvx",
      "args": ["outsource-mcp"]
    }
  }
}
```

Or for development:

```json
{
  "mcpServers": {
    "outsource-mcp": {
      "command": "uv",
      "args": ["run", "server.py"]
    }
  }
}
```

## Tools

### get_models

Detects available AI models based on your configured API keys.

**Returns:**
- `all_models`: List of all available models
- `text_models`: Models that can generate text
- `image_models`: Models that can generate images
- `by_provider`: Models grouped by provider

### outsource_text

Generates text using a specified AI model.

**Parameters:**
- `model` (string): The model to use (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
- `prompt` (string): The prompt to send to the model

**Returns:** Generated text response

### outsource_image

Generates images using DALL-E models.

**Parameters:**
- `model` (string): The model to use (e.g., "dall-e-3", "dall-e-2")
- `prompt` (string): The image generation prompt

**Returns:** Base64 encoded image data

## Supported Models

### OpenAI
- Text: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
- Image: dall-e-3, dall-e-2

### Anthropic
- Text: claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022, claude-3-opus-20240229

### Google
- Text: gemini-2.0-flash-exp, gemini-1.5-pro, gemini-1.5-flash

## Development

### Testing with MCP Inspector

```bash
mcp dev server.py
```

### Running tests

```bash
uv run pytest
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.