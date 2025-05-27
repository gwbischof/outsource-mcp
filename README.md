# outsource-mcp

An MCP server that allows you to outsource tasks to other models.

This project uses [Agno](https://github.com/agno-agi/agno), a powerful framework for building AI agents with tool use capabilities. For more information on working with multimodal agents (text and image generation), see the [Agno multimodal documentation](https://docs.agno.com/agents/multimodal).

## Tools

### get_models
Detects available AI models based on your configured API keys. Returns a comprehensive list of models organized by:
- All available models across providers
- Text generation capable models
- Image generation capable models
- Models grouped by provider (OpenAI, Anthropic, Google)

### outsource_text
Creates an Agno agent with a specified model to generate text responses. Pass any available text model and a prompt to get AI-generated content. Supports models from OpenAI (GPT-4, GPT-3.5), Anthropic (Claude), and Google (Gemini).

### outsource_image
Creates an Agno agent to generate images using DALL-E models. Currently supports DALL-E 3 and DALL-E 2 for image generation. Returns base64 encoded image data that can be directly used in applications.

## Installation

### Via uvx (from GitHub)

```bash
uvx --from git+https://github.com/yourusername/outsource-mcp.git outsource-mcp
```

### From source

```bash
git clone https://github.com/yourusername/outsource-mcp.git
cd outsource-mcp
uv pip install -e .
```

## Claude Desktop Integration

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "outsource-mcp": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/gwbischof/outsource-mcp.git", "outsource-mcp"],
      "env": {
        "OPENAI_API_KEY": "your-openai-key",
        "ANTHROPIC_API_KEY": "your-anthropic-key",
        "GOOGLE_API_KEY": "your-google-key",
        "GROQ_API_KEY": "your-groq-key",
        "DEEPSEEK_API_KEY": "your-deepseek-key",
        "XAI_API_KEY": "your-xai-key",
        "PERPLEXITY_API_KEY": "your-perplexity-key"
      }
    }
  }
}
```

Note: The environment variables are optional. Only include the API keys for the providers you want to use. The `get_models` tool will detect which providers are available based on the configured keys.

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

## Supported Models

### OpenAI
- Text: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
- Image: dall-e-3, dall-e-2

### Anthropic
- Text: claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022, claude-3-opus-20240229

### Google
- Text: gemini-2.0-flash-exp, gemini-1.5-pro, gemini-1.5-flash

### Groq
- Text: llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768

### DeepSeek
- Text: deepseek-chat, deepseek-coder

### xAI
- Text: grok-beta, grok-vision-beta

### Perplexity
- Text: sonar, sonar-pro

This MCP server supports all models available through [Agno](https://docs.agno.com/models/introduction#supported-models). Additional providers like AWS Bedrock, Azure, Cerebras, Cohere, and others can be added by extending the MODEL_PROVIDERS configuration.

## Development

### Testing with MCP Inspector

```bash
mcp dev server.py
```

### Running tests
**Note:** The integration tests will use your configured API keys.

```bash
uv run pytest
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
