# outsource-mcp

An MCP server that allows you to outsource tasks to other models.

This project uses [Agno](https://github.com/agno-agi/agno), a powerful framework for building AI agents with tool use capabilities. For more information on working with multimodal agents (text and image generation), see the [Agno multimodal documentation](https://docs.agno.com/agents/multimodal).

## Tools

### outsource_text
Creates an Agno agent with a specified model to generate text responses. Pass any supported model name and a prompt to get AI-generated content. The tool automatically routes to the appropriate provider based on the model name.

### outsource_image
Generates images using AI models. Currently supports DALL-E 3 and DALL-E 2 for image generation. Returns base64 encoded image data that can be directly used in applications.

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
        "PERPLEXITY_API_KEY": "your-perplexity-key",
        "COHERE_API_KEY": "your-cohere-key",
        "FIREWORKS_API_KEY": "your-fireworks-key",
        "HUGGINGFACE_API_KEY": "your-huggingface-key",
        "MISTRAL_API_KEY": "your-mistral-key",
        "NVIDIA_API_KEY": "your-nvidia-key",
        "OLLAMA_HOST": "http://localhost:11434",
        "OPENROUTER_API_KEY": "your-openrouter-key",
        "TOGETHER_API_KEY": "your-together-key",
        "CEREBRAS_API_KEY": "your-cerebras-key",
        "DEEPINFRA_API_KEY": "your-deepinfra-key",
        "SAMBANOVA_API_KEY": "your-sambanova-key"
      }
    }
  }
}
```

Note: The environment variables are optional. Only include the API keys for the providers you want to use.

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

This MCP server supports all models available through [Agno](https://docs.agno.com/models/introduction#supported-models):

### Core Providers
- **OpenAI**: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo, dall-e-3, dall-e-2
- **Anthropic**: claude-3-5-sonnet, claude-3-5-haiku, claude-3-opus
- **Google**: gemini-2.0-flash-exp, gemini-1.5-pro, gemini-1.5-flash
- **Groq**: llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b
- **DeepSeek**: deepseek-chat, deepseek-coder
- **xAI**: grok-beta, grok-vision-beta
- **Perplexity**: sonar, sonar-pro

### Additional Providers
- **Cohere**: command-r-plus, command-r, command
- **Fireworks**: Various open source models
- **HuggingFace**: Access to HuggingFace model hub
- **Mistral**: mistral-large, mistral-medium, mistral-small
- **NVIDIA**: Meta Llama models via NVIDIA API
- **Ollama**: Local models (llama3, mistral, codellama)
- **OpenRouter**: Access to multiple providers
- **Together AI**: Open source models
- **Cerebras**: Fast inference models
- **DeepInfra**: Optimized open source models
- **SambaNova**: High-performance models

### Enterprise Providers (Supported by Agno, configuration needed)
- **AWS Bedrock**: Access AWS-hosted models
- **Azure AI**: Azure-hosted models including OpenAI
- **IBM WatsonX**: IBM's AI models
- **LiteLLM**: Universal model interface
- **Vercel v0**: Vercel's AI service
- **Meta Llama**: Direct Meta model access

Note: Each provider requires its corresponding API key to be set as an environment variable.

## How It Works

The server automatically routes models to the appropriate provider based on the model name:
- Models starting with "gpt" or "dall-e" → OpenAI
- Models starting with "claude" → Anthropic  
- Models starting with "gemini" → Google
- Models starting with "llama" or "mixtral" → Groq
- Models starting with "deepseek" → DeepSeek
- And many more patterns for other providers

If a model name doesn't match any known pattern, the server will raise an error with the unrecognized model name.

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
