# outsource-mcp

An MCP server that allows you to outsource AI tasks to various model providers.

This project uses [Agno](https://github.com/agno-agi/agno), a powerful framework for building AI agents with tool use capabilities.

## Tools

### outsource_text
Creates an Agno agent with a specified provider and model to generate text responses. 

**Arguments:**
- `provider`: The provider name (e.g., "openai", "anthropic", "google", "groq", etc.)
- `model`: The model name (e.g., "gpt-4o", "claude-3-5-sonnet-20241022", "gemini-2.0-flash-exp")
- `prompt`: The text prompt to send to the model

### outsource_image
Generates images using AI models. Currently supports OpenAI's DALL-E models.

**Arguments:**
- `provider`: The provider name (currently only "openai" is supported)
- `model`: The model name ("dall-e-3" or "dall-e-2")
- `prompt`: The image generation prompt

Returns base64 encoded image data that can be directly used in applications.

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

## Supported Providers

The following providers are supported. Use the provider name (in parentheses) as the `provider` argument:

### Core Providers
- **OpenAI** (`openai`) - GPT-4, GPT-3.5, DALL-E, etc. | [Models](https://platform.openai.com/docs/models)
- **Anthropic** (`anthropic`) - Claude 3.5, Claude 3, etc. | [Models](https://docs.anthropic.com/en/docs/about-claude/models/overview)
- **Google** (`google`) - Gemini Pro, Gemini Flash, etc. | [Models](https://ai.google.dev/models)
- **Groq** (`groq`) - Llama 3, Mixtral, etc. | [Models](https://console.groq.com/docs/models)
- **DeepSeek** (`deepseek`) - DeepSeek Chat & Coder | [Models](https://api-docs.deepseek.com/api/list-models)
- **xAI** (`xai`) - Grok models | [Models](https://docs.x.ai/docs/models)
- **Perplexity** (`perplexity`) - Sonar models | [Models](https://docs.perplexity.ai/guides/model-cards)

### Additional Providers
- **Cohere** (`cohere`) - Command models | [Models](https://docs.cohere.com/v2/docs/models)
- **Mistral AI** (`mistral`) - Mistral Large, Medium, Small | [Models](https://docs.mistral.ai/getting-started/models/models_overview/)
- **NVIDIA** (`nvidia`) - Various models | [Models](https://build.nvidia.com/models)
- **HuggingFace** (`huggingface`) - Open source models | [Models](https://huggingface.co/models)
- **Ollama** (`ollama`) - Local models | [Models](https://ollama.com/library)
- **Fireworks AI** (`fireworks`) - Fast inference | [Models](https://fireworks.ai/models?view=list)
- **OpenRouter** (`openrouter`) - Multi-provider access | [Models](https://openrouter.ai/docs/overview/models)
- **Together AI** (`together`) - Open source models | [Models](https://docs.together.ai/docs/serverless-models)
- **Cerebras** (`cerebras`) - Fast inference | [Models](https://cerebras.ai/models)
- **DeepInfra** (`deepinfra`) - Optimized models | [Models](https://deepinfra.com/docs/models)
- **SambaNova** (`sambanova`) - Enterprise models | [Models](https://docs.sambanova.ai/cloud/docs/get-started/supported-models)

### Enterprise Providers
- **AWS Bedrock** (`aws` or `bedrock`) - AWS-hosted models | [Models](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html)
- **Azure AI** (`azure`) - Azure-hosted models | [Models](https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/foundry-models-overview)
- **IBM WatsonX** (`ibm` or `watsonx`) - IBM models | [Models](https://www.ibm.com/docs/en/software-hub/5.1.x?topic=install-foundation-models)
- **LiteLLM** (`litellm`) - Universal interface | [Models](https://docs.litellm.ai/docs/providers)
- **Vercel v0** (`vercel` or `v0`) - Vercel AI | [Models](https://sdk.vercel.ai/docs/introduction)
- **Meta Llama** (`meta`) - Direct Meta access | [Models](https://www.llama.com/get-started/)

Note: Each provider requires its corresponding API key to be set as an environment variable.

## Examples

### Text Generation
```
# Using OpenAI
provider: openai
model: gpt-4o-mini
prompt: Write a haiku about coding

# Using Anthropic
provider: anthropic
model: claude-3-5-sonnet-20241022
prompt: Explain quantum computing in simple terms

# Using Google
provider: google
model: gemini-2.0-flash-exp
prompt: Create a recipe for chocolate chip cookies
```

### Image Generation
```
# Using DALL-E 3
provider: openai
model: dall-e-3
prompt: A serene Japanese garden with cherry blossoms

# Using DALL-E 2
provider: openai
model: dall-e-2
prompt: A futuristic cityscape at sunset
```

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
