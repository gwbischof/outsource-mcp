# Outsource MCP

An MCP (Model Context Protocol) server that enables AI applications to outsource tasks to various model providers through a unified interface.

Compatible with any AI tool that supports the Model Context Protocol, including Claude Desktop, Cline, and other MCP-enabled applications.

Built with [FastMCP](https://github.com/mcp/fastmcp) for the MCP server implementation and [Agno](https://github.com/agno-agi/agno) for AI agent capabilities.

<img width="1154" alt="image" src="https://github.com/user-attachments/assets/cd364a7c-eae5-4c58-bc1f-fdeea6cb8434" />

## Features

- 🤖 **Multi-Provider Support**: Access 20+ AI providers through a single interface
- 📝 **Text Generation**: Generate text using models from OpenAI, Anthropic, Google, and more
- 🎨 **Image Generation**: Create images using DALL-E 3 and DALL-E 2
- 🔧 **Simple API**: Consistent interface with just three parameters: provider, model, and prompt
- 🔑 **Flexible Authentication**: Only configure API keys for the providers you use

## Configuration

Add the following configuration to your MCP client. Consult your MCP client's documentation for specific configuration details.

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

## Quick Start

Once installed and configured, you can use the tools in your MCP client:

1. **Generate text**: Use the `outsource_text` tool with provider "openai", model "gpt-4o-mini", and prompt "Write a haiku about coding"
2. **Generate images**: Use the `outsource_image` tool with provider "openai", model "dall-e-3", and prompt "A futuristic city skyline at sunset"

## Tools

### outsource_text
Creates an Agno agent with a specified provider and model to generate text responses. 

**Arguments:**
- `provider`: The provider name (e.g., "openai", "anthropic", "google", "groq", etc.)
- `model`: The model name (e.g., "gpt-4o", "claude-3-5-sonnet-20241022", "gemini-2.0-flash-exp")
- `prompt`: The text prompt to send to the model

### outsource_image
Generates images using AI models.

**Arguments:**
- `provider`: The provider name (currently only "openai" is supported)
- `model`: The model name ("dall-e-3" or "dall-e-2")
- `prompt`: The image generation prompt

Returns the URL of the generated image.

> **Note**: Image generation is currently only supported by OpenAI models (DALL-E 2 and DALL-E 3). Other providers only support text generation.

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

### Environment Variables

Each provider requires its corresponding API key:

| Provider | Environment Variable | Example |
|----------|---------------------|---------|
| OpenAI | `OPENAI_API_KEY` | sk-... |
| Anthropic | `ANTHROPIC_API_KEY` | sk-ant-... |
| Google | `GOOGLE_API_KEY` | AIza... |
| Groq | `GROQ_API_KEY` | gsk_... |
| DeepSeek | `DEEPSEEK_API_KEY` | sk-... |
| xAI | `XAI_API_KEY` | xai-... |
| Perplexity | `PERPLEXITY_API_KEY` | pplx-... |
| Cohere | `COHERE_API_KEY` | ... |
| Fireworks | `FIREWORKS_API_KEY` | ... |
| HuggingFace | `HUGGINGFACE_API_KEY` | hf_... |
| Mistral | `MISTRAL_API_KEY` | ... |
| NVIDIA | `NVIDIA_API_KEY` | nvapi-... |
| Ollama | `OLLAMA_HOST` | http://localhost:11434 |
| OpenRouter | `OPENROUTER_API_KEY` | ... |
| Together | `TOGETHER_API_KEY` | ... |
| Cerebras | `CEREBRAS_API_KEY` | ... |
| DeepInfra | `DEEPINFRA_API_KEY` | ... |
| SambaNova | `SAMBANOVA_API_KEY` | ... |
| AWS Bedrock | AWS credentials | Via AWS CLI/SDK |
| Azure AI | Azure credentials | Via Azure CLI/SDK |
| IBM WatsonX | `IBM_WATSONX_API_KEY` | ... |
| Meta Llama | `LLAMA_API_KEY` | ... |

**Note**: Only configure the API keys for providers you plan to use.

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

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
git clone https://github.com/gwbischof/outsource-mcp.git
cd outsource-mcp
uv sync
```

### Testing with MCP Inspector

The MCP Inspector allows you to test the server interactively:

```bash
mcp dev server.py
```

### Running Tests

The test suite includes integration tests that verify both text and image generation:

```bash
# Run all tests
uv run pytest
```

**Note:** Integration tests require API keys to be set in your environment.

## Troubleshooting

### Common Issues

1. **"Error: Unknown provider"**
   - Check that you're using a supported provider name from the list above
   - Provider names are case-insensitive

2. **"Error: OpenAI API error"** 
   - Verify your API key is correctly set in the environment variables
   - Check that your API key has access to the requested model
   - Ensure you have sufficient credits/quota

3. **"Error: No image was generated"**
   - This can happen if the image generation request fails
   - Try a simpler prompt or different model (dall-e-2 vs dall-e-3)

4. **Environment variables not working**
   - Make sure to restart your MCP client after updating the configuration
   - Verify the configuration file location for your specific MCP client
   - Check that the environment variables are properly formatted in the configuration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
