# outsource-mcp

An MCP server that allows you to outsource tasks to other models.

This project uses [Agno](https://github.com/agno-agi/agno), a powerful framework for building AI agents with tool use capabilities. For more information on working with multimodal agents (text and image generation), see the [Agno multimodal documentation](https://docs.agno.com/agents/multimodal).

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

## Supported Models

This MCP server supports all models available through [Agno](https://docs.agno.com/models/introduction#supported-models):

You can follow the link to get a list of available models:
*   **OpenAI:** [https://platform.openai.com/docs/models](https://platform.openai.com/docs/models)
*   **Anthropic:** [https://docs.anthropic.com/en/docs/about-claude/models/overview](https://docs.anthropic.com/en/docs/about-claude/models/overview)
*   **Google:** [https://ai.google.dev/models](https://ai.google.dev/models)
*   **Groq:** [https://console.groq.com/docs/models](https://console.groq.com/docs/models)
*   **DeepSeek:** [https://api-docs.deepseek.com/api/list-models](https://api-docs.deepseek.com/api/list-models) 
*   **xAI:** [https://docs.x.ai/docs/models](https://docs.x.ai/docs/models)
*   **Perplexity:** [https://docs.perplexity.ai/guides/model-cards](https://docs.perplexity.ai/guides/model-cards)
*   **Cohere:** [https://docs.cohere.com/v2/docs/models](https://docs.cohere.com/v2/docs/models)
*   **Mistral AI:** [https://docs.mistral.ai/getting-started/models/models_overview/](https://docs.mistral.ai/getting-started/models/models_overview/)
*   **NVIDIA:** [https://build.nvidia.com/models](https://build.nvidia.com/models)
*   **HuggingFace:** [https://huggingface.co/models](https://huggingface.co/models)
*   **Ollama:** [https://ollama.com/library](https://ollama.com/library)
*   **Fireworks AI:** [https://fireworks.ai/models?view=list](https://fireworks.ai/models?view=list)
*   **OpenRouter:** [https://openrouter.ai/docs/overview/models](https://openrouter.ai/docs/overview/models)
*   **Together AI:** [https://docs.together.ai/docs/serverless-models](https://docs.together.ai/docs/serverless-models)
*   **Cerebras:** [https://training-api.cerebras.ai/en/latest/wsc/Model-zoo/Components/model_zoo_registry.html](https://training-api.cerebras.ai/en/latest/wsc/Model-zoo/Components/model_zoo_registry.html)
*   **DeepInfra:** [https://deepinfra.com/docs/models](https://deepinfra.com/docs/models)
*   **SambaNova Systems:** [https://docs.sambanova.ai/cloud/docs/get-started/supported-models](https://docs.sambanova.ai/cloud/docs/get-started/supported-models)
*   **AWS Bedrock:** [https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html) 
*   **Azure AI:** [https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/foundry-models-overview](https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/foundry-models-overview)
*   **IBM WatsonX:** [https://www.ibm.com/docs/en/software-hub/5.1.x?topic=install-foundation-models](https://www.ibm.com/docs/en/software-hub/5.1.x?topic=install-foundation-models)
*   **LiteLLM:** [https://docs.litellm.ai/docs/providers](https://docs.litellm.ai/docs/providers)
*   **Vercel AI SDK:** [https://sdk.vercel.ai/docs/introduction](https://sdk.vercel.ai/docs/introduction)
*   **Meta Llama:** [https://www.llama.com/get-started/](https://www.llama.com/get-started/)

Note: Each provider requires its corresponding API key to be set as an environment variable.

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
