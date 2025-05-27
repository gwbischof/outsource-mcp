"""
MCP server for outsourcing tasks to AI agents.

This server provides tools to:
1. Detect available AI models based on environment variables
2. Generate images using Agno agents with specified models
3. Generate text using Agno agents with specified models
"""

import os
from typing import Dict, List

from mcp.server.fastmcp import FastMCP
from agno.agent import Agent

# Core imports that are always available
from agno.models.openai import OpenAIChat
from agno.models.anthropic import Claude
from agno.models.google import Gemini
from agno.models.groq import Groq
from agno.models.deepseek import DeepSeek
from agno.models.xai import xAI
from agno.models.perplexity import Perplexity

# Optional imports - these are imported on demand to avoid dependency issues
# The imports are handled in get_model_class() function

# Create the MCP server instance
mcp = FastMCP("outsource-mcp")

# Model configurations
MODEL_PROVIDERS = {
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "models": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "dall-e-3",
            "dall-e-2",
        ],
        "text_models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "image_models": ["dall-e-3", "dall-e-2"],
    },
    "anthropic": {
        "env_key": "ANTHROPIC_API_KEY",
        "models": [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ],
        "text_models": [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ],
        "image_models": [],
    },
    "google": {
        "env_key": "GOOGLE_API_KEY",
        "models": ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"],
        "text_models": ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"],
        "image_models": [],
    },
    "groq": {
        "env_key": "GROQ_API_KEY",
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
        ],
        "text_models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
        ],
        "image_models": [],
    },
    "deepseek": {
        "env_key": "DEEPSEEK_API_KEY",
        "models": ["deepseek-chat", "deepseek-coder"],
        "text_models": ["deepseek-chat", "deepseek-coder"],
        "image_models": [],
    },
    "xai": {
        "env_key": "XAI_API_KEY",
        "models": ["grok-beta", "grok-vision-beta"],
        "text_models": ["grok-beta", "grok-vision-beta"],
        "image_models": [],
    },
    "perplexity": {
        "env_key": "PERPLEXITY_API_KEY",
        "models": ["sonar", "sonar-pro"],
        "text_models": ["sonar", "sonar-pro"],
        "image_models": [],
    },
    "cohere": {
        "env_key": "COHERE_API_KEY",
        "models": ["command-r-plus", "command-r", "command"],
        "text_models": ["command-r-plus", "command-r", "command"],
        "image_models": [],
    },
    "fireworks": {
        "env_key": "FIREWORKS_API_KEY",
        "models": ["accounts/fireworks/models/llama-v3p1-8b-instruct"],
        "text_models": ["accounts/fireworks/models/llama-v3p1-8b-instruct"],
        "image_models": [],
    },
    "huggingface": {
        "env_key": "HUGGINGFACE_API_KEY",
        "models": ["meta-llama/Llama-2-7b-chat-hf"],
        "text_models": ["meta-llama/Llama-2-7b-chat-hf"],
        "image_models": [],
    },
    "mistral": {
        "env_key": "MISTRAL_API_KEY",
        "models": [
            "mistral-large-latest",
            "mistral-medium-latest",
            "mistral-small-latest",
        ],
        "text_models": [
            "mistral-large-latest",
            "mistral-medium-latest",
            "mistral-small-latest",
        ],
        "image_models": [],
    },
    "nvidia": {
        "env_key": "NVIDIA_API_KEY",
        "models": ["meta/llama3-70b-instruct"],
        "text_models": ["meta/llama3-70b-instruct"],
        "image_models": [],
    },
    "ollama": {
        "env_key": "OLLAMA_HOST",
        "models": ["llama3", "mistral", "codellama"],
        "text_models": ["llama3", "mistral", "codellama"],
        "image_models": [],
    },
    "openrouter": {
        "env_key": "OPENROUTER_API_KEY",
        "models": ["openai/gpt-4", "anthropic/claude-3-opus"],
        "text_models": ["openai/gpt-4", "anthropic/claude-3-opus"],
        "image_models": [],
    },
    "together": {
        "env_key": "TOGETHER_API_KEY",
        "models": ["togethercomputer/llama-2-70b-chat"],
        "text_models": ["togethercomputer/llama-2-70b-chat"],
        "image_models": [],
    },
    "cerebras": {
        "env_key": "CEREBRAS_API_KEY",
        "models": ["llama3.1-8b", "llama3.1-70b"],
        "text_models": ["llama3.1-8b", "llama3.1-70b"],
        "image_models": [],
    },
    "deepinfra": {
        "env_key": "DEEPINFRA_API_KEY",
        "models": ["meta-llama/Llama-2-70b-chat-hf"],
        "text_models": ["meta-llama/Llama-2-70b-chat-hf"],
        "image_models": [],
    },
    "sambanova": {
        "env_key": "SAMBANOVA_API_KEY",
        "models": ["Meta-Llama-3.1-8B-Instruct"],
        "text_models": ["Meta-Llama-3.1-8B-Instruct"],
        "image_models": [],
    },
}


def get_model_class(model_name: str):
    """Get the appropriate model class based on model name."""
    # Core models (always imported)
    if model_name.startswith(("gpt", "dall-e")):
        return OpenAIChat
    elif model_name.startswith("claude"):
        return Claude
    elif model_name.startswith("gemini"):
        return Gemini
    elif any(model_name.startswith(x) for x in ["llama-3.3", "llama-3.1", "mixtral"]):
        return Groq
    elif model_name.startswith("deepseek"):
        return DeepSeek
    elif model_name.startswith("grok"):
        return xAI
    elif model_name.startswith("sonar"):
        return Perplexity

    # Optional models (imported on demand)
    try:
        # Cohere models
        if model_name.startswith("command"):
            from agno.models.cohere import Cohere

            return Cohere
        # Fireworks models
        elif "fireworks" in model_name:
            from agno.models.fireworks import Fireworks

            return Fireworks
        # HuggingFace models
        elif "/" in model_name and "llama" in model_name.lower():
            from agno.models.huggingface import HuggingFace

            return HuggingFace
        # Mistral models
        elif model_name.startswith("mistral"):
            from agno.models.mistral import MistralChat

            return MistralChat
        # NVIDIA models
        elif model_name.startswith("meta/"):
            from agno.models.nvidia import Nvidia

            return Nvidia
        # Ollama models (local)
        elif model_name in ["llama3", "mistral", "codellama"]:
            from agno.models.ollama import Ollama

            return Ollama
        # OpenRouter models
        elif "/" in model_name and any(
            x in model_name for x in ["openai/", "anthropic/"]
        ):
            from agno.models.openrouter import OpenRouter

            return OpenRouter
        # Together models
        elif "togethercomputer/" in model_name:
            from agno.models.together import Together

            return Together
        # Cerebras models
        elif model_name.startswith("llama3.1"):
            from agno.models.cerebras import Cerebras

            return Cerebras
        # DeepInfra models
        elif "meta-llama/" in model_name:
            from agno.models.deepinfra import DeepInfra

            return DeepInfra
        # SambaNova models
        elif "Meta-Llama" in model_name:
            from agno.models.sambanova import Sambanova

            return Sambanova
        else:
            raise ValueError(f"Unknown model: {model_name}")
    except ImportError as e:
        raise ImportError(
            f"Provider for model '{model_name}' requires additional dependencies: {str(e)}"
        )


@mcp.tool()
def get_models() -> Dict[str, List[str]]:
    """
    Detect available AI models based on environment variables.

    Returns a dictionary with categories:
    - all_models: List of all available models
    - text_models: Models that can generate text
    - image_models: Models that can generate images
    - by_provider: Models grouped by provider
    """
    available_models = {
        "all_models": [],
        "text_models": [],
        "image_models": [],
        "by_provider": {},
    }

    for provider, config in MODEL_PROVIDERS.items():
        if os.getenv(config["env_key"]):
            available_models["all_models"].extend(config["models"])
            available_models["text_models"].extend(config["text_models"])
            available_models["image_models"].extend(config["image_models"])
            available_models["by_provider"][provider] = config["models"]

    return available_models


@mcp.tool()
async def outsource_text(model: str, prompt: str) -> str:
    """
    Create an Agno agent with the specified model and generate text.

    Args:
        model: The model name (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
        prompt: The prompt to send to the model

    Returns:
        The generated text response
    """
    # Validate model is available
    available = get_models()
    if model not in available["all_models"]:
        return f"Error: Model '{model}' is not available. Available models: {', '.join(available['all_models'])}"

    if model not in available["text_models"]:
        return f"Error: Model '{model}' does not support text generation."

    try:
        # Get the appropriate model class
        model_class = get_model_class(model)

        # Create the agent
        agent = Agent(
            model=model_class(id=model),
            name="Text Generation Agent",
            instructions="You are a helpful AI assistant. Respond to the user's prompt directly and concisely.",
        )

        # Run the agent and get response
        response = await agent.arun(prompt)

        # Extract the text content from the response
        if hasattr(response, "content"):
            return response.content
        else:
            return str(response)

    except Exception as e:
        return f"Error generating text: {str(e)}"


@mcp.tool()
async def outsource_image(model: str, prompt: str) -> str:
    """
    Create an Agno agent with the specified model and generate an image.

    Args:
        model: The model name (e.g., "dall-e-3", "dall-e-2")
        prompt: The image generation prompt

    Returns:
        Base64 encoded image data or error message
    """
    # Validate model is available
    available = get_models()
    if model not in available["all_models"]:
        return f"Error: Model '{model}' is not available. Available models: {', '.join(available['all_models'])}"

    if model not in available["image_models"]:
        return f"Error: Model '{model}' does not support image generation. Available image models: {', '.join(available['image_models'])}"

    try:
        # For OpenAI image generation, we need to use their API directly
        # as Agno agents are primarily for text-based interactions
        if model in ["dall-e-3", "dall-e-2"]:
            import openai

            client = openai.OpenAI()

            # Generate image
            response = client.images.generate(
                model=model,
                prompt=prompt,
                n=1,
                size="1024x1024" if model == "dall-e-3" else "512x512",
                response_format="b64_json",
            )

            # Return base64 encoded image
            return f"data:image/png;base64,{response.data[0].b64_json}"
        else:
            return (
                f"Error: Image generation for model '{model}' is not yet implemented."
            )

    except Exception as e:
        return f"Error generating image: {str(e)}"


def main():
    """Entry point for the MCP server."""
    # Run the server
    mcp.run()


if __name__ == "__main__":
    main()
