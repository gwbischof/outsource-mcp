"""
MCP server for outsourcing tasks to AI agents.

This server provides tools to:
1. Detect available AI models based on environment variables
2. Generate images using Agno agents with specified models
3. Generate text using Agno agents with specified models
"""

import os

from mcp.server.fastmcp import FastMCP
from agno.agent import Agent

from agno.models.openai import OpenAIChat
from agno.models.anthropic import Claude
from agno.models.google import Gemini
from agno.models.groq import Groq
from agno.models.deepseek import DeepSeek
from agno.models.xai import xAI
from agno.models.perplexity import Perplexity
from agno.models.cohere import Cohere
from agno.models.fireworks import Fireworks
from agno.models.huggingface import HuggingFace
from agno.models.mistral import MistralChat
from agno.models.nvidia import Nvidia
from agno.models.ollama import Ollama
from agno.models.openrouter import OpenRouter
from agno.models.sambanova import Sambanova
from agno.models.together import Together
from agno.models.litellm import LiteLLM
from agno.models.vercel import v0
from agno.models.aws import AwsBedrock
from agno.models.azure import AzureAIFoundry
from agno.models.cerebras import Cerebras
from agno.models.meta import Llama
from agno.models.deepinfra import DeepInfra
from agno.models.ibm import WatsonX

# Create the MCP server instance
mcp = FastMCP("outsource-mcp")


def get_model_class(model_name: str):
    """Get the appropriate model class based on model name."""
    # OpenAI models
    if model_name.startswith(("gpt", "dall-e")):
        return OpenAIChat
    # Anthropic models
    elif model_name.startswith("claude"):
        return Claude
    # Google models
    elif model_name.startswith("gemini"):
        return Gemini
    # Groq models
    elif any(model_name.startswith(x) for x in ["llama-3.3", "llama-3.1", "mixtral"]):
        return Groq
    # DeepSeek models
    elif model_name.startswith("deepseek"):
        return DeepSeek
    # xAI models
    elif model_name.startswith("grok"):
        return xAI
    # Perplexity models
    elif model_name.startswith("sonar"):
        return Perplexity
    # Cohere models
    elif model_name.startswith("command"):
        return Cohere
    # Fireworks models
    elif "fireworks" in model_name:
        return Fireworks
    # HuggingFace models
    elif "/" in model_name and "llama" in model_name.lower():
        return HuggingFace
    # Mistral models
    elif model_name.startswith("mistral"):
        return MistralChat
    # NVIDIA models
    elif model_name.startswith("meta/"):
        return Nvidia
    # Ollama models (local)
    elif model_name in ["llama3", "mistral", "codellama"]:
        return Ollama
    # OpenRouter models
    elif "/" in model_name and any(x in model_name for x in ["openai/", "anthropic/"]):
        return OpenRouter
    # Together models
    elif "togethercomputer/" in model_name:
        return Together
    # Cerebras models
    elif model_name.startswith("llama3.1"):
        return Cerebras
    # DeepInfra models
    elif "meta-llama/" in model_name:
        return DeepInfra
    # SambaNova models
    elif "Meta-Llama" in model_name:
        return Sambanova
    # LiteLLM models
    elif "litellm" in model_name.lower():
        return LiteLLM
    # Vercel v0 models
    elif "v0" in model_name.lower():
        return v0
    # AWS Bedrock models
    elif "bedrock" in model_name.lower() or model_name.startswith("amazon."):
        return AwsBedrock
    # Azure AI Foundry models
    elif "azure" in model_name.lower() or model_name.startswith("microsoft"):
        return AzureAIFoundry
    # Meta Llama models
    elif model_name.startswith("meta-llama/") and "LLAMA_API_KEY" in os.environ:
        return Llama
    # IBM WatsonX models
    elif "watsonx" in model_name.lower() or "watson" in model_name.lower():
        return WatsonX
    else:
        raise ValueError(f"Unknown model: {model_name}")



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
