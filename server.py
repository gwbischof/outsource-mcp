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

# Provider to model class mapping
PROVIDER_MODEL_MAP = {
    "openai": OpenAIChat,
    "anthropic": Claude,
    "google": Gemini,
    "groq": Groq,
    "deepseek": DeepSeek,
    "xai": xAI,
    "perplexity": Perplexity,
    "cohere": Cohere,
    "fireworks": Fireworks,
    "huggingface": HuggingFace,
    "mistral": MistralChat,
    "nvidia": Nvidia,
    "ollama": Ollama,
    "openrouter": OpenRouter,
    "sambanova": Sambanova,
    "together": Together,
    "litellm": LiteLLM,
    "vercel": v0,
    "v0": v0,
    "aws": AwsBedrock,
    "bedrock": AwsBedrock,
    "azure": AzureAIFoundry,
    "cerebras": Cerebras,
    "meta": Llama,
    "deepinfra": DeepInfra,
    "ibm": WatsonX,
    "watsonx": WatsonX,
}


@mcp.tool()
async def outsource_text(provider: str, model: str, prompt: str) -> str:
    """
    Delegate text generation to another AI model. Use this when you need capabilities
    or perspectives from a different model than yourself.

    Args:
        provider: The AI provider to use (e.g., "openai", "anthropic", "google", "groq")
        model: The specific model identifier (e.g., "gpt-4o", "claude-3-5-sonnet-20241022", "gemini-2.0-flash-exp")
        prompt: The instruction or query to send to the external model

    Returns:
        The text response from the external model, or an error message if the request fails

    Example usage:
        To get a different perspective: provider="anthropic", model="claude-3-5-sonnet-20241022", prompt="Analyze this problem from a different angle..."
        To leverage specialized models: provider="deepseek", model="deepseek-coder", prompt="Write optimized Python code for..."
    """
    try:
        # Get the appropriate model class based on provider
        provider_lower = provider.lower()

        if provider_lower not in PROVIDER_MODEL_MAP:
            raise ValueError(f"Unknown provider: {provider}")

        model_class = PROVIDER_MODEL_MAP[provider_lower]

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
async def outsource_image(provider: str, model: str, prompt: str) -> str:
    """
    Delegate image generation to an external AI model. Use this when you need to create
    visual content.

    Args:
        provider: The AI provider to use (currently only "openai" is supported)
        model: The image model to use ("dall-e-3" for high quality, "dall-e-2" for faster/cheaper)
        prompt: A detailed description of the image you want to generate

    Returns:
        The URL of the generated image, which can be shared with users or used in responses

    Example usage:
        For high-quality images: provider="openai", model="dall-e-3", prompt="A photorealistic rendering of..."
        For quick concepts: provider="openai", model="dall-e-2", prompt="A simple sketch showing..."

    Note: Only OpenAI currently supports image generation. Other providers will return an error.
    """
    try:
        provider_lower = provider.lower()

        # Currently only OpenAI supports image generation through our integration
        if provider_lower == "openai":
            if model in ["dall-e-3", "dall-e-2"]:
                import openai

                # Use OpenAI directly for more control
                client = openai.AsyncOpenAI()

                # Generate image with appropriate parameters for each model
                try:
                    if model == "dall-e-3":
                        response = await client.images.generate(
                            model=model,
                            prompt=prompt,
                            n=1,
                            size="1024x1024",
                            response_format="url"
                        )
                    else:  # dall-e-2
                        response = await client.images.generate(
                            model=model,
                            prompt=prompt,
                            n=1,
                            size="512x512",
                            response_format="url"
                        )

                    # Get the image URL
                    image_url = response.data[0].url
                    return image_url

                except openai.OpenAIError as e:
                    return f"Error: OpenAI API error - {str(e)}"

            else:
                return f"Error: Model '{model}' is not a supported OpenAI image generation model. Supported models: dall-e-3, dall-e-2"
        else:
            return f"Error: Provider '{provider}' does not support image generation through this tool. Currently only 'openai' is supported."

    except Exception as e:
        return f"Error generating image: {str(e)}"


def main():
    """Entry point for the MCP server."""
    # Run the server
    mcp.run(transport='stdio')


if __name__ == "__main__":
    main()
