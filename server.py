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
    Create an Agno agent with the specified provider and model to generate text.

    Args:
        provider: The provider name (e.g., "openai", "anthropic", "google")
        model: The model name (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
        prompt: The prompt to send to the model

    Returns:
        The generated text response
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
    Generate an image using the specified provider and model.

    Args:
        provider: The provider name (e.g., "openai")
        model: The model name (e.g., "dall-e-3", "dall-e-2")
        prompt: The image generation prompt

    Returns:
        Base64 encoded image data or error message
    """
    try:
        provider_lower = provider.lower()

        # Currently only OpenAI supports image generation through our integration
        if provider_lower == "openai":
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
                return f"Error: Model '{model}' is not a supported OpenAI image generation model. Supported models: dall-e-3, dall-e-2"
        else:
            return f"Error: Provider '{provider}' does not support image generation through this tool. Currently only 'openai' is supported."

    except Exception as e:
        return f"Error generating image: {str(e)}"


def main():
    """Entry point for the MCP server."""
    # Run the server
    mcp.run()


if __name__ == "__main__":
    main()
