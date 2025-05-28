FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen

# Copy source code
COPY server.py ./

# Expose port
EXPOSE 8000

# Set environment variables for MCP
ENV PYTHONUNBUFFERED=1

# Run the MCP server
CMD ["uv", "run", "python", "server.py"]
