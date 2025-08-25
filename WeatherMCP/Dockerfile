FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

RUN (uv venv .venv) && (. .venv/bin/activate) && (uv pip install mcp_weather_server)

CMD ["uv","run","mcp_weather_server"]
