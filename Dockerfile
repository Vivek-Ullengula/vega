# project-vega/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the essential packaging files
COPY pyproject.toml README.md ./
# Copy the source code directory (required for 'pip install .')
COPY coaction_agent_platform/ ./coaction_agent_platform/

# Install the package and its dependencies
RUN pip install --no-cache-dir .

# Copy the rest of the application (UI, scripts, etc.)
COPY . .

# Set environment variables for production
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# AWS App Runner / ECS standard port
EXPOSE 8080

# Start the unified platform (UI + API)
CMD ["python", "ui/gradio_app.py"]
