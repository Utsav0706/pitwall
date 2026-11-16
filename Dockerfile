# The routing service. Small on purpose: no torch, no MLflow server, no model inside.
# The model comes from the MLflow registry at run time and is served by Ollama, which
# runs outside this container (OLLAMA_BASE_URL, e.g. http://host.docker.internal:11434).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY prompts ./prompts
RUN pip install ".[serve]" && useradd --create-home pitwall && mkdir -p /app/data && chown -R pitwall /app/data
USER pitwall

ENV DATABASE_URL=sqlite:////app/data/pitwall.db \
    MODEL_CACHE_DIR=/app/data/models
EXPOSE 8003
CMD ["pitwall", "serve", "--host", "0.0.0.0"]
