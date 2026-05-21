# syntax=docker/dockerfile:1

# Stage 1: Build
FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev
COPY src/ src/
COPY README.md ./

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Pre-download the embedding model so it's baked into the image
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Stage 2: Runtime
FROM python:3.11-slim AS runtime
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/pyproject.toml /app/
COPY --from=builder /app/README.md /app/
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface
COPY src/priv/ src/priv/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["relay"]
CMD ["--help"]
