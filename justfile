set dotenv-load := false
set export := true

# Default: show available recipes
default:
    @just --list

# Install dependencies with uv
install:
    uv sync

# Start Qdrant (docker compose)
up:
    docker compose up -d qdrant
    @echo "Waiting for Qdrant to be healthy..."
    @until curl -sf http://localhost:6333/healthz > /dev/null 2>&1; do sleep 1; done
    @echo "✓ Qdrant is ready at http://localhost:6333"

# Stop all services
down:
    docker compose down

# Stop and remove all data (clean slate)
nuke:
    docker compose down -v
    @echo "✓ All data wiped"

# Build the relay Docker image
build:
    docker compose build relay

# Rebuild without cache
rebuild:
    docker compose build --no-cache relay

# Ingest a document
ingest file tenant="default" valid_from="2024-01-01":
    uv run relay ingest --tenant {{tenant}} --file {{file}} --valid-from {{valid_from}}

# Supersede a document
supersede old new tenant="default":
    uv run relay supersede --old-doc {{old}} --new-doc {{new}} --tenant {{tenant}}

# Query semantic memory (latest)
query text tenant="default":
    uv run relay query --text "{{text}}" --tenant {{tenant}}

# Query at a specific time
query-at text timestamp tenant="default":
    uv run relay query --text "{{text}}" --at {{timestamp}} --tenant {{tenant}}

# Query at a specific epoch
query-epoch text epoch tenant="default":
    uv run relay query --text "{{text}}" --epoch {{epoch}} --tenant {{tenant}}

# Diff two epochs
diff from to tenant="default":
    uv run relay diff --from {{from}} --to {{to}} --tenant {{tenant}}

# Verify a retrieval
verify request_id tenant="default":
    uv run relay verify --request-id {{request_id}} --tenant {{tenant}}

# Show epoch status
epochs tenant="default":
    uv run relay epoch status --tenant {{tenant}}

# Show specific epoch
epoch id tenant="default":
    uv run relay epoch status --epoch {{id}} --tenant {{tenant}}

# Run relay CLI in Docker
docker-relay *args:
    docker compose run --rm --profile cli relay {{args}}

# Run all tests (unit + integration, needs Qdrant for integration)
test: up
    uv run pytest tests/ -v

# Run unit tests only (no Qdrant needed)
test-unit:
    uv run pytest tests/ -v -k "not integration"

# Run integration tests only (needs Qdrant)
test-int: up
    uv run pytest tests/test_integration.py -v

# Type-check with mypy
typecheck:
    uv run mypy src/relay

# Lint with ruff
lint:
    uv run ruff check src/ tests/ examples/

# Format with ruff
fmt:
    uv run ruff format src/ tests/ examples/

# Run all checks (lint + typecheck + tests)
ci: lint typecheck test

# Run the full demo scenario from the spec (§12)
demo: up
    uv run python examples/epoch_lifecycle.py && uv run python examples/twelve_labs_video_demo.py

# Reset Qdrant collections (clean slate for demo re-runs)
reset:
    uv run python -c "\
    from qdrant_client import QdrantClient; \
    c = QdrantClient(host='localhost', port=6333); \
    [c.delete_collection(col) for col in ['relay_documents', 'relay_epochs', 'relay_retrieval_logs']]; \
    print('✓ All collections deleted')"

# Run relay CLI directly (pass any args)
run *args:
    uv run relay {{args}}

# Check Qdrant health
health:
    @curl -sf http://localhost:6333/healthz && echo " ✓ Qdrant healthy" || echo " ✗ Qdrant not reachable"

# Show help
help:
    uv run relay --help
