"""Epoch lifecycle: ingest → supersede → query → diff → verify → RAG.

Demonstrates the full relay product flow from ingestion through to
LlamaIndex-powered RAG synthesis using a local HuggingFace model.

Architecture within relay::

    pkg.llamaindex (RAG layer)              ← Phase 7
      RetrieverQueryEngine
        RelayRetriever, LocalHFLLM
          │ calls relay.query
    relay core (retrieval engine)           ← Phases 1-6
      query, epoch, merkle, diff, verify
          │ stores vectors
    Qdrant (vector DB)

Usage:
    uv run python examples/epoch_lifecycle.py
"""

import subprocess
import sys
import uuid
from pathlib import Path
from typing import Optional

PRIV = Path("src/priv")
TENANT = f"demo_{uuid.uuid4().hex[:8]}"


def cli(*args):
    subprocess.run(["uv", "run", "relay", *args], check=True)


def header(msg):
    sys.stderr.write(f"\n{msg}\n{'─' * len(msg)}\n")
    sys.stderr.flush()


def _resolve_text(doc_id: str, source_file: Optional[str]) -> str:
    """Load document text from ``src/priv/{source_file}``."""
    if source_file is None:
        return ""
    path = PRIV / source_file
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def main():
    header("Phase 1: Ingest Kafka architecture docs (2024)")
    cli(
        "ingest",
        "--tenant",
        TENANT,
        "--file",
        f"{PRIV}/kafka.md",
        "--valid-from",
        "2024-01-01",
    )

    header("Phase 2a: Ingest NATS migration docs (2025)")
    cli(
        "ingest",
        "--tenant",
        TENANT,
        "--file",
        f"{PRIV}/nats_migration.md",
        "--valid-from",
        "2025-01-01",
    )

    header("Phase 2b: Supersede Kafka → NATS")
    cli(
        "supersede",
        "--tenant",
        TENANT,
        "--old-doc",
        "kafka.md",
        "--new-doc",
        "nats_migration.md",
    )

    header("Phase 3: Historical query — 'event bus' at 2025-06-01")
    cli(
        "query",
        "--tenant",
        TENANT,
        "--text",
        "event bus architecture",
        "--at",
        "2025-06-01",
    )

    header("Phase 4: Latest query — 'event bus' (current state)")
    cli("query", "--tenant", TENANT, "--text", "event bus architecture")

    header("Phase 5: Semantic diff — epoch 1 vs 2")
    cli("diff", "--tenant", TENANT, "--from", "1", "--to", "2")

    header("Phase 6: Epoch status")
    cli("epoch", "status", "--tenant", TENANT)

    header("Phase 7a: LlamaIndex retrieval (via pkg.llamaindex.RelayRetriever)")
    phase_7a_retrieval()

    header(
        "Phase 7b: RAG synthesis — local HF model (via pkg.llamaindex.create_query_engine)"
    )
    phase_7b_rag_local()

    header("Phase 7c: RAG synthesis — cloud LLM (via pkg.llamaindex, requires API key)")
    phase_7c_rag_cloud()

    sys.stderr.write(f"\n  ✓ Demo complete (tenant={TENANT})\n\n")


def phase_7a_retrieval():
    """LlamaIndex consumption layer: RelayRetriever returns nodes with resolved text."""
    from pkg.llamaindex import RelayRetriever

    retriever = RelayRetriever(
        tenant_id=TENANT,
        top_k=3,
        text_resolver=_resolve_text,
    )
    nodes, epoch_id = retriever.retrieve_with_epoch("Why NATS over Kafka?")
    for nws in nodes:
        print(f"\n  doc_id={nws.node.id_}")
        print(f"  score={nws.score:.4f}")
        print(f"  source_file={nws.node.metadata['source_file']}")
        print(f"  text_preview={nws.node.text[:80]}...")
        print(f"  epoch_id={epoch_id}")


def phase_7b_rag_local():
    """Relay RAG layer: local HuggingFace model via pkg.llamaindex."""
    from pkg.llamaindex import create_query_engine

    engine = create_query_engine(
        tenant_id=TENANT,
        top_k=5,
        text_resolver=_resolve_text,
    )

    for q in [
        "Why are we moving from Kafka to NATS?",
        "What were the operational issues with Kafka?",
    ]:
        print(f"\n  Query: {q}")
        print(f"  {'─' * (len(q) + 8)}")
        print(f"  Answer: {engine.query(q)}\n")


def phase_7c_rag_cloud():
    """Relay RAG layer: cloud LLM provider via pkg.llamaindex."""
    import os
    from pkg.llamaindex import create_query_engine

    provider = os.environ.get("RELAY_LLM_PROVIDER", "") or "local"
    if provider not in ("openai", "anthropic", "local"):
        print(
            "\n  SKIPPED — set RELAY_LLM_PROVIDER=openai or =anthropic with a valid API key"
        )
        return

    engine = create_query_engine(
        tenant_id=TENANT,
        top_k=5,
        text_resolver=_resolve_text,
        llm_provider=provider,
    )
    print(f"\n  Provider: {provider}")
    print("  Query: Why NATS over Kafka?")
    print(f"  {'─' * 35}")
    print(f"  Answer: {engine.query('Why are we moving from Kafka to NATS?')}\n")


if __name__ == "__main__":
    main()
