"""Epoch lifecycle: ingest → supersede → query → diff → verify.

Runs the relay CLI commands directly — output is the real CLI.

Usage:
    uv run python examples/epoch_lifecycle.py
"""

import subprocess
import sys
import uuid

PRIV = "src/priv"
TENANT = f"demo_{uuid.uuid4().hex[:8]}"


def cli(*args):
    subprocess.run(["uv", "run", "relay", *args], check=True)


def header(msg):
    sys.stderr.write(f"\n{msg}\n{'─' * len(msg)}\n")
    sys.stderr.flush()


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

    sys.stderr.write(f"\n  ✓ Demo complete (tenant={TENANT})\n\n")


if __name__ == "__main__":
    main()
