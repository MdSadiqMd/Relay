"""TwelveLabs video ingestion: analyze → ingest → query → diff → verify → RAG.

Requires:
  - A valid ``RELAY_TWELVE_LABS_API_KEY`` environment variable
  - Qdrant running on localhost:6333
  - Internet access to reach the TwelveLabs API and public video URLs

Usage:
    export RELAY_TWELVE_LABS_API_KEY="tlk_..."
    uv run python examples/twelve_labs_video_demo.py
"""

import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Optional

PRIV = Path("src/priv")
TENANT = f"video_demo_{uuid.uuid4().hex[:8]}"

# Track ingested doc IDs for supersede phase
_INGESTED_DOCS: dict[str, str] = {}


VIDEO_URLS = {
    "bigbuckbunny": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_1MB.mp4",
    "sintel": "https://test-videos.co.uk/vids/sintel/mp4/h264/720/Sintel_720_10s_1MB.mp4",
}


def cli(*args):
    subprocess.run(["uv", "run", "relay", *args], check=True)


def header(msg):
    sys.stderr.write(f"{msg}\n{'─' * 60}\n")
    sys.stderr.flush()


def step(msg):
    sys.stderr.write(f"  ▸ {msg}\n")
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
    api_key = os.environ.get("RELAY_TWELVE_LABS_API_KEY", "")
    if not api_key:
        sys.stderr.write(
            "\n  ✗ RELAY_TWELVE_LABS_API_KEY is not set.\n"
            "    Set it in your environment or .env file:\n\n"
            '    export RELAY_TWELVE_LABS_API_KEY="tlk_..."\n\n'
        )
        sys.exit(1)

    sys.stderr.write(f"\n  TwelveLabs Video Demo\n")
    sys.stderr.write(f"  Tenant: {TENANT}\n")
    sys.stderr.write(f"  Videos: Big Buck Bunny + Sintel\n\n")

    # Phase 1: Analyze & ingest Big Buck Bunny
    header("Phase 1: Analyze Big Buck Bunny with Pegasus → ingest into relay")
    analyze_and_ingest(
        video_key="bigbuckbunny",
        title="Big Buck Bunny",
        valid_from="2026-01-15",
        tags=["animation", "blender", "short-film"],
        prompt=(
            "Describe the visual scenes, characters, and storyline of this animated short film. "
            "Identify key scenes, character interactions, and the overall narrative arc."
        ),
    )

    # Phase 2: Analyze & ingest Sintel
    header("Phase 2: Analyze Sintel with Pegasus → ingest into relay")
    analyze_and_ingest(
        video_key="sintel",
        title="Sintel",
        valid_from="2026-03-01",
        tags=["animation", "blender", "short-film"],
        prompt=(
            "Describe the visual scenes, characters, and storyline of this animated short film. "
            "Identify key scenes, visual style, and thematic elements."
        ),
    )

    # Phase 3: Supersede — Big Buck Bunny → Sintel
    header("Phase 3: Supersede — Big Buck Bunny superseded by Sintel")
    bbb_doc_id = _INGESTED_DOCS.get("bigbuckbunny", "")
    ed_doc_id = _INGESTED_DOCS.get("sintel", "")
    step(f"Marking '{bbb_doc_id[:28]}…' as superseded by '{ed_doc_id[:28]}…'")
    cli(
        "supersede",
        "--tenant",
        TENANT,
        "--old-doc",
        bbb_doc_id,
        "--new-doc",
        ed_doc_id,
    )

    # Phase 4: Time-travel queries
    header("Phase 4a: Epoch-pinned query — epoch 1 (Big Buck Bunny only)")
    step("Returns only the Big Buck Bunny analysis from the first ingest")
    cli(
        "query",
        "--tenant",
        TENANT,
        "--text",
        "animated bunny forest scenes",
        "--epoch",
        "1",
    )

    header("Phase 4b: Latest query — 'dream sequence visual style'")
    step("Returns Elephants Dream as primary result (supersedes Big Buck Bunny)")
    cli("query", "--tenant", TENANT, "--text", "dream sequence visual style")

    # Phase 5: Semantic diff
    header("Phase 5: Semantic diff — epoch 1 vs epoch 2")
    step("Comparing Big Buck Bunny analysis vs Elephants Dream analysis")
    cli("diff", "--tenant", TENANT, "--from", "1", "--to", "2")

    # Phase 6: Epoch status
    header("Phase 6: Epoch status")
    cli("epoch", "status", "--tenant", TENANT)

    # Phase 7: LlamaIndex RAG over video content
    header("Phase 7: RAG retrieval over video-derived content")
    _phase_rag()

    sys.stderr.write(f"\n  {'═' * 60}\n")
    sys.stderr.write(f"  ✓ Demo complete (tenant={TENANT})\n\n")


def analyze_and_ingest(
    video_key: str,
    title: str,
    valid_from: str,
    tags: list[str],
    prompt: str,
):
    """Step 1: Analyze video with TwelveLabs Pegasus → transcript.
    Step 2: Persist transcript to src/priv/ for RAG resolution.
    Step 3: Ingest into relay with full epoch/Merkle commitment.
    """
    from pkg.twelvelabs.client import analyze_video

    video_url = VIDEO_URLS[video_key]
    source_file = f"video_{video_key}.md"
    doc_id = f"{video_key}_{uuid.uuid4().hex[:8]}"

    step(f"Analyzing video via TwelveLabs Pegasus: {video_url}")
    step(f"  Prompt: {prompt[:60]}…")
    step("  Waiting for Pegasus analysis... (may take ~15s)")

    transcript = analyze_video(
        video_url=video_url,
        prompt=prompt,
    )

    # Persist transcript so the RAG phase can resolve full text
    transcript_path = PRIV / source_file
    transcript_path.write_text(transcript, encoding="utf-8")
    step(f"Transcript saved to {transcript_path} ({len(transcript)} chars)")

    # Ingest into relay
    from relay.ingest import ingest_text

    result = ingest_text(
        text=transcript,
        tenant_id=TENANT,
        valid_from=valid_from,
        semantic_tags=tags + ["video"],
        source_file=source_file,
        doc_id=doc_id,
    )

    _INGESTED_DOCS[video_key] = result.doc_id
    sys.stderr.write(
        f"  ✓ Ingested: doc_id={result.doc_id}  "
        f"epoch={result.epoch_id}  "
        f"merkle_root={result.merkle_root[:16]}...\n"
    )


def _phase_rag():
    """LlamaIndex-powered RAG over video content."""
    step("Retrieving video context for RAG synthesis...")

    from pkg.llamaindex import RelayRetriever

    retriever = RelayRetriever(
        tenant_id=TENANT,
        top_k=3,
        text_resolver=_resolve_text,
    )

    for query_text in [
        "What characters appear in the forest scene?",
        "What is the visual style or setting of the second video?",
    ]:
        nodes, epoch_id = retriever.retrieve_with_epoch(query_text)
        step(f'  Query: "{query_text}"')
        step(f"  Retrieved {len(nodes)} results from epoch {epoch_id}")
        for nws in nodes:
            meta = nws.node.metadata
            preview = (
                nws.node.text[:100].replace("\n", " ") if nws.node.text else "(no text)"
            )
            sys.stderr.write(
                f"    └ doc_id={nws.node.id_[:28]:28s}  "
                f"score={nws.score:.4f}  "
                f"source={meta.get('source_file', '-'):20s}\n"
            )


if __name__ == "__main__":
    main()
