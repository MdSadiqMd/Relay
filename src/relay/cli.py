"""relay CLI — Cryptographically Versioned Semantic Memory.

Commands:
    ingest    — Ingest a document into relay
"""

import json
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from relay.config import CONFIG

app = typer.Typer(
    name="relay",
    help="Cryptographically Versioned Semantic Memory with Qdrant",
    no_args_is_help=True,
)
epoch_app = typer.Typer(help="Epoch management commands")
app.add_typer(epoch_app, name="epoch")

console = Console()


@app.command()
def ingest(
    file: str = typer.Option(..., "--file", "-f", help="Path to file to ingest"),
    tenant: str = typer.Option(
        CONFIG.default_tenant, "--tenant", "-t", help="Tenant ID"
    ),
    valid_from: str = typer.Option(
        ..., "--valid-from", help="Validity start date (YYYY-MM-DD)"
    ),
    valid_to: Optional[str] = typer.Option(
        None, "--valid-to", help="Validity end date"
    ),
    supersedes: Optional[str] = typer.Option(
        None, "--supersedes", help="Doc ID this supersedes"
    ),
    tags: Optional[str] = typer.Option(
        None, "--tags", help="Comma-separated semantic tags"
    ),
):
    """Ingest a document into relay."""
    from relay.ingest import ingest_file

    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    with console.status("[bold green]Ingesting document..."):
        result = ingest_file(
            file_path=file,
            tenant_id=tenant,
            valid_from=valid_from,
            valid_to=valid_to,
            supersedes=supersedes,
            semantic_tags=tag_list,
        )

    panel = Panel(
        f"[bold green]✓ Document ingested[/]\n\n"
        f"  [cyan]doc_id[/]:        {result.doc_id}\n"
        f"  [cyan]epoch_id[/]:      {result.epoch_id}\n"
        f"  [cyan]content_hash[/]:  {result.content_hash[:16]}...\n"
        f"  [cyan]embed_hash[/]:    {result.embedding_hash[:16]}...\n"
        f"  [cyan]merkle_root[/]:   {result.merkle_root[:16]}...\n"
        f"  [cyan]source[/]:        {result.source_file}",
        title="[bold]relay ingest[/]",
        border_style="green",
    )
    console.print(panel)


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
