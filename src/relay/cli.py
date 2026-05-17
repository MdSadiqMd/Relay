"""relay CLI — Cryptographically Versioned Semantic Memory.

Commands:
    ingest    — Ingest a document into relay
    epoch     — Epoch management (status, list)
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


@epoch_app.command("status")
def epoch_status(
    epoch: Optional[int] = typer.Option(None, "--epoch", help="Specific epoch ID"),
    tenant: str = typer.Option(
        CONFIG.default_tenant, "--tenant", "-t", help="Tenant ID"
    ),
    output_json: bool = typer.Option(False, "--json", help="Output raw JSON"),
):
    """Show epoch status and details."""
    from relay.collections import ensure_collections
    from relay.epochs import get_epoch, list_epochs

    client = ensure_collections()

    if epoch is not None:
        ep = get_epoch(client, tenant, epoch)
        if ep is None:
            console.print(f"[red]Epoch {epoch} not found for tenant {tenant}[/]")
            raise typer.Exit(1)

        if output_json:
            console.print_json(ep.model_dump_json(indent=2))
            return

        panel = Panel(
            f"  [cyan]epoch_id[/]:      {ep.epoch_id}\n"
            f"  [cyan]created_at[/]:    {ep.created_at}\n"
            f"  [cyan]model_version[/]: {ep.model_version}\n"
            f"  [cyan]merkle_root[/]:   {ep.merkle_root[:24]}...\n"
            f"  [cyan]doc_count[/]:     {ep.doc_count}\n"
            f"  [cyan]parent_epoch[/]:  {ep.parent_epoch or '-'}\n"
            f"  [cyan]tenant_id[/]:     {ep.tenant_id}",
            title=f"[bold]Epoch {epoch}[/]",
            border_style="blue",
        )
        console.print(panel)
    else:
        epochs = list_epochs(client, tenant)
        if not epochs:
            console.print(f"[dim]No epochs found for tenant '{tenant}'[/]")
            return

        if output_json:
            data = [ep.model_dump() for ep in epochs]
            console.print_json(json.dumps(data, indent=2, default=str))
            return

        table = Table(
            title=f"Epochs for tenant '{tenant}'",
            show_header=True,
            header_style="bold blue",
        )
        table.add_column("ID", justify="right", width=5)
        table.add_column("Created At", min_width=20)
        table.add_column("Model", min_width=18)
        table.add_column("Docs", justify="right", width=6)
        table.add_column("Merkle Root", min_width=20)
        table.add_column("Parent", justify="right", width=7)

        for ep in epochs:
            table.add_row(
                str(ep.epoch_id),
                ep.created_at,
                ep.model_version,
                str(ep.doc_count),
                ep.merkle_root[:20] + "...",
                str(ep.parent_epoch or "-"),
            )

        console.print(table)


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
