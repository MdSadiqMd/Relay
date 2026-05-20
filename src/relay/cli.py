"""relay CLI — Cryptographically Versioned Semantic Memory.

Commands:
    ingest    — Ingest a document into relay
    supersede — Supersede one document with another
    query     — Temporal semantic query (--at, --epoch, latest)
    diff      — Compare semantic states between epochs
    verify    — Verify retrieval against Merkle commitment
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
    supersedes: Optional[list[str]] = typer.Option(
        None, "--supersedes", help="Doc ID(s) this supersedes (repeatable)"
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


@app.command()
def supersede(
    old_doc: list[str] = typer.Option(
        ..., "--old-doc", help="Old document(s) (filename or doc_id, repeatable)"
    ),
    new_doc: str = typer.Option(
        ..., "--new-doc", help="New document (filename or doc_id)"
    ),
    tenant: str = typer.Option(
        CONFIG.default_tenant, "--tenant", "-t", help="Tenant ID"
    ),
):
    """Supersede one or more old documents with a new one."""
    from relay.supersede import supersede as do_supersede

    with console.status("[bold yellow]Superseding document..."):
        result = do_supersede(
            old_doc_refs=old_doc,
            new_doc_ref=new_doc,
            tenant_id=tenant,
        )

    old_ids = ", ".join(result.old_doc_ids)
    panel = Panel(
        f"[bold yellow]⟳ Document superseded[/]\n\n"
        f"  [cyan]old_docs[/]:    {old_ids}\n"
        f"  [cyan]new_doc[/]:     {result.new_doc_id}\n"
        f"  [cyan]old_valid_to[/]: {result.old_valid_to}\n"
        f"  [cyan]new_epoch[/]:   {result.epoch.epoch_id}\n"
        f"  [cyan]merkle_root[/]: {result.epoch.merkle_root[:16]}...",
        title="[bold]relay supersede[/]",
        border_style="yellow",
    )
    console.print(panel)


@app.command()
def query(
    text: str = typer.Option(..., "--text", help="Query text"),
    tenant: str = typer.Option(
        CONFIG.default_tenant, "--tenant", "-t", help="Tenant ID"
    ),
    at: Optional[str] = typer.Option(
        None, "--at", help="Retrieve as-of timestamp (YYYY-MM-DD)"
    ),
    epoch: Optional[int] = typer.Option(None, "--epoch", help="Pin to specific epoch"),
    retrieval_policy: str = typer.Option(
        "dense", "--retrieval-policy", help="Retrieval policy"
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
    output_json: bool = typer.Option(False, "--json", help="Output raw JSON"),
):
    """Execute a temporal semantic query."""
    from relay.query import query as do_query

    with console.status("[bold blue]Querying semantic memory..."):
        result = do_query(
            text=text,
            tenant_id=tenant,
            at=at,
            epoch_id=epoch,
            retrieval_policy=retrieval_policy,
            top_k=top_k,
        )

    if output_json:
        console.print_json(result.model_dump_json(indent=2))
        return

    # Header
    mode = "latest"
    if at:
        mode = f"as-of {at}"
    elif epoch:
        mode = f"epoch {epoch}"

    console.print(f"\n[bold blue]relay query[/] — [dim]{mode}[/]")
    console.print(f'  [cyan]query[/]: "{text}"')
    console.print(
        f"  [cyan]epoch[/]: {result.epoch_id}  [cyan]request_id[/]: {result.request_id[:12]}..."
    )
    console.print()

    if not result.results:
        console.print("[dim]No results found.[/]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("doc_id", min_width=20)
    table.add_column("score", justify="right", width=8)
    table.add_column("source", min_width=15)
    table.add_column("valid_from", width=12)
    table.add_column("valid_to", width=12)
    table.add_column("status", width=12)

    for i, r in enumerate(result.results, 1):
        status = "[green]active[/]"
        if r.superseded_by:
            status = "[red]superseded[/]"
        elif r.valid_to:
            status = "[yellow]expired[/]"

        table.add_row(
            str(i),
            r.doc_id,
            f"{r.score:.4f}",
            r.source_file or "-",
            r.valid_from or "-",
            r.valid_to or "-",
            status,
        )

    console.print(table)
    console.print(f"\n  [dim]{result.result_count} results returned[/]\n")


@app.command()
def diff(
    from_epoch: int = typer.Option(..., "--from", help="Source epoch ID"),
    to_epoch: int = typer.Option(..., "--to", help="Target epoch ID"),
    tenant: str = typer.Option(
        CONFIG.default_tenant, "--tenant", "-t", help="Tenant ID"
    ),
    output_json: bool = typer.Option(False, "--json", help="Output raw JSON"),
):
    """Compare semantic states between two epochs."""
    from relay.diff import diff_epochs

    with console.status("[bold magenta]Computing semantic diff..."):
        result = diff_epochs(
            from_epoch=from_epoch,
            to_epoch=to_epoch,
            tenant_id=tenant,
        )

    if output_json:
        console.print_json(result.model_dump_json(indent=2))
        return

    console.print(f"\n[bold magenta]relay diff[/] — epoch {from_epoch} → {to_epoch}")
    console.print()

    s = result.summary
    console.print(
        f"  [green]+{s.added_count} added[/]  "
        f"[red]-{s.removed_count} removed[/]  "
        f"[yellow]~{s.changed_count} changed[/]  "
        f"[cyan]⟳{s.superseded_count} superseded[/]"
    )
    console.print(f"  [dim]semantic drift: {result.semantic_drift.value}[/]")
    console.print()

    if result.added:
        console.print("[bold green]Added:[/]")
        for d in result.added:
            console.print(f"  + {d.doc_id} ({d.source_file or '-'})")

    if result.removed:
        console.print("[bold red]Removed:[/]")
        for d in result.removed:
            console.print(f"  - {d.doc_id} ({d.source_file or '-'})")

    if result.changed:
        console.print("[bold yellow]Changed:[/]")
        for c in result.changed:
            sim = c.cosine_similarity if c.cosine_similarity is not None else "N/A"
            console.print(f"  ~ {c.doc_id} (cos_sim={sim})")

    if result.superseded:
        console.print("[bold cyan]Superseded:[/]")
        for sup in result.superseded:
            console.print(f"  ⟳ {sup.doc_id} supersedes {', '.join(sup.supersedes)}")

    console.print()


@app.command()
def verify(
    request_id: str = typer.Option(
        ..., "--request-id", help="Retrieval request ID to verify"
    ),
    tenant: str = typer.Option(
        CONFIG.default_tenant, "--tenant", "-t", help="Tenant ID"
    ),
    output_json: bool = typer.Option(False, "--json", help="Output raw JSON"),
):
    """Verify a past retrieval against Merkle commitment."""
    from relay.verify import verify_retrieval

    with console.status("[bold]Verifying retrieval integrity..."):
        result = verify_retrieval(request_id=request_id, tenant_id=tenant)

    if output_json:
        console.print_json(result.model_dump_json(indent=2))
        return

    if result.status.value == "VERIFIED":
        panel = Panel(
            f"[bold green]✓ VERIFIED[/]\n\n"
            f"  [cyan]epoch[/]:           {result.epoch_id}\n"
            f"  [cyan]merkle_root[/]:     {(result.stored_merkle_root or '')[:16]}...\n"
            f"  [cyan]root_match[/]:      {result.root_match}\n"
            f"  [cyan]docs_verified[/]:   {result.retrieved_doc_count}/{result.epoch_doc_count}\n"
            f"  [cyan]tenant_match[/]:    {result.tenant_match}\n"
            f'  [cyan]query[/]:           "{result.query_preview[:50]}"',
            title="[bold]relay verify[/]",
            border_style="green",
        )
    else:
        reasons = []
        if result.root_match is False:
            reasons.append("Merkle root mismatch")
        if result.missing_docs:
            reasons.append(f"Missing docs: {result.missing_docs}")
        if result.tenant_match is False:
            reasons.append("Tenant mismatch")

        panel = Panel(
            f"[bold red]✗ FAILED[/]\n\n"
            f"  [red]Reasons[/]: {'; '.join(reasons) if reasons else (result.reason or 'unknown')}\n"
            f"  [cyan]stored_root[/]:   {(result.stored_merkle_root or 'N/A')[:16]}...\n"
            f"  [cyan]computed_root[/]: {(result.computed_merkle_root or 'N/A')[:16]}...",
            title="[bold]relay verify[/]",
            border_style="red",
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
