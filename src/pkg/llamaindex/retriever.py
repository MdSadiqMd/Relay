"""RelayRetriever — LlamaIndex retriever backed by relay's temporal semantic memory."""

from typing import Callable, Optional

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle

from relay.config import CONFIG
from relay.query import query_nodes as relay_query_nodes


class RelayRetriever(BaseRetriever):
    """LlamaIndex retriever backed by relay's temporal semantic memory.

    Args:
        tenant_id: Tenant namespace for isolation.
        at: ISO timestamp for time-travel queries (e.g. ``"2025-01-01"``).
        epoch_id: Pin to a specific epoch (overrides ``at``).
        top_k: Number of results to return.
        retrieval_policy: ``"dense"`` (default) or ``"hybrid"``.
        text_resolver: Optional callback ``(doc_id, source_file) -> str`` to
            populate ``TextNode.text`` from your document store.  Relay stores
            only metadata + vectors in Qdrant; the resolver bridges to your
            actual document content.  If omitted, node text is empty.
    """

    def __init__(
        self,
        tenant_id: str = CONFIG.default_tenant,
        at: Optional[str] = None,
        epoch_id: Optional[int] = None,
        top_k: int = CONFIG.default_top_k,
        retrieval_policy: str = CONFIG.default_retrieval_policy,
        text_resolver: Optional[Callable[[str, Optional[str]], str]] = None,
    ) -> None:
        super().__init__()
        self._tenant_id = tenant_id
        self._at = at
        self._epoch_id = epoch_id
        self._top_k = top_k
        self._retrieval_policy = retrieval_policy
        self._text_resolver = text_resolver

    @classmethod
    def class_name(cls) -> str:
        return "RelayRetriever"

    def retrieve_with_epoch(self, query_str: str) -> tuple[list[NodeWithScore], int]:
        """Like :meth:`retrieve` but also returns the epoch ID used."""
        return self._query(query_str)

    def _query(self, query_str: str) -> tuple[list[NodeWithScore], int]:
        return relay_query_nodes(
            text=query_str,
            tenant_id=self._tenant_id,
            at=self._at,
            epoch_id=self._epoch_id,
            retrieval_policy=self._retrieval_policy,
            top_k=self._top_k,
            text_resolver=self._text_resolver,
        )

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        nodes, _ = self._query(query_bundle.query_str)
        return nodes
