"""LlamaIndex RAG layer — built-in retrieval-augmented generation for relay.

Relay core provides the retrieval engine (``relay.query``).  The
``pkg.llamaindex`` subpackage adds LlamaIndex-powered retrieval and synthesis
on top — turning raw temporally-versioned retrieval into a full RAG pipeline.

Usage::

    from pkg.llamaindex import RelayRetriever, create_query_engine

    def resolve_text(doc_id, source_file):
        ...  # bridge to your document store

    engine = create_query_engine(tenant_id="default", text_resolver=resolve_text)
    response = engine.query("What is event sourcing?")
"""

from pkg.llamaindex.llm import (
    DEFAULT_LLM_MODEL,
    LocalHFLLM,
    create_query_engine,
    get_llm,
)
from pkg.llamaindex.retriever import RelayRetriever

__all__ = [
    "DEFAULT_LLM_MODEL",
    "LocalHFLLM",
    "RelayRetriever",
    "create_query_engine",
    "get_llm",
]
