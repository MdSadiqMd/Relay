"""Merkle tree implementation for epoch commitments.

Each document produces a leaf:
    leaf = SHA256(doc_id || content_hash || embedding_hash || model_version || valid_from || valid_to || supersedes)

The Merkle root over all leaves commits the entire epoch's semantic state.
Leaves are ordered by topological sort of the supersession DAG so the root
encodes lineage structure, not arbitrary alphabetical order.
"""

import hashlib
from graphlib import CycleError, TopologicalSorter
from typing import Optional


def compute_leaf(
    doc_id: str,
    content_hash: str,
    embedding_hash: str,
    model_version: str,
    valid_from: str,
    valid_to: Optional[str],
    supersedes: list[str],
    video_embedding_hash: Optional[str] = None,
) -> bytes:
    """Compute a Merkle leaf hash for a single document.

    If ``video_embedding_hash`` is provided, it is included in the leaf payload.
    When absent, the leaf hash remains identical to the pre-video behaviour for
    backward compatibility.
    """
    parts = [
        doc_id,
        content_hash,
        embedding_hash,
        model_version,
        valid_from,
        valid_to or "",
        ",".join(sorted(supersedes or [])),  # sorted for determinism
    ]
    if video_embedding_hash is not None:
        parts.append(video_embedding_hash)
    payload = "||".join(parts).encode("utf-8")
    return hashlib.sha256(payload).digest()


def _hash_pair(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(left + right).digest()


def build_supersession_dag(docs: list) -> dict[str, list[str]]:
    """Build adjacency dict: doc_id → list of parent doc_ids it supersedes.

    Used to topologically sort leaves so the Merkle root encodes lineage.
    """
    return {doc.doc_id: list(doc.supersedes) for doc in docs}


def toposort_docs(docs: list, dag: dict[str, list[str]]) -> list:
    """Return docs in topological order (parents before children).

    Raises ValueError on cycle — prevents corrupt supersession chains.
    """
    ts = TopologicalSorter(dag)
    try:
        order = list(ts.static_order())
    except CycleError as e:
        raise ValueError(f"Cycle in supersession DAG: {e}") from e

    doc_map = {doc.doc_id: doc for doc in docs}
    return [doc_map[did] for did in order if did in doc_map]


def compute_merkle_root(leaves: list[bytes], ordered: bool = False) -> str:
    """Compute the Merkle root from a list of leaf hashes.

    If ordered=True, leaves are already in toposort order and are not re-sorted.
    Otherwise falls back to sorted() for determinism.

    Returns the hex-encoded root hash. Empty leaves → SHA256 of empty string.
    """
    if not leaves:
        return hashlib.sha256(b"").hexdigest()

    current_level = list(leaves) if ordered else sorted(leaves)

    while len(current_level) > 1:
        next_level: list[bytes] = []
        for i in range(0, len(current_level), 2):
            left = current_level[i]
            right = current_level[i + 1] if i + 1 < len(current_level) else left
            next_level.append(_hash_pair(left, right))
        current_level = next_level

    return current_level[0].hex()


def verify_leaf_in_tree(
    leaf: bytes, all_leaves: list[bytes], expected_root: str
) -> bool:
    """Verify that a leaf is part of the tree that produces the expected root."""
    computed_root = compute_merkle_root(all_leaves)
    return computed_root == expected_root and leaf in all_leaves
