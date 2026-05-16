"""Merkle tree implementation for epoch commitments.

Each document produces a leaf:
    leaf = SHA256(doc_id || content_hash || embedding_hash || model_version || valid_from || valid_to || supersedes)

The Merkle root over all leaves commits the entire epoch's semantic state.
"""

import hashlib
from typing import Optional


def compute_leaf(
    doc_id: str,
    content_hash: str,
    embedding_hash: str,
    model_version: str,
    valid_from: str,
    valid_to: Optional[str],
    supersedes: Optional[str],
) -> bytes:
    """Compute a Merkle leaf hash for a single document.

    Concatenates all fields with '||' separator then SHA256 hashes the result.
    """
    parts = [
        doc_id,
        content_hash,
        embedding_hash,
        model_version,
        valid_from,
        valid_to or "",
        supersedes or "",
    ]
    payload = "||".join(parts).encode("utf-8")
    return hashlib.sha256(payload).digest()


def _hash_pair(left: bytes, right: bytes) -> bytes:
    """Hash two nodes together."""
    return hashlib.sha256(left + right).digest()


def compute_merkle_root(leaves: list[bytes]) -> str:
    """Compute the Merkle root from a list of leaf hashes.

    Uses a standard binary Merkle tree. If the number of leaves at any
    level is odd, the last leaf is duplicated.

    Returns the hex-encoded root hash. If no leaves are provided,
    returns the SHA256 of an empty string.
    """
    if not leaves:
        return hashlib.sha256(b"").hexdigest()

    # Sort leaves for deterministic ordering
    current_level = sorted(leaves)

    while len(current_level) > 1:
        next_level: list[bytes] = []

        for i in range(0, len(current_level), 2):
            left = current_level[i]
            # Duplicate last leaf if odd count
            right = current_level[i + 1] if i + 1 < len(current_level) else left
            next_level.append(_hash_pair(left, right))

        current_level = next_level

    return current_level[0].hex()


def verify_leaf_in_tree(
    leaf: bytes, all_leaves: list[bytes], expected_root: str
) -> bool:
    """Verify that a leaf is part of the tree that produces the expected root.

    Recomputes the full Merkle root and checks equality.
    """
    computed_root = compute_merkle_root(all_leaves)
    return computed_root == expected_root and leaf in all_leaves
