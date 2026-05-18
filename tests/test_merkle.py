"""Unit tests for the Merkle tree implementation — pure crypto, no Qdrant."""

import hashlib

from relay.merkle import compute_leaf, compute_merkle_root, verify_leaf_in_tree


class TestComputeLeaf:
    def test_deterministic(self):
        """Same inputs always produce the same leaf."""
        a = compute_leaf("d1", "ch", "eh", "v1", "2024-01-01", None, None)
        b = compute_leaf("d1", "ch", "eh", "v1", "2024-01-01", None, None)
        assert a == b

    def test_different_inputs_differ(self):
        a = compute_leaf("d1", "ch", "eh", "v1", "2024-01-01", None, None)
        b = compute_leaf("d2", "ch", "eh", "v1", "2024-01-01", None, None)
        assert a != b

    def test_valid_to_affects_hash(self):
        a = compute_leaf("d1", "ch", "eh", "v1", "2024-01-01", None, None)
        b = compute_leaf("d1", "ch", "eh", "v1", "2024-01-01", "2025-01-01", None)
        assert a != b

    def test_supersedes_affects_hash(self):
        a = compute_leaf("d1", "ch", "eh", "v1", "2024-01-01", None, None)
        b = compute_leaf("d1", "ch", "eh", "v1", "2024-01-01", None, "old_doc")
        assert a != b

    def test_returns_bytes(self):
        leaf = compute_leaf("d1", "ch", "eh", "v1", "2024-01-01", None, None)
        assert isinstance(leaf, bytes)
        assert len(leaf) == 32  # SHA256 digest


class TestComputeMerkleRoot:
    def test_empty_leaves(self):
        """Empty tree should return SHA256 of empty bytes."""
        root = compute_merkle_root([])
        expected = hashlib.sha256(b"").hexdigest()
        assert root == expected

    def test_single_leaf(self):
        leaf = compute_leaf("d1", "ch", "eh", "v1", "2024-01-01", None, None)
        root = compute_merkle_root([leaf])
        assert root == leaf.hex()

    def test_two_leaves(self):
        l1 = compute_leaf("d1", "ch1", "eh1", "v1", "2024-01-01", None, None)
        l2 = compute_leaf("d2", "ch2", "eh2", "v1", "2024-01-01", None, None)
        root = compute_merkle_root([l1, l2])
        # Root should be different from either leaf
        assert root != l1.hex()
        assert root != l2.hex()
        assert len(root) == 64  # hex-encoded SHA256

    def test_order_independent(self):
        """Merkle root is deterministic regardless of input order (leaves are sorted)."""
        l1 = compute_leaf("d1", "ch1", "eh1", "v1", "2024-01-01", None, None)
        l2 = compute_leaf("d2", "ch2", "eh2", "v1", "2024-01-01", None, None)
        root_a = compute_merkle_root([l1, l2])
        root_b = compute_merkle_root([l2, l1])
        assert root_a == root_b

    def test_odd_number_of_leaves(self):
        """Odd leaf count should still produce a valid root (last leaf duplicated)."""
        leaves = [
            compute_leaf(f"d{i}", f"ch{i}", f"eh{i}", "v1", "2024-01-01", None, None)
            for i in range(3)
        ]
        root = compute_merkle_root(leaves)
        assert isinstance(root, str)
        assert len(root) == 64

    def test_many_leaves(self):
        leaves = [
            compute_leaf(f"d{i}", f"ch{i}", f"eh{i}", "v1", "2024-01-01", None, None)
            for i in range(100)
        ]
        root = compute_merkle_root(leaves)
        assert isinstance(root, str)
        assert len(root) == 64


class TestVerifyLeafInTree:
    def test_leaf_present(self):
        l1 = compute_leaf("d1", "ch1", "eh1", "v1", "2024-01-01", None, None)
        l2 = compute_leaf("d2", "ch2", "eh2", "v1", "2024-01-01", None, None)
        all_leaves = [l1, l2]
        root = compute_merkle_root(all_leaves)
        assert verify_leaf_in_tree(l1, all_leaves, root) is True

    def test_leaf_not_present(self):
        l1 = compute_leaf("d1", "ch1", "eh1", "v1", "2024-01-01", None, None)
        l2 = compute_leaf("d2", "ch2", "eh2", "v1", "2024-01-01", None, None)
        l3 = compute_leaf("d3", "ch3", "eh3", "v1", "2024-01-01", None, None)
        root = compute_merkle_root([l1, l2])
        assert verify_leaf_in_tree(l3, [l1, l2], root) is False

    def test_wrong_root(self):
        l1 = compute_leaf("d1", "ch1", "eh1", "v1", "2024-01-01", None, None)
        assert verify_leaf_in_tree(l1, [l1], "deadbeef" * 8) is False
