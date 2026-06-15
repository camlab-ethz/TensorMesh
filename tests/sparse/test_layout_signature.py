"""Behaviour of :attr:`SparseMatrix.layout_signature` (sequence identity,
zero-sync, in-place / clone semantics)."""
import torch
import pytest

from tensormesh.sparse import SparseMatrix


def _mat(row, col, vals, n):
    return SparseMatrix(
        torch.as_tensor(vals, dtype=torch.float64),
        torch.as_tensor(row, dtype=torch.long),
        torch.as_tensor(col, dtype=torch.long),
        (n, n),
    )


def test_shared_storage_same_signature():
    row = torch.tensor([0, 1, 2], dtype=torch.long)
    col = torch.tensor([0, 1, 2], dtype=torch.long)
    A = _mat(row, col, [1.0, 2.0, 3.0], 3)
    B = _mat(row, col, [4.0, 5.0, 6.0], 3)  # same row/col tensor refs, diff values
    assert A.layout_signature == B.layout_signature
    assert A.has_same_layout(B)


def test_different_storage_different_signature():
    row1 = torch.tensor([0, 1, 2], dtype=torch.long)
    col1 = torch.tensor([0, 1, 2], dtype=torch.long)
    row2 = row1.clone()
    col2 = col1.clone()
    A = _mat(row1, col1, [1.0, 2.0, 3.0], 3)
    B = _mat(row2, col2, [1.0, 2.0, 3.0], 3)
    # Different data_ptr -> different signature, even with same content
    # (this is intentional; see mixin.py docstring).
    assert A.layout_signature != B.layout_signature
    assert not A.has_same_layout(B)


def test_inplace_modify_invalidates_signature():
    row = torch.tensor([0, 1, 2], dtype=torch.long)
    col = torch.tensor([0, 1, 2], dtype=torch.long)
    A = _mat(row, col, [1.0, 2.0, 3.0], 3)
    sig0 = A.layout_signature
    # In-place mutate row indices (we hold the same tensor object that
    # SparseMatrix stored). PyTorch bumps _version.
    A.row_indices.add_(0)  # no-op value-wise but it's still an in-place op
    sig1 = A.layout_signature
    assert sig0 != sig1, "expected _version bump after in-place op"


def test_signature_is_hashable_and_dict_key():
    A = _mat([0, 1], [0, 1], [1.0, 2.0], 2)
    cache = {A.layout_signature: "cached"}
    assert cache[A.layout_signature] == "cached"


def test_legacy_layout_hash_deprecated_but_works():
    A = _mat([0, 1], [0, 1], [1.0, 2.0], 2)
    with pytest.warns(DeprecationWarning):
        h = A.layout_hash
    assert h == A.layout_signature


def test_has_same_layout_accepts_signature_tuple():
    A = _mat([0, 1], [0, 1], [1.0, 2.0], 2)
    sig = A.layout_signature
    assert A.has_same_layout(sig)
