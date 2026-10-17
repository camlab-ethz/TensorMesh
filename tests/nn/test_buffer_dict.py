"""BufferDict dict-like protocol: membership, iteration, lookup."""
import pytest
import torch

from tensormesh.nn import BufferDict


@pytest.fixture
def bd():
    return BufferDict({
        "triangle": torch.zeros(3, 3, dtype=torch.long),
        "line3": torch.zeros(2, 3, dtype=torch.long),
        "2-bad name": torch.ones(2),                  # not an identifier -> fallback store
    })


def test_contains_tests_keys(bd):
    # used to raise KeyError('0 is not found …') via positional __getitem__
    assert "line3" in bd
    assert "triangle" in bd
    assert "2-bad name" in bd
    assert "quad" not in bd
    assert 0 not in bd


def test_iteration_yields_keys(bd):
    assert list(bd) == ["triangle", "line3", "2-bad name"]
    assert dict(bd.items()).keys() == set(bd)
    assert len(bd) == 3


def test_missing_key_raises_keyerror(bd):
    with pytest.raises(KeyError):
        bd["quad"]


def test_contains_follows_parameter_promotion():
    bd = BufferDict({"weights": torch.ones(3)})      # floating point: may require grad
    bd.as_parameter("weights")
    assert "weights" in bd
    assert isinstance(bd["weights"], torch.nn.Parameter)
    bd.as_buffer("weights")
    assert "weights" in bd
    assert not bd["weights"].requires_grad
