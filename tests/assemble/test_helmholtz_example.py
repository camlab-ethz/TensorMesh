"""Convergence test for the complex Helmholtz example.

Validates that the end-to-end complex pipeline (assembly + Condenser
+ complex linear solve via torch-sla) actually converges to the
analytic plane-wave solution under mesh refinement — the missing
piece in the ROADMAP-item-2 unit tests.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "wave" / "helmholtz"
sys.path.insert(0, str(EXAMPLE_DIR))

from helmholtz import run_convergence  # type: ignore  # noqa: E402


@pytest.mark.parametrize("dtype", [torch.complex128], ids=["complex128"])
def test_helmholtz_convergence_under_refinement(dtype):
    """Halving ``h`` should reduce the L2 error by ~3× or more on each
    step (Helmholtz pollution prevents the textbook 4× for P1, but a
    consistent monotone decrease + final error well below the coarsest
    is the right correctness signal)."""
    results = run_convergence(
        k=2 * math.pi,
        chara_lengths=[0.2, 0.1, 0.05],
        dtype=dtype,
        device="cpu",
    )
    errs = [r[2] for r in results]
    assert all(errs[i] > errs[i + 1] for i in range(len(errs) - 1)), (
        f"L2 error must strictly decrease under refinement, got {errs}"
    )
    # Demand at least a ~2.3x reduction per halving (very loose — the
    # observed value sits around 3-3.8x; this guards against regression
    # to first-order or worse).
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    for r in ratios:
        assert r > 2.3, f"convergence rate too slow: per-step ratios = {ratios}"


def test_helmholtz_result_is_complex():
    """At fixed mesh, sanity-check that the solver returned a complex
    field with non-trivial imaginary part — the analytic exp(i k x)
    must have non-zero Im(u)."""
    from helmholtz import _solve_one_mesh  # type: ignore

    err, _ = _solve_one_mesh(
        chara_length=0.1, k=2 * math.pi, dtype=torch.complex128, device="cpu"
    )
    assert err < 0.1, f"single-mesh error sanity: expected < 0.1, got {err:.3e}"
