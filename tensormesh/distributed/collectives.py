"""Distributed-safe utilities for shared random initialisation.

When TensorMesh runs across multiple processes (Gloo CPU or NCCL CUDA),
any value that influences the global problem state -- dataset
coefficients, sampled hyperparameters, initial guesses -- must be
identical on every rank. The natural failure mode is silent: each rank
calls the same constructor, samples from its own RNG, picks different
values, and the resulting distributed solve converges to a different
answer per rank with no error raised.

This module provides one helper:

* :func:`broadcast_from_rank0` -- call a factory on rank 0, broadcast
  the result to every other rank, return the shared tensor.

Library code that samples from RNG at construction time should route
through this helper so users do not need to litter their PDE setup
with manual ``torch.manual_seed`` / ``dist.broadcast`` plumbing. The
single-process path is a no-op (factory is called normally).

Usage::

    from tensormesh.distributed import broadcast_from_rank0

    class MyDataset:
        def __init__(self, K=8):
            self.a = broadcast_from_rank0(
                lambda: torch.empty((K, K)).uniform_(-1, 1)
            )
"""
from __future__ import annotations

from typing import Callable, Optional

import torch


def broadcast_from_rank0(
    factory: Callable[[], torch.Tensor],
    *,
    dst_device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Sample on rank 0, broadcast to every rank.

    Parameters
    ----------
    factory
        Zero-arg callable returning a fresh ``torch.Tensor``. Called
        only on rank 0 when a process group is active; called on
        every rank in single-process / no-process-group mode.
    dst_device
        Optional override for the device the broadcast lands on.
        Defaults to CUDA when NCCL is the active backend, CPU
        otherwise; the returned tensor is moved back to CPU so it
        composes with caller code that does its own ``.to(device)``
        later. Pass an explicit device if you want a different
        placement.

    Returns
    -------
    torch.Tensor
        A tensor with the same content on every rank. Single-process
        mode returns ``factory()`` directly.

    Notes
    -----
    Uses :func:`torch.distributed.broadcast_object_list` under the
    hood, which pickles the tensor. This is fine for the typical
    use case (small coefficient matrices, scalar parameters); for
    large buffers prefer manual ``dist.broadcast`` with pre-allocated
    receive buffers on every rank.
    """
    try:
        import torch.distributed as dist
    except ImportError:
        return factory()

    if not (dist.is_available() and dist.is_initialized()):
        return factory()

    world = dist.get_world_size()
    if world == 1:
        return factory()

    rank = dist.get_rank()
    if rank == 0:
        t = factory()
        payload = [t.detach().cpu()]
    else:
        payload = [None]

    dist.broadcast_object_list(payload, src=0)
    out: torch.Tensor = payload[0]
    if dst_device is not None:
        out = out.to(dst_device)
    return out
