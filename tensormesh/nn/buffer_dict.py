"""Module-aware dict/list containers for non-trainable tensors (buffers).

PyTorch ships :class:`torch.nn.ParameterDict` / :class:`torch.nn.ParameterList`
(trainable parameters) and :class:`torch.nn.ModuleDict` / :class:`torch.nn.ModuleList`
(submodules), but it does **not** ship a container for *buffers* — i.e.
non-trainable tensors that still need to follow the parent module under
:meth:`~torch.nn.Module.to`, appear in :meth:`~torch.nn.Module.state_dict`,
and be checkpointed. :class:`BufferDict` (this module) and :class:`BufferList`
(:mod:`tensormesh.nn.buffer_list`) fill that gap.

In TensorMesh, every container of integer connectivity, point/field data, or
precomputed quadrature/shape tables is a :class:`BufferDict` keyed by element
type string — see e.g. :attr:`tensormesh.Mesh.cells`,
:attr:`tensormesh.Mesh.point_data`, and the per-element-type buffers on each
:class:`~tensormesh.ElementAssembler`.
"""
import re
import torch
import torch.nn as nn
from itertools import chain
from collections import OrderedDict
from typing import Optional, Dict, Iterable, Tuple, Mapping


class BufferDict(nn.Module):
    r"""Module-aware dict of tensors stored as buffers (non-trainable).

    Use it whenever you need a dict of plain tensors attached to a
    :class:`~torch.nn.Module` — for example integer connectivity
    ``[n_element, n_basis]``, vector point data ``[n_point, D]``, or
    precomputed quadrature tables — keyed by element type or field name.
    Tensors registered through :class:`BufferDict` follow the parent module
    under ``.to(device)`` / ``.float()`` / ``.cuda()``, appear in
    :meth:`~torch.nn.Module.state_dict`, and do not require gradients.

    Two behaviours go beyond a plain :meth:`~torch.nn.Module.register_buffer`:

    1. **Keys that aren't valid Python identifiers** (anything not matching
       ``^[a-zA-Z_][a-zA-Z0-9_]*$``, e.g. ``"123x"`` or names with dashes)
       are stored in an internal :class:`~collections.OrderedDict`
       (:attr:`_data`) instead of being registered as buffers — Python's
       ``register_buffer`` rejects such names. Their tensors are still moved
       by :meth:`_apply`, so ``.to(device)`` and friends still work; they
       just don't appear in :meth:`~torch.nn.Module.state_dict`.
    2. **Buffer ↔ parameter promotion**: :meth:`as_parameter` turns a stored
       buffer into a trainable :class:`~torch.nn.Parameter` in place, and
       :meth:`as_buffer` reverses it. This lets the same container serve
       both pure-FEM workflows (everything as buffers) and ML workflows
       where some fields need gradients (e.g. learnable material parameters).

    Parameters
    ----------
    data : Dict[str, torch.Tensor], optional
        Initial key→tensor mapping. Keys matching
        ``^[a-zA-Z_][a-zA-Z0-9_]*$`` are registered as buffers via
        :meth:`~torch.nn.Module.register_buffer`; the rest are kept in the
        fallback :attr:`_data` dict. Default: empty.

    Examples
    --------
    >>> import torch
    >>> from tensormesh.nn import BufferDict
    >>> cells = BufferDict({
    ...     "triangle": torch.zeros(10, 3, dtype=torch.long),
    ...     "quad":     torch.zeros(5,  4, dtype=torch.long),
    ... })
    >>> cells.to("cuda")              # both tensors move to GPU
    >>> list(cells.keys())
    ['triangle', 'quad']
    >>> cells["triangle"].device.type
    'cuda'
    """
    def __init__(self, data:Optional[Dict[str, torch.Tensor]] = None):
        super().__init__()
        if data is None:
            data= {}
        self._data:Dict[str,torch.Tensor] = OrderedDict() # used for storing data that cannot be used as a valid name
        pattern = re.compile("^[a-zA-Z_][a-zA-Z0-9_]*$")
      
        for key in list(data.keys()):
            if not pattern.match(key):
                self._data[key] = data.pop(key)
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                self.register_buffer(key, value)
            else:
                raise TypeError(f"Cannot register a {type(value)} as a buffer or a parameter")
    
    def as_parameter(self, key:str):
        """Promote the buffer at ``key`` to a trainable :class:`torch.nn.Parameter` in place.

        After this call, ``self[key]`` is a Parameter (gradient-tracking, will
        appear in :meth:`~torch.nn.Module.parameters`); the same key must
        currently live in :attr:`_buffers` or this will raise ``KeyError``.
        Reverse with :meth:`as_buffer`.
        """
        buffer = self._buffers.pop(key)
        self.register_parameter(key, nn.Parameter(buffer))

    def as_buffer(self, key:str):
        """Demote the parameter at ``key`` back to a (non-trainable) buffer in place.

        Inverse of :meth:`as_parameter`. The same key must currently live in
        :attr:`_parameters`. The underlying storage is shared (via
        :meth:`~torch.Tensor.detach`); the result no longer requires grad.
        """
        parameter = self._parameters.pop(key)
        self.register_buffer(key, parameter.detach())

    def keys(self)->Iterable[str]:
        """Iterate over keys across all three backing stores (buffers, parameters, fallback)."""
        return chain(self._buffers.keys(), self._parameters.keys(), self._data.keys())

    def items(self)->Iterable[Tuple[str, torch.Tensor]]:
        """Iterate over ``(key, tensor)`` pairs across all three backing stores."""
        return chain(self._buffers.items(), self._parameters.items(), self._data.items()) # type: ignore

    def values(self)->Iterable[torch.Tensor]:
        """Iterate over tensors across all three backing stores."""
        return chain(self._buffers.values(), self._parameters.values(), self._data.values()) # type: ignore
    
    def __hash__(self):
        return hash((super().__hash__(), hash(tuple(self._data.keys())), hash(tuple(self._data.values()))))

    def __getitem__(self, key:str)->torch.Tensor:
        if key not in self.keys():
            raise KeyError(f"{key} is not found in the BufferDict")
        
        assert key in self.keys()

        if key in self._buffers.keys():
            return self._buffers[key] # type: ignore
        elif key in self._parameters.keys():
            return self._parameters[key] # type: ignore
        else:
            return self._data[key]
        
    def __setitem__(self, key:str, value:torch.Tensor):
        pattern = re.compile("^[a-zA-Z_][a-zA-Z0-9_]*$")
        if not pattern.match(key):
            self._data[key] = value
        else:
            if isinstance(value, torch.Tensor):
                self.register_buffer(key, value)
            else:
                raise TypeError(f"Cannot register a {type(value)} as a buffer or a parameter")

    def __len__(self):
        return len(self._buffers) + len(self._parameters) + len(self._data)

    def __contains__(self, key) -> bool:
        """``key in bd`` tests the *keys* (like a dict).

        Without this, Python falls back to ``__getitem__(0)``, ``(1)``, …
        and ``"line3" in mesh.cells`` raised ``KeyError('0 is not found …')``.
        """
        return key in self._buffers or key in self._parameters or key in self._data

    def __iter__(self):
        """Iterate over the keys (like a dict)."""
        return iter(list(self.keys()))

    def is_floating_point(self)->bool:
        """Return ``True`` if any stored tensor has a floating-point dtype."""
        return any(map(lambda x:x.is_floating_point(), self.values()))

    def is_complex(self)->bool:
        """Return ``True`` if any stored tensor has a complex dtype."""
        return any(map(lambda x:x.is_complex(), self.values()))

    @property
    def dtype(self):
        """:class:`torch.dtype` of the first registered buffer (representative)."""
        return next(iter(self.buffers().values())).dtype # type: ignore

    @property
    def device(self):
        """:class:`torch.device` of the first registered buffer (representative)."""
        return next(iter(self.buffers().values())).device # type: ignore

    def _apply(self, fn):
        """Override of :meth:`torch.nn.Module._apply` so that fallback-stored
        tensors in :attr:`_data` also follow ``.to(device)`` / ``.float()`` /
        ``.cuda()`` — without this, only the entries in :attr:`_buffers` and
        :attr:`_parameters` would be moved.
        """
        self = super()._apply(fn)
        for key, value in self._data.items():
            self._data[key] = fn(value)
        return self

    def __str__(self):
        return f"""BufferDict(
        {', '.join([f"{key} = {value}" for key, value in self.items()])}
        )"""
      
    def __repr__(self):
        return str(self)
    
    def to_dict(self)->Mapping[str, torch.Tensor|nn.Module]:
        """Return a plain :class:`dict` view of the contents (no module wiring)."""
        return {key:value for key, value in self.items()}

    def clone(self)->'BufferDict':
        """Return a deep copy: every stored tensor is cloned, then wrapped in a fresh :class:`BufferDict`."""
        data = {key:value.clone() for key, value in self.items()} # type: ignore
        return BufferDict(data)