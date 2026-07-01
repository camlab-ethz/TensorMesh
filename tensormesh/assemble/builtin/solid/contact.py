"""Contact / boundary-facet built-in assembler."""

import inspect
from typing import Optional, Dict, Union, Callable

import torch

from tensormesh.assemble.element_assembler import InputBroadcast
from tensormesh.assemble.facet_assembler import FacetAssembler
from tensormesh.vmap import vmap


class ContactAssembler(FacetAssembler):
    r"""Contact/Boundary Facet Assembler.

    Assembler for integrating energy contributions over boundary facets (surfaces in 3D,
    edges in 2D). This is useful for implementing:

    - **Contact mechanics**: Penalty or barrier methods for non-penetration constraints
    - **Surface tension**: Capillary effects in fluid-structure interaction
    - **Pressure loads**: Follower forces that remain normal to deformed surface
    - **Robin boundary conditions**: Mixed Dirichlet-Neumann conditions

    **Penalty Contact Formulation:**

    For a penalty-based contact between surface :math:`\Gamma` and an obstacle:

    .. math::

        \Pi_{contact} = \int_{\Gamma} \frac{\kappa}{2} \langle g_n \rangle_-^2 \, \mathrm{d}S

    where:

    - :math:`g_n` is the normal gap function (negative when penetrating)
    - :math:`\langle \cdot \rangle_- = \min(\cdot, 0)` is the negative part (Macaulay bracket)
    - :math:`\kappa` is the penalty stiffness

    **Usage Pattern:**

    Subclass ``ContactAssembler`` and implement ``element_energy`` to define
    the specific contact/boundary energy density.

    Examples
    --------
    .. code-block:: python

        class PenaltyContact(ContactAssembler):
            def __post_init__(self, kappa=1e6, obstacle_y=0.0):
                self.kappa = kappa
                self.obstacle_y = obstacle_y

            def element_energy(self, x):
                gap = x[..., 1] - self.obstacle_y         # y-coordinate gap
                penetration = torch.clamp(-gap, min=0.0)
                return 0.5 * self.kappa * penetration ** 2
    """
    def energy(self, points:Optional[torch.Tensor] = None,
                       func:Optional[Callable] = None,
                       point_data:Optional[Dict[str, torch.Tensor]] = None,
                       element_data:Optional[Union[Dict[str, Dict[str,torch.Tensor]], Dict[str,torch.Tensor]]] = None,
                       scalar_data:Optional[Dict[str, torch.Tensor]] = None,
                       batch_size:int = -1):
        r"""Compute total boundary / contact energy.

        Integrates ``element_energy`` over all selected boundary facets:

        .. math::

            \Pi = \int_{\Gamma} \psi(\mathbf{x}, \mathbf{u}, \ldots) \, \mathrm{d}S

        Parameters
        ----------
        points : torch.Tensor, optional
            Updated nodal coordinates; if ``None``, the cached points are used.
        func : Callable, optional
            Custom energy density to use *in place of* :meth:`element_energy`.
        point_data : dict[str, torch.Tensor], optional
            Nodal fields to interpolate at quadrature points.
        element_data : dict, optional
            Element-wise data (constant or per-quadrature).
        scalar_data : dict, optional
            Global scalar parameters.
        batch_size : int, optional
            Quadrature-point batch size; ``-1`` (default) means no batching.

        Returns
        -------
        torch.Tensor
            Scalar total boundary energy.
        """
        if point_data is None: point_data = {}
        if element_data is None: element_data = {element_type:{} for element_type in self.element_types}
        if scalar_data is None: scalar_data = {}

        for key, value in point_data.items():
            assert value.shape[0] == self.n_points

        if points is not None:
            self = self.type(points.dtype).to(points.device)
            for element_type in self.element_types:
                self.transformation[element_type].update_points(points)
        else:
            points = next(iter(self.transformation.values())).points

        point_data["x"] = points

        fn = self.element_energy if func is None else func
        signature = inspect.signature(fn)

        broadcast_fns = [
            (lambda x: x in element_data.keys(), InputBroadcast(True, False, False, False)),
            (lambda x: x in scalar_data.keys(), InputBroadcast(False, False, False, False)),
            (lambda x: x in point_data.keys(), InputBroadcast(True, True, False, False)),
        ]

        element_dims = []
        quadrature_dims = []

        for key in signature.parameters:
            is_match = False
            for condition, broadcast in broadcast_fns:
                if condition(key):
                    element_dims.append(broadcast.element)
                    quadrature_dims.append(broadcast.quadrature)
                    is_match = True
                    break
            if not is_match:
                 raise ValueError(f"{key} is not supported for contact energy calculation.")

        element_dims = tuple(element_dims)
        quadrature_dims = tuple(quadrature_dims)

        parallel_fn = vmap(
            vmap(fn, in_dims=quadrature_dims),
            in_dims=element_dims
        )

        total_energy = 0.0

        for element_type in self.element_types:
            trans = self.transformation[element_type]

            if trans.element.is_mix_facet:
                 raise NotImplementedError("Mixed facet elements not fully supported in simple energy loop yet.")
            else:
                m = self.facet_mask[element_type].item()
                elem_indices, facet_indices = torch.where(m)

                if len(elem_indices) == 0:
                    continue

                shape_val_sel = trans.facet_shape_val[facet_indices]
                FxW = trans.FxW[m]

                args = []
                for key in signature.parameters:
                    if key in point_data:
                        u_global = point_data[key]
                        nodes = trans.elements[elem_indices]
                        u_nodes = u_global[nodes]
                        val_interp = torch.einsum("sbd,sqb->sqd", u_nodes, shape_val_sel)
                        args.append(val_interp)
                    elif key in scalar_data:
                        args.append(scalar_data[key])
                    elif key in element_data:
                        args.append(element_data[key][element_type][elem_indices])
                    else:
                        raise ValueError(f"Unknown arg {key}")

                energy_density = parallel_fn(*args)
                energy_val = (energy_density * FxW).sum()
                total_energy += energy_val

        return total_energy

    def element_energy(self, **kwargs):
        """Override this method to define the boundary energy density."""
        raise NotImplementedError
