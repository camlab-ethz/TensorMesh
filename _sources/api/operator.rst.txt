tensormesh.operator
===================

.. py:module:: tensormesh.operator

Condenser
---------

.. autoclass:: tensormesh.Condenser
    :members:
    :show-inheritance:
    :exclude-members: dirichlet_mask, dirichlet_value,
        inner_row, inner_col, ou2in_row, ou2in_col,
        is_inner_edge, is_ou2in_edge, is_inner_dof, is_outer_dof,
        inner_shape, ou2in_shape, n_inner_dof, n_outer_dof, n_dof,
        layout_hash, K_ou2in

BlochReducer
------------

.. autoclass:: tensormesh.BlochReducer
    :members:
    :show-inheritance:
    :exclude-members: master_dof, node_R

Wave boundary operators
-----------------------

Assembled boundary matrix / load of first-order absorbing and
plane-wave-port conditions — see the :doc:`open-domain wave examples
</example_gallery/open_domain_wave>`.

.. autofunction:: tensormesh.robin_operator

.. autofunction:: tensormesh.port_source
