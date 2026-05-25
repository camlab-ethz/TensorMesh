tensormesh.dataset
==================

.. py:module:: tensormesh.dataset

mesh
----

.. autoclass:: tensormesh.MeshGen
    :members:
    :show-inheritance:


Batch mesh generators
~~~~~~~~~~~~~~~~~~~~~

Module-level helpers that produce a fresh :class:`~tensormesh.Mesh`
each call — used by the dataset-generation scripts under
``examples/dataset/``. The :class:`~tensormesh.Mesh` class also exposes
classmethods with the same names (e.g. :meth:`~tensormesh.Mesh.gen_rectangle`)
for one-off use.

.. autofunction:: tensormesh.dataset.gen_rectangle

.. autofunction:: tensormesh.dataset.gen_hollow_rectangle

.. autofunction:: tensormesh.dataset.gen_circle

.. autofunction:: tensormesh.dataset.gen_hollow_circle

.. autofunction:: tensormesh.dataset.gen_L

.. autofunction:: tensormesh.dataset.gen_cube

.. autofunction:: tensormesh.dataset.gen_hollow_cube

.. autofunction:: tensormesh.dataset.gen_sphere

.. autofunction:: tensormesh.dataset.gen_hollow_sphere


equation
--------

.. autoclass:: tensormesh.dataset.PoissonMultiFrequency
    :members:
    :show-inheritance:

.. autoclass:: tensormesh.dataset.HeatMultiFrequency
    :members:
    :show-inheritance:

.. autoclass:: tensormesh.dataset.WaveMultiFrequency
    :members:
    :show-inheritance:
