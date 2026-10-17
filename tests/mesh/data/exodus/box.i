# One element type on [0,2]x[0,1](x[0,1]); side sets and node sets
# left/right/bottom/top(/back/front) from GeneratedMeshGenerator.
#   moose-opt -i box.i --mesh-only <type>.e Mesh/box/dim=<2|3> Mesh/box/elem_type=<TYPE>
[Mesh]
  [box]
    type = GeneratedMeshGenerator
    dim = 3
    nx = 2
    ny = 2
    nz = 2
    xmax = 2
  []
[]
[Problem]
  solve = false
[]
[Executioner]
  type = Steady
[]
