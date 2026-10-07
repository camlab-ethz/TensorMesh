# Blocks 1 (HEX8), 2 (WEDGE6) and 3 (HEX8) side by side on [0,3]x[0,1]x[0,1],
# not stitched, so the element numbering interleaves the two element types.
#   moose-opt -i mixed.i --mesh-only mixed.e
[Mesh]
  [a]
    type = GeneratedMeshGenerator
    dim = 3
    elem_type = HEX8
    subdomain_ids = 1
  []
  [b]
    type = GeneratedMeshGenerator
    dim = 3
    elem_type = PRISM6
    xmin = 1
    xmax = 2
    subdomain_ids = 2
  []
  [c]
    type = GeneratedMeshGenerator
    dim = 3
    elem_type = HEX8
    xmin = 2
    xmax = 3
    subdomain_ids = 3
  []
  [combine]
    type = CombinerGenerator
    inputs = 'a b c'
  []
[]
[Problem]
  solve = false
[]
[Executioner]
  type = Steady
[]
