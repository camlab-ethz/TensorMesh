# QUAD4 on [0,2]x[0,1]: block 0 (x < 1), block 1 (x > 1), the interior side
# set "interface" on x = 1 seen from block 0, and an EDGE2 lower-dimensional
# block 2 on it.
#   moose-opt -i interface.i --mesh-only interface.e
[Mesh]
  [box]
    type = GeneratedMeshGenerator
    dim = 2
    nx = 4
    ny = 2
    xmax = 2
  []
  [right_block]
    type = SubdomainBoundingBoxGenerator
    input = box
    bottom_left = '1 0 0'
    top_right = '2 1 0'
    block_id = 1
  []
  [interface]
    type = SideSetsBetweenSubdomainsGenerator
    input = right_block
    primary_block = 0
    paired_block = 1
    new_boundary = interface
  []
  [lower]
    type = LowerDBlockFromSidesetGenerator
    input = interface
    sidesets = interface
    new_block_id = 2
    new_block_name = interface_lower
  []
[]
[Problem]
  solve = false
[]
[Executioner]
  type = Steady
[]
