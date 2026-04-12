"""
Per-element node ordering visualization (TensorMesh/FEniCS vs Gmsh/VTK).

User-facing goal:
- Avoid hand-written node coordinates in examples.
- For each element, generate one image with two panels:
  - Left: TensorMesh internal basis-node order (often matches FEniCS style)
  - Right: Gmsh/VTK basis-node order

Implementation notes:
- Node coordinates are taken from `Element.get_basis(order)`, so they are consistent with
  TensorMesh shape functions / basis definitions.
- The "Gmsh/VTK order" panel is obtained by permuting those basis points using
  `Element.get_gmsh_permutation(n_nodes)` (which maps gmsh->internal).

Outputs are saved next to this script, e.g.:
  - triangle_p2_order_compare.png
  - quad_p3_order_compare.png
  - tet_p2_order_compare.png
"""

import os
import numpy as np
import torch
import pyvista as pv

from tensormesh.element import Triangle, Quadrilateral, Tetrahedron, Hexahedron, Pyramid, Prism
from tensormesh.visualization import setup_headless


def _to_xyz(points: torch.Tensor) -> np.ndarray:
    pts = points.detach().cpu().numpy()
    if pts.shape[1] == 2:
        pts = np.pad(pts, ((0, 0), (0, 1)))
    return pts.astype(float)


def _polydata_polygon(points_xyz: np.ndarray, conn: np.ndarray) -> pv.PolyData:
    conn = np.asarray(conn, dtype=np.int64).reshape(-1)
    faces = np.concatenate([[len(conn)], conn]).astype(np.int64)
    return pv.PolyData(points_xyz, faces)


def _label_font_size(n_points: int) -> int:
    if n_points <= 10:
        return 14
    if n_points <= 30:
        return 10
    return 8


def _add_labels(plotter: pv.Plotter, dataset):
    pts = dataset.points
    labels = [str(i) for i in range(pts.shape[0])]
    plotter.add_point_labels(
        pts,
        labels,
        font_size=_label_font_size(pts.shape[0]),
        text_color="white",
        point_color="red",
        shape_color="black",
        shape_opacity=0.5,
        always_visible=True,
    )


def _gmsh_points_from_internal(element_cls, internal_points: torch.Tensor) -> torch.Tensor:
    """
    Return basis points re-indexed into Gmsh/VTK node order.

    `perm_gmsh_to_internal[i_gmsh] = i_internal`, so:
      points_gmsh[i_gmsh] = points_internal[perm_gmsh_to_internal[i_gmsh]]
    """
    n_nodes = int(internal_points.shape[0])
    perm = element_cls.get_gmsh_permutation(n_nodes, device=internal_points.device)
    return internal_points.index_select(0, perm)


def _gmsh_index_from_internal(element_cls, n_nodes: int, device: torch.device) -> torch.Tensor:
    perm_gmsh_to_internal = element_cls.get_gmsh_permutation(n_nodes, device=device)
    inv = torch.argsort(perm_gmsh_to_internal)
    return inv


def _faces_from_facets(facet_list) -> np.ndarray:
    """Build PyVista faces array from a list of facet tensors (possibly different sizes).

    For quads (4-node faces), the facet tensor stores nodes as [a, b, c, d] where
    a-b are one edge pair and c-d the corresponding pair. PyVista needs polygon
    winding order, so we reorder to [a, b, d, c] (swap last two).
    """
    parts = []
    for f in facet_list:
        n_nodes_per_face = f.shape[1]
        for i in range(f.shape[0]):
            idx = f[i].detach().cpu().numpy().astype(np.int64).reshape(-1)
            if n_nodes_per_face == 4:
                idx = idx[[0, 1, 3, 2]]  # fix winding order
            parts.append(np.concatenate([[idx.size], idx]))
    return np.concatenate(parts).astype(np.int64)


def _render_2d_subplot(plotter: pv.Plotter, row: int, col: int, element_cls, order: int, gmsh: bool):
    basis_internal = element_cls.get_basis(order=order, dtype=torch.float64)
    contour_internal = element_cls.get_contour(order=order).to(torch.long)
    n_nodes = int(basis_internal.shape[0])

    title = f"p{order}"
    if not gmsh:
        pts = _to_xyz(basis_internal)
        conn = contour_internal.detach().cpu().numpy()
        ds = _polydata_polygon(pts, conn)
        plotter.subplot(row, col)
        plotter.add_text(f"{title}\nTensorMesh/FEniCS", font_size=10)
        # Avoid confusing internal triangulation lines: VTK triangulates polygons for rendering.
        plotter.add_mesh(ds, color="lightgray", show_edges=False, lighting=False)
        # Draw only the outer boundary as a polyline
        loop = np.concatenate([conn, conn[:1]]).astype(np.int64)
        lines = np.concatenate([[loop.size], loop]).astype(np.int64)
        boundary = pv.PolyData(pts)
        boundary.lines = lines
        plotter.add_mesh(boundary, color="black", line_width=2)
        _add_labels(plotter, ds)
        plotter.view_xy()
        plotter.reset_camera()
        plotter.camera.zoom(0.8)  # zoom out for whitespace
        return

    perm = element_cls.get_gmsh_permutation(n_nodes, device=basis_internal.device)
    identity = torch.equal(perm.cpu(), torch.arange(n_nodes, dtype=torch.long))
    inv_internal_to_gmsh = torch.argsort(perm)
    basis_gmsh = basis_internal.index_select(0, perm)
    contour_gmsh = inv_internal_to_gmsh.index_select(0, contour_internal)

    pts = _to_xyz(basis_gmsh)
    conn = contour_gmsh.detach().cpu().numpy()
    ds = _polydata_polygon(pts, conn)
    plotter.subplot(row, col)
    plotter.add_text(f"{title}\nGmsh/VTK", font_size=10)
    if identity and n_nodes not in (3, 4):  # avoid confusing "identity" notes for low-order simplex
        plotter.add_text("perm=identity (not implemented)", font_size=8, position="lower_left")
    plotter.add_mesh(ds, color="lightgray", show_edges=False, lighting=False)
    loop = np.concatenate([conn, conn[:1]]).astype(np.int64)
    lines = np.concatenate([[loop.size], loop]).astype(np.int64)
    boundary = pv.PolyData(pts)
    boundary.lines = lines
    plotter.add_mesh(boundary, color="black", line_width=2)
    _add_labels(plotter, ds)
    plotter.view_xy()
    plotter.reset_camera()
    plotter.camera.zoom(0.8)


def _render_3d_compare(plotter: pv.Plotter, row: int, col: int, element_cls, order: int, gmsh: bool):
    basis_internal = element_cls.get_basis(order=order, dtype=torch.float64)
    n_nodes = int(basis_internal.shape[0])

    # Use order=1 (linear) facets for the surface geometry so that PyVista gets
    # simple triangles/quads instead of high-order polygons with interior nodes.
    linear_facets = element_cls.get_facet(order=1)
    if isinstance(linear_facets, tuple):
        linear_facets = [f.to(torch.long) for f in linear_facets]
    else:
        linear_facets = [linear_facets.to(torch.long)]

    title = f"p{order}"
    if not gmsh:
        pts = _to_xyz(basis_internal)
        faces = _faces_from_facets(linear_facets)
        surf = pv.PolyData(pts, faces)
        plotter.subplot(row, col)
        plotter.add_text(f"{title}\nTensorMesh/FEniCS", font_size=10)
        plotter.add_mesh(surf, color="lightgray", show_edges=True, opacity=0.95)
        _add_labels(plotter, pv.PolyData(pts))
        plotter.view_isometric()
        plotter.reset_camera()
        plotter.camera.zoom(0.8)
        return

    perm = element_cls.get_gmsh_permutation(n_nodes, device=basis_internal.device)
    identity = torch.equal(perm.cpu(), torch.arange(n_nodes, dtype=torch.long))
    inv_internal_to_gmsh = torch.argsort(perm)
    basis_gmsh = basis_internal.index_select(0, perm)
    linear_facets_gmsh = [inv_internal_to_gmsh[f] for f in linear_facets]

    pts = _to_xyz(basis_gmsh)
    faces = _faces_from_facets(linear_facets_gmsh)
    surf = pv.PolyData(pts, faces)
    plotter.subplot(row, col)
    plotter.add_text(f"{title}\nGmsh/VTK", font_size=10)
    if identity and n_nodes not in (4, 8, 5, 6):
        plotter.add_text("perm=identity (not implemented)", font_size=8, position="lower_left")
    plotter.add_mesh(surf, color="lightgray", show_edges=True, opacity=0.95)
    _add_labels(plotter, pv.PolyData(pts))
    plotter.view_isometric()
    plotter.reset_camera()
    plotter.camera.zoom(0.8)


def main():
    setup_headless()

    out_dir = os.path.dirname(os.path.abspath(__file__))

    orders = [2, 3, 4]

    # 2D elements: one figure per element, 2x3 (rows: internal vs gmsh; cols: p2,p3,p4)
    for element_cls, name in [
        (Triangle, "triangle"),
        (Quadrilateral, "quad"),
    ]:
        p = pv.Plotter(off_screen=True, shape=(2, 3), window_size=(2100, 1400))
        p.set_background("white")
        for j, order in enumerate(orders):
            _render_2d_subplot(p, 0, j, element_cls, order, gmsh=False)
            _render_2d_subplot(p, 1, j, element_cls, order, gmsh=True)
        out = os.path.join(out_dir, f"{name}_p2p3p4_order_compare.png")
        p.screenshot(out)
        p.close()
        print(f"Saved: {out}")

    # 3D elements: one figure per element, 2x3
    for element_cls, name in [
        (Tetrahedron, "tet"),
        (Hexahedron, "hex"),
        (Pyramid, "pyr"),
        (Prism, "pri"),
    ]:
        p = pv.Plotter(off_screen=True, shape=(2, 3), window_size=(2400, 1500))
        p.set_background("white")
        for j, order in enumerate(orders):
            try:
                _render_3d_compare(p, 0, j, element_cls, order, gmsh=False)
            except Exception as e:
                p.subplot(0, j)
                p.add_text(f"p{order}\nTensorMesh/FEniCS\nERROR: {type(e).__name__}", font_size=10)
            try:
                _render_3d_compare(p, 1, j, element_cls, order, gmsh=True)
            except Exception as e:
                p.subplot(1, j)
                p.add_text(f"p{order}\nGmsh/VTK\nERROR: {type(e).__name__}", font_size=10)
        out = os.path.join(out_dir, f"{name}_p2p3p4_order_compare.png")
        p.screenshot(out)
        p.close()
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()


