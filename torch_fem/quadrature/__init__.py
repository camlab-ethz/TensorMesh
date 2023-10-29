from .line import gauss_points as gauss_points_line
from .quad import gauss_points as gauss_points_quad
from .tetra import gauss_points as gauss_points_tetra
from .tri import gauss_points as gauss_points_tri


def get_quadrature(element_type, n):
    if element_type.startswith("line"):
        return gauss_points_line(n)
    elif element_type.startswith("tri"):
        return gauss_points_tri(n)
    elif element_type.startswith("quad"):
        return gauss_points_quad(n)
    elif element_type.startswith("tetra"):
        return gauss_points_tetra(n)
    else:
        raise ValueError(f"Unknown element type: {element_type}")