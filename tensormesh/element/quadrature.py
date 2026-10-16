"""Reference-element quadrature rules for line, triangle, quad, tet, hex,
pyramid, prism, and their facets.

Internal helpers wired into :meth:`~tensormesh.Element.get_quadrature` and
:meth:`~tensormesh.Element.get_facet_quadrature`. Not part of the public API.
"""
import torch
from typing import Tuple
from scipy.special import roots_jacobi
from .polynomial import Polynomials
from .types import Tensorx2, Tensorx4, Tensorx5


def lin_quadrature(order: int = 1,
                   dtype: torch.dtype = torch.float32,
                   device: torch.device = torch.device('cpu'),
                   ) -> Tensorx2:
    """Gauss-Legendre quadrature on the unit interval :math:`[0, 1]`.

    Parameters
    ----------
    order : int, optional
        Quadrature order between 1 and 7. Defaults to 1.
    dtype : torch.dtype, optional
        Output dtype.
    device : torch.device, optional
        Output device.

    Returns
    -------
    weights : torch.Tensor
        1D tensor of shape ``[n_quadrature]``.
    points : torch.Tensor
        2D tensor of shape ``[n_quadrature, 1]``.
    """
    assert order >= 1 and order <= 7, f"Invalid order {order} for lin quadrature, expected one of [1,7]"
    weights_table = {
        1 : torch.tensor([2.0], dtype=dtype),
        2 : torch.tensor([1.0, 1.0], dtype=dtype),
        3 : torch.tensor([0.55555555555555555556, 0.88888888888888888889, 0.55555555555555555556], dtype=dtype),
        4 : torch.tensor([0.34785484513745385737, 0.65214515486254614263, 0.65214515486254614263, 0.34785484513745385737], dtype=dtype),
        5 : torch.tensor([0.23692688505618908751, 0.47862867049936646804, 0.56888888888888888889, 0.47862867049936646804, 0.23692688505618908751], dtype=dtype),
        6 : torch.tensor([0.17132449237917034504, 0.36076157304813860757, 0.46791393457269104739, 0.46791393457269104739, 0.36076157304813860757, 0.17132449237917034504], dtype=dtype),
        7 : torch.tensor([0.12948496616886969327, 0.27970539148927666790, 0.38183005050511894495, 0.41795918367346938776, 0.38183005050511894495, 0.27970539148927666790, 0.12948496616886969327], dtype=dtype)
    }

    points_table = {
        1 : torch.tensor([[0.0]], dtype=dtype),
        2 : torch.tensor([[-0.57735026918962576451],[0.57735026918962576451 ]], dtype=dtype),
        3 : torch.tensor([[-0.77459666924148337704],[0],[0.77459666924148337704]], dtype=dtype),
        4 : torch.tensor([[-0.8611363115940526],[-0.3399810435848563],[0.3399810435848563],[0.8611363115940526]], dtype=dtype),
        5 : torch.tensor([[-0.9061798459386640],[-0.5384693101056831],[0],[0.5384693101056831],[0.9061798459386640]], dtype=dtype),
        6 : torch.tensor([[-0.9324695142031521],[-0.6612093864662645],[-0.2386191860831969],[0.2386191860831969],[0.6612093864662645],[0.9324695142031521]], dtype=dtype),
        7 : torch.tensor([[-0.9491079123427585],[-0.7415311855993945],[-0.4058451513773972],[0],[0.4058451513773972],[0.7415311855993945],[0.9491079123427585]], dtype=dtype)
    }
    assert order in weights_table, f"Invalid order {order} for lin quadrature, expected one of {weights_table.keys()}"
    weights = weights_table[order]
    points  = points_table[order]

    weights = weights / 2 
    # points  = (points + 1)
    points = (points + 1) / 2

    weights = weights.to(device)
    points  = points.to(device)

    return weights, points

def tri_quadrature(order: int = 1,
                   dtype: torch.dtype = torch.float32,
                   device: torch.device = torch.device('cpu'),
                   ) -> Tensorx2:
    """Quadrature rule on the reference triangle.

    Parameters
    ----------
    order : int, optional
        Quadrature order between 1 and 7. Defaults to 1.
    dtype : torch.dtype, optional
        Output dtype.
    device : torch.device, optional
        Output device.

    Returns
    -------
    weights : torch.Tensor
        1D tensor of shape ``[n_quadrature]``.
    points : torch.Tensor
        2D tensor of shape ``[n_quadrature, 2]``.
    """
    assert order >= 1 and order <= 7, f"Invalid order {order} for tri quadrature, expected one of [1,7]"
    weights_table = {
        1: torch.tensor([0.5],dtype=dtype),
        2: torch.tensor([ 0.16666666666666666, 0.16666666666666666, 0.16666666666666666],dtype=dtype),
        3: torch.tensor([ -0.28125, 0.260416666666667, 0.260416666666667, 0.260416666666667],dtype=dtype),
        4: torch.tensor([ 0.111690794839006, 0.054975871827661, 0.111690794839006, 0.054975871827661, 0.111690794839006, 0.054975871827661],dtype=dtype),
        5: torch.tensor([ 0.1125, 0.066197076394253, 0.0629695902724135, 0.066197076394253, 0.0629695902724135, 0.066197076394253, 0.0629695902724135],dtype=dtype),
        6: torch.tensor([ 0.0583931378631895, 0.0254224531851035, 0.0583931378631895, 0.0254224531851035, 0.0583931378631895, 0.0254224531851035, 0.041425537809187, 0.041425537809187, 0.041425537809187, 0.041425537809187, 0.041425537809187, 0.041425537809187],dtype=dtype),
        7: torch.tensor([ -0.074785022233841, 0.087807628716604, 0.026673617804419, 0.087807628716604, 0.026673617804419, 0.087807628716604, 0.026673617804419, 0.0385568804451285, 0.0385568804451285, 0.0385568804451285, 0.0385568804451285, 0.0385568804451285, 0.0385568804451285],dtype=dtype)
    }
    points_table = {
        1: torch.tensor([ [ 0.3333333333333333, 0.3333333333333333]],dtype=dtype),
        2: torch.tensor([ [ 0.16666666666666666, 0.16666666666666666], [ 0.6666666666666666, 0.16666666666666666], [ 0.16666666666666666, 0.6666666666666666]],dtype=dtype),
        3: torch.tensor([ [ 0.333333333333333, 0.333333333333333,], [ 0.2, 0.6,], [ 0.6, 0.2,], [ 0.2, 0.2,]],dtype=dtype),
        4: torch.tensor([ [ 0.445948490915965, 0.10810301816807,], [ 0.0915762135097699, 0.816847572980459,], [ 0.10810301816807, 0.445948490915965,], [ 0.816847572980459, 0.091576213509771,], [ 0.445948490915965, 0.445948490915965,], [ 0.091576213509771, 0.0915762135097699,]],dtype=dtype),
        5: torch.tensor([ [ 0.333333333333333, 0.333333333333333,], [ 0.470142064105115, 0.05971587178977,], [ 0.101286507323457, 0.797426985353087,], [ 0.05971587178977, 0.470142064105115,], [ 0.797426985353087, 0.101286507323456,], [ 0.470142064105115, 0.470142064105115,], [ 0.101286507323456, 0.101286507323457,]],dtype=dtype),
        6: torch.tensor([ [ 0.249286745170911, 0.501426509658179,], [ 0.0630890144915021, 0.873821971016996,], [ 0.501426509658179, 0.24928674517091,], [ 0.873821971016996, 0.063089014491502,], [ 0.24928674517091, 0.249286745170911,], [ 0.063089014491502, 0.0630890144915021,], [ 0.636502499121399, 0.053145049844817,], [ 0.310352451033784, 0.053145049844817,], [ 0.053145049844817, 0.310352451033784,], [ 0.053145049844817, 0.636502499121399,], [ 0.310352451033784, 0.636502499121399,], [ 0.636502499121399, 0.310352451033784,]],dtype=dtype),
        7: torch.tensor([ [ 0.333333333333333, 0.333333333333333,], [ 0.26034596607904, 0.47930806784192,], [ 0.065130102902216, 0.869739794195568,], [ 0.47930806784192, 0.26034596607904,], [ 0.869739794195568, 0.065130102902216,], [ 0.26034596607904, 0.26034596607904,], [ 0.065130102902216, 0.065130102902216,], [ 0.63844418856981, 0.048690315425316,], [ 0.312865496004874, 0.048690315425316,], [ 0.048690315425316, 0.312865496004874,], [ 0.048690315425316, 0.63844418856981,], [ 0.312865496004874, 0.63844418856981,], [ 0.63844418856981, 0.312865496004874,]],dtype=dtype)
    }
    assert order in weights_table, f"Invalid order {order} for tri quadrature, expected one of {weights_table.keys()}"
    weights = weights_table[order]
    points  = points_table[order]

    weights = weights.to(device)
    points  = points.to(device)

    return weights, points 

def tet_quadrature(order: int = 1,
                   dtype: torch.dtype = torch.float32,
                   device: torch.device = torch.device('cpu'),
                   ) -> Tensorx2:
    """Quadrature rule on the reference tetrahedron.

    Parameters
    ----------
    order : int, optional
        Quadrature order between 1 and 7. Defaults to 1.
    dtype : torch.dtype, optional
        Output dtype.
    device : torch.device, optional
        Output device.

    Returns
    -------
    weights : torch.Tensor
        1D tensor of shape ``[n_quadrature]``.
    points : torch.Tensor
        2D tensor of shape ``[n_quadrature, 3]``.
    """
    assert order >= 1 and order <= 7, f"Invalid order {order} for tet quadrature, expected one of [1,7]"
    weights_table = {
        1: torch.tensor([ 0.16666666666666666],dtype=dtype),
        2: torch.tensor([ 0.041666666666666664, 0.041666666666666664, 0.041666666666666664, 0.041666666666666664],dtype=dtype),
        3: torch.tensor([ -0.13333333333333333, 0.075, 0.075, 0.075, 0.075],dtype=dtype),
        4: torch.tensor([ -0.01315555555555555, 0.007622222222222217, 0.007622222222222217, 0.007622222222222217, 0.007622222222222217, 0.02488888888888888, 0.02488888888888888, 0.02488888888888888, 0.02488888888888888, 0.02488888888888888, 0.02488888888888888],dtype=dtype),
        5: torch.tensor([ 0.003174603174603167, 0.003174603174603167, 0.003174603174603167, 0.003174603174603167, 0.003174603174603167, 0.003174603174603167, 0.014764970790496783, 0.014764970790496783, 0.014764970790496783, 0.014764970790496783, 0.022139791114265117, 0.022139791114265117, 0.022139791114265117, 0.022139791114265117],dtype=dtype),
        6: torch.tensor([ 0.030283678097089182, 0.006026785714285717, 0.006026785714285717, 0.006026785714285717, 0.006026785714285717, 0.011645249086028967, 0.011645249086028967, 0.011645249086028967, 0.011645249086028967, 0.010949141561386449, 0.010949141561386449, 0.010949141561386449, 0.010949141561386449, 0.010949141561386449, 0.010949141561386449],dtype=dtype),
        7: torch.tensor([ 0.0066537917096946494, 0.0066537917096946494, 0.0066537917096946494, 0.0066537917096946494, 0.0016795351758867834, 0.0016795351758867834, 0.0016795351758867834, 0.0016795351758867834, 0.009226196923942399, 0.009226196923942399, 0.009226196923942399, 0.009226196923942399, 0.008035714285714283, 0.008035714285714283, 0.008035714285714283, 0.008035714285714283, 0.008035714285714283, 0.008035714285714283, 0.008035714285714283, 0.008035714285714283, 0.008035714285714283, 0.008035714285714283, 0.008035714285714283, 0.008035714285714283],dtype=dtype)
    }
    points_table = {
        1: torch.tensor([ [ 0.25, 0.25, 0.25]],dtype=dtype),
        2: torch.tensor([ [ 0.5854101966249685, 0.1381966011250105, 0.1381966011250105,], [ 0.1381966011250105, 0.1381966011250105, 0.1381966011250105,], [ 0.1381966011250105, 0.1381966011250105, 0.5854101966249685,], [ 0.1381966011250105, 0.5854101966249685, 0.1381966011250105,],],dtype=dtype,),
        3: torch.tensor([ [ 0.25, 0.25, 0.25,], [ 0.5, 0.1666666666666667, 0.1666666666666667,], [ 0.1666666666666667, 0.1666666666666667, 0.1666666666666667,], [ 0.1666666666666667, 0.1666666666666667, 0.5,], [ 0.1666666666666667, 0.5, 0.1666666666666667,],],dtype=dtype),
        4: torch.tensor([ [ 0.25, 0.25, 0.25,], [ 0.7857142857142857, 0.0714285714285714, 0.0714285714285714,], [ 0.0714285714285714, 0.0714285714285714, 0.0714285714285714,], [ 0.0714285714285714, 0.0714285714285714, 0.7857142857142857,], [ 0.0714285714285714, 0.7857142857142857, 0.0714285714285714,], [ 0.1005964238332008, 0.3994035761667992, 0.3994035761667992,], [ 0.3994035761667992, 0.1005964238332008, 0.3994035761667992,], [ 0.3994035761667992, 0.3994035761667992, 0.1005964238332008,], [ 0.3994035761667992, 0.1005964238332008, 0.1005964238332008,], [ 0.1005964238332008, 0.3994035761667992, 0.1005964238332008,], [ 0.1005964238332008, 0.1005964238332008, 0.3994035761667992,],],dtype=dtype),
        5: torch.tensor([ [ 0.0, 0.5, 0.5,], [ 0.5, 0.0, 0.5,], [ 0.5, 0.5, 0.0,], [ 0.5, 0.0, 0.0,], [ 0.0, 0.5, 0.0,], [ 0.0, 0.0, 0.5,], [ 0.6984197043243866, 0.1005267652252045, 0.1005267652252045,], [ 0.1005267652252045, 0.1005267652252045, 0.1005267652252045,], [ 0.1005267652252045, 0.1005267652252045, 0.6984197043243866,], [ 0.1005267652252045, 0.6984197043243866, 0.1005267652252045,], [ 0.0568813795204234, 0.3143728734931922, 0.3143728734931922,], [ 0.3143728734931922, 0.3143728734931922, 0.3143728734931922,], [ 0.3143728734931922, 0.3143728734931922, 0.0568813795204234,], [ 0.3143728734931922, 0.0568813795204234, 0.3143728734931922,],],dtype=dtype),
        6: torch.tensor([ [ 0.25, 0.25, 0.25,], [ 0.0, 0.3333333333333333, 0.3333333333333333,], [ 0.3333333333333333, 0.3333333333333333, 0.3333333333333333,], [ 0.3333333333333333, 0.3333333333333333, 0.0,], [ 0.3333333333333333, 0.0, 0.3333333333333333,], [ 0.7272727272727273, 0.0909090909090909, 0.0909090909090909,], [ 0.0909090909090909, 0.0909090909090909, 0.0909090909090909,], [ 0.0909090909090909, 0.0909090909090909, 0.7272727272727273,], [ 0.0909090909090909, 0.7272727272727273, 0.0909090909090909,], [ 0.4334498464263357, 0.0665501535736643, 0.0665501535736643,], [ 0.0665501535736643, 0.4334498464263357, 0.0665501535736643,], [ 0.0665501535736643, 0.0665501535736643, 0.4334498464263357,], [ 0.0665501535736643, 0.4334498464263357, 0.4334498464263357,], [ 0.4334498464263357, 0.0665501535736643, 0.4334498464263357,], [ 0.4334498464263357, 0.4334498464263357, 0.0665501535736643,],],dtype=dtype),
        7: torch.tensor([ [ 0.3561913862225449, 0.2146028712591517, 0.2146028712591517,], [ 0.2146028712591517, 0.2146028712591517, 0.2146028712591517,], [ 0.2146028712591517, 0.2146028712591517, 0.3561913862225449,], [ 0.2146028712591517, 0.3561913862225449, 0.2146028712591517,], [ 0.877978124396166, 0.0406739585346113, 0.0406739585346113,], [ 0.0406739585346113, 0.0406739585346113, 0.0406739585346113,], [ 0.0406739585346113, 0.0406739585346113, 0.877978124396166,], [ 0.0406739585346113, 0.877978124396166, 0.0406739585346113,], [ 0.0329863295731731, 0.3223378901422757, 0.3223378901422757,], [ 0.3223378901422757, 0.3223378901422757, 0.3223378901422757,], [ 0.3223378901422757, 0.3223378901422757, 0.0329863295731731,], [ 0.3223378901422757, 0.0329863295731731, 0.3223378901422757,], [ 0.2696723314583159, 0.0636610018750175, 0.0636610018750175,], [ 0.0636610018750175, 0.2696723314583159, 0.0636610018750175,], [ 0.0636610018750175, 0.0636610018750175, 0.2696723314583159,], [ 0.6030056647916491, 0.0636610018750175, 0.0636610018750175,], [ 0.0636610018750175, 0.6030056647916491, 0.0636610018750175,], [ 0.0636610018750175, 0.0636610018750175, 0.6030056647916491,], [ 0.0636610018750175, 0.2696723314583159, 0.6030056647916491,], [ 0.2696723314583159, 0.6030056647916491, 0.0636610018750175,], [ 0.6030056647916491, 0.0636610018750175, 0.2696723314583159,], [ 0.0636610018750175, 0.6030056647916491, 0.2696723314583159,], [ 0.2696723314583159, 0.0636610018750175, 0.6030056647916491,], [ 0.6030056647916491, 0.2696723314583159, 0.0636610018750175,],],dtype=dtype)
    }
    assert order in weights_table, f"Invalid order {order} for tet quadrature, expected one of {weights_table.keys()}"
    weights = weights_table[order]
    points  = points_table[order]

    weights = weights.to(device)
    points  = points.to(device)

    return weights, points


def quad_quadrature(order: int = 1,
                    dtype: torch.dtype = torch.float32,
                    device: torch.device = torch.device('cpu'),
                    ) -> Tensorx2:
    """Tensor-product quadrature rule on the reference quadrilateral.

    Built from :func:`lin_quadrature` along each axis.

    Parameters
    ----------
    order : int, optional
        Quadrature order between 1 and 7. Defaults to 1.
    dtype : torch.dtype, optional
        Output dtype.
    device : torch.device, optional
        Output device.

    Returns
    -------
    weights : torch.Tensor
        1D tensor of shape ``[n_quadrature]``.
    points : torch.Tensor
        2D tensor of shape ``[n_quadrature, 2]``.
    """
    weights, points = lin_quadrature(order, dtype, device) # [n_quadrature], [n_quadrature, 1]

    w_x, w_y = torch.meshgrid(weights, weights, indexing='xy')
    w_x, w_y = w_x.flatten(), w_y.flatten()
    p_x, p_y = torch.meshgrid(points[:, 0], points[:, 0], indexing='xy')
    p_x, p_y = p_x.flatten(), p_y.flatten() 

    weights = w_x * w_y
    points  = torch.stack([p_x, p_y], -1)

    return weights, points

def hex_quadrature(order: int = 1,
                   dtype: torch.dtype = torch.float32,
                   device: torch.device = torch.device('cpu'),
                   ) -> Tensorx2:
    """Tensor-product quadrature rule on the reference hexahedron.

    Built from :func:`lin_quadrature` along each axis.

    Parameters
    ----------
    order : int, optional
        Quadrature order between 1 and 7. Defaults to 1.
    dtype : torch.dtype, optional
        Output dtype.
    device : torch.device, optional
        Output device.

    Returns
    -------
    weights : torch.Tensor
        1D tensor of shape ``[n_quadrature]``.
    points : torch.Tensor
        2D tensor of shape ``[n_quadrature, 3]``.
    """
    weights, points = lin_quadrature(order, dtype, device)
    w_x, w_y, w_z   = torch.meshgrid(weights, weights, weights, indexing='xy')
    w_x, w_y, w_z   = w_x.flatten(), w_y.flatten(), w_z.flatten()
    p_x, p_y, p_z   = torch.meshgrid(points[:, 0], points[:, 0], points[:, 0], indexing='xy')
    p_x, p_y, p_z   = p_x.flatten(), p_y.flatten(), p_z.flatten()

    weights         = w_x * w_y * w_z 
    points          = torch.stack([p_x, p_y, p_z], dim=-1)

    return weights, points

def pyr_quadrature(order: int = 1,
                   dtype: torch.dtype = torch.float32,
                   device: torch.device = torch.device('cpu'),
                   ) -> Tensorx2:
    r"""Quadrature rule on the reference pyramid.

    Conical product rule: the collapse :math:`(x, y, z) = ((1 - t)u, (1 - t)v, t)`
    maps the unit cube onto the pyramid with Jacobian :math:`(1 - t)^2`.
    ``order`` Gauss-Legendre points are used in :math:`u` and :math:`v`, and
    ``order`` Gauss-Jacobi points for the weight :math:`(1 - t)^2` in
    :math:`t`. The rule is exact for every integrand that is a polynomial of
    degree at most ``2 * order - 1`` in each of :math:`u, v, t`: in particular
    for :math:`x^a y^b z^c` with :math:`a + b + c \le 2\,\mathrm{order} - 1`,
    and for the mass and stiffness integrands of the order ``order - 1``
    pyramid basis (:class:`~tensormesh.element.PyramidBasisFns`) on affine
    pyramids.

    Parameters
    ----------
    order : int, optional
        Number of points per direction, between 1 and 7. Defaults to 1.
    dtype : torch.dtype, optional
        Output dtype.
    device : torch.device, optional
        Output device.

    Returns
    -------
    weights : torch.Tensor
        1D tensor of shape ``[n_quadrature]``.
    points : torch.Tensor
        2D tensor of shape ``[n_quadrature, 3]``.
    """
    lin_weights, lin_points = lin_quadrature(order, dtype,device)
    t, w_t = roots_jacobi(order, 2, 0)                              # weight (1 - s)^2 on [-1, 1]
    t      = torch.tensor((1 + t) / 2, dtype=dtype, device=device)  # s -> t on [0, 1]
    w_t    = torch.tensor(w_t / 8, dtype=dtype, device=device)      # (1 - s)^2 ds = 8 (1 - t)^2 dt

    w_x, w_y, w_z = torch.meshgrid(lin_weights, lin_weights, w_t, indexing='xy')
    w_x, w_y, w_z = w_x.flatten(), w_y.flatten(), w_z.flatten()
    p_x, p_y, p_z = torch.meshgrid(lin_points[:,0], lin_points[:,0], t, indexing='xy')
    p_x, p_y, p_z = p_x.flatten(), p_y.flatten(), p_z.flatten()
    scaling       = 1 - p_z
    p_x, p_y      = scaling * p_x, scaling * p_y

    weights       = w_x * w_y * w_z   # the Jacobian (1 - t)^2 is in w_z
    points        = torch.stack([p_x, p_y, p_z], -1)

    return weights, points

def pri_quadrature(order: int = 1,
                   dtype: torch.dtype = torch.float32,
                   device: torch.device = torch.device('cpu'),
                   ) -> Tensorx2:
    """Quadrature rule on the reference prism (wedge).

    Tensor product of :func:`tri_quadrature` (triangle base) with
    :func:`lin_quadrature` (height direction).

    Parameters
    ----------
    order : int, optional
        Quadrature order between 1 and 7. Defaults to 1.
    dtype : torch.dtype, optional
        Output dtype.
    device : torch.device, optional
        Output device.

    Returns
    -------
    weights : torch.Tensor
        1D tensor of shape ``[n_quadrature]``.
    points : torch.Tensor
        2D tensor of shape ``[n_quadrature, 3]``.
    """
    
    tri_weights, tri_points = tri_quadrature(order, dtype, device)  
    lin_weights, lin_points = lin_quadrature(order, dtype, device)

    weights = (tri_weights[:, None] * lin_weights[None, :]).flatten()
    points = torch.cat([
        tri_points[:, None, :].repeat(1, len(lin_weights), 1),
        lin_points[None, :, :].repeat(len(tri_weights), 1, 1) 
    ], -1).reshape(-1, 3)

    return weights, points


# facet quadrature 

def facet_quadrature_2d(facet_mapping: Polynomials,
                        order: int = 1,
                        transform: bool = True,
                        ) -> Tensorx2:
    """Facet quadrature for 2D elements (triangle, quadrilateral).

    Generates a 1D Gauss-Legendre rule on the reference line, then optionally
    pushes it through ``facet_mapping`` to obtain points in the reference
    element coordinates.

    Parameters
    ----------
    facet_mapping : Polynomials
        Polynomial mapping from facet-local to cell-local coordinates. Shape
        ``[n_facet, dim]`` with ``n_vars = dim - 1`` and ``n_terms = dim``.
    order : int, optional
        Quadrature order. Defaults to 1.
    transform : bool, optional
        If ``True`` (default), return points in cell coordinates. If
        ``False``, return points in facet-local coordinates.

    Returns
    -------
    weights : torch.Tensor
        Shape ``[n_facet, n_quadrature_per_facet]`` when ``transform`` is
        ``True``; shape ``[n_quadrature_per_facet]`` otherwise.
    points : torch.Tensor
        Shape ``[n_facet, n_quadrature_per_facet, dim]`` when ``transform``
        is ``True``; shape ``[n_quadrature_per_facet, dim - 1]`` otherwise.
    """
    dtype = facet_mapping.dtype
    device= facet_mapping.device

    if transform:
        facet_quadrature_weights, facet_quadrature_points = lin_quadrature(order, dtype, device)   # [n_quadrature_per_facet], [n_quadrature_per_facet, dim-1]
        facet_quadrature_points = facet_quadrature_points[:, None, None, :]
        facet_quadrature_points = facet_quadrature_points.repeat(1, *facet_mapping.shape, 1) # [n_quadrature_per_facet, n_facet, dim, dim-1]
        facet_quadrature_points = facet_mapping(facet_quadrature_points)            # [n_quadrature_per_facet, n_facet, dim]
        facet_quadrature_points = facet_quadrature_points.permute(1, 0, 2)        # [n_facet, n_quadrature_per_facet, dim]
        n_facet = facet_quadrature_points.shape[0]
        facet_quadrature_weights= facet_quadrature_weights.repeat(n_facet, 1)
        return facet_quadrature_weights, facet_quadrature_points
    else:
        return lin_quadrature(order, dtype, device) # [n_quadrature_per_facet], [n_quadrature_per_facet, dim-1]
                            
def tet_facet_quadrature(facet_mapping: Polynomials,
                         order: int = 1,
                         transform: bool = True,
                         ) -> Tensorx2:
    """Facet quadrature for the tetrahedron (triangular facets).

    Generates a 2D triangle rule on the reference facet, then optionally
    pushes it through ``facet_mapping`` to obtain points in the reference
    element coordinates.

    Parameters
    ----------
    facet_mapping : Polynomials
        Polynomial mapping from facet-local to cell-local coordinates. Shape
        ``[n_facet, dim]`` with ``n_vars = dim - 1`` and ``n_terms = dim``.
    order : int, optional
        Quadrature order. Defaults to 1.
    transform : bool, optional
        If ``True`` (default), return points in cell coordinates. If
        ``False``, return points in facet-local coordinates.

    Returns
    -------
    weights : torch.Tensor
        Shape ``[n_facet, n_quadrature_per_facet]`` when ``transform`` is
        ``True``; shape ``[n_quadrature_per_facet]`` otherwise.
    points : torch.Tensor
        Shape ``[n_facet, n_quadrature_per_facet, dim]`` when ``transform``
        is ``True``; shape ``[n_quadrature_per_facet, dim - 1]`` otherwise.
    """
    dtype = facet_mapping.dtype
    device= facet_mapping.device
    if transform:
        facet_quadrature_weights, facet_quadrature_points = tri_quadrature(order, dtype, device) # [n_quadrature_per_facet], [n_quadrature_per_facet, dim-1]
        facet_quadrature_points = facet_quadrature_points[:, None, None, :]
        facet_quadrature_points = facet_quadrature_points.repeat(1, *facet_mapping.shape, 1) # [n_quadrature_per_facet, n_facet, dim, dim-1]
        facet_quadrature_points = facet_mapping(facet_quadrature_points)          # [n_quadrature_per_facet, n_facet, dim]
        facet_quadrature_points = facet_quadrature_points.permute(1, 0, 2) # [n_facet, n_quadrature_per_facet, dim]
        n_facet                 = facet_quadrature_points.shape[0]
        facet_quadrature_weights = facet_quadrature_weights.repeat(n_facet, 1)
        return facet_quadrature_weights, facet_quadrature_points
    else:
        return tri_quadrature(order, dtype, device)                            # [n_quadrature_per_facet], [n_quadrature_per_facet, dim-1]

def hex_facet_quadrature(facet_mapping: Polynomials,
                         order: int = 1,
                         transform: bool = True,
                         ) -> Tensorx2:
    """Facet quadrature for the hexahedron (quadrilateral facets).

    Generates a 2D tensor-product quadrilateral rule on the reference facet,
    then optionally pushes it through ``facet_mapping`` to obtain points in
    the reference element coordinates.

    Parameters
    ----------
    facet_mapping : Polynomials
        Polynomial mapping from facet-local to cell-local coordinates. Shape
        ``[n_facet, dim]`` with ``n_vars = dim - 1`` and ``n_terms = dim``.
    order : int, optional
        Quadrature order. Defaults to 1.
    transform : bool, optional
        If ``True`` (default), return points in cell coordinates. If
        ``False``, return points in facet-local coordinates.

    Returns
    -------
    weights : torch.Tensor
        Shape ``[n_facet, n_quadrature_per_facet]`` when ``transform`` is
        ``True``; shape ``[n_quadrature_per_facet]`` otherwise.
    points : torch.Tensor
        Shape ``[n_facet, n_quadrature_per_facet, dim]`` when ``transform``
        is ``True``; shape ``[n_quadrature_per_facet, dim - 1]`` otherwise.
    """
    dtype = facet_mapping.dtype
    device= facet_mapping.device
    if transform:
        facet_quadrature_weights, facet_quadrature_points = quad_quadrature(order, dtype, device)  # [n_quadrature_per_facet], [n_quadrature_per_facet, dim-1]
        facet_quadrature_points = facet_quadrature_points[:, None, None, :]
        facet_quadrature_points = facet_quadrature_points.repeat(1, *facet_mapping.shape, 1) # [n_quadrature_per_facet, n_facet, dim, dim-1]
        facet_quadrature_points = facet_mapping(facet_quadrature_points)      # [n_quadrature_per_facet, n_facet, dim]
        facet_quadrature_points = facet_quadrature_points.permute(1, 0, 2)  # [n_facet, n_quadrature_per_facet, dim]
        n_facet                 = facet_quadrature_points.shape[0]
        facet_quadrature_weights = facet_quadrature_weights.repeat(n_facet, 1)
        return facet_quadrature_weights, facet_quadrature_points
       
    else:
        return quad_quadrature(order, dtype, device)

def mix_facet_quadrature_3d(facet_mapping: Polynomials,
                            facets: Tuple[Tuple[int, ...], ...],
                            order: int = 1,
                            transform: bool = True,
                            ) -> Tensorx4:
    """Facet quadrature for 3D elements with mixed triangular/quadrilateral facets.

    Used by prisms and pyramids, where some facets are triangles and the
    rest are quadrilaterals. Returns four tensors — weights and points for
    each facet kind.

    Parameters
    ----------
    facet_mapping : Polynomials
        Polynomial mapping from facet-local to cell-local coordinates. Shape
        ``[n_facet, dim]`` with ``n_vars = dim - 1`` and ``n_terms = dim``.
    facets : Tuple[Tuple[int, ...], ...]
        Per-facet vertex-index tuples — ``len == 3`` marks a triangular
        facet, ``len == 4`` marks a quadrilateral facet.
    order : int, optional
        Quadrature order. Defaults to 1.
    transform : bool, optional
        If ``True`` (default), return points in cell coordinates. If
        ``False``, return points in facet-local coordinates.

    Returns
    -------
    tri_weights : torch.Tensor
        Triangular-facet weights. Shape
        ``[n_tri_facet, n_quadrature_per_tri_facet]`` when ``transform`` is
        ``True``; shape ``[n_quadrature_per_tri_facet]`` otherwise.
    tri_points : torch.Tensor
        Triangular-facet points. Shape
        ``[n_tri_facet, n_quadrature_per_tri_facet, dim]`` when ``transform``
        is ``True``; shape ``[n_quadrature_per_tri_facet, dim - 1]`` otherwise.
    quad_weights : torch.Tensor
        Quadrilateral-facet weights. Shape
        ``[n_quad_facet, n_quadrature_per_quad_facet]`` when ``transform`` is
        ``True``; shape ``[n_quadrature_per_quad_facet]`` otherwise.
    quad_points : torch.Tensor
        Quadrilateral-facet points. Shape
        ``[n_quad_facet, n_quadrature_per_quad_facet, dim]`` when ``transform``
        is ``True``; shape ``[n_quadrature_per_quad_facet, dim - 1]`` otherwise.
    """
    dtype  = facet_mapping.dtype
    device = facet_mapping.device
    if transform:
        tri_facet_quadrature_weights, tri_facet_quadrature_points = tri_quadrature(order, dtype, device)      # [n_quadrature_per_tri_facet, dim-1]
        quad_facet_quadrature_weight, quad_facet_quadrature_points= quad_quadrature(order, dtype, device)     # [n_quadrature_per_quad_facet,dim-1]
        tri_mask  = torch.tensor([len(face) == 3 for face in facets])
        quad_mask = ~tri_mask 
        n_tri_facet  = tri_mask.sum().item()
        n_quad_facet = quad_mask.sum().item()   
        dim          = 3

        tri_facet_mapping     = facet_mapping[tri_mask]              # Polynomials [n_tri_facet, dim] n_vars=dim-1 n_term = dim
        quad_facet_mapping    = facet_mapping[quad_mask]             # Polynomials [n_quad_facet, dim] n_vars=dim-1 n_term = dim
     
        assert isinstance(tri_facet_mapping, Polynomials) and tri_facet_mapping.shape  == (n_tri_facet, dim)
        assert isinstance(quad_facet_mapping, Polynomials) and quad_facet_mapping.shape == (n_quad_facet, dim)  

        tri_facet_quadrature_points  = tri_facet_quadrature_points[:, None, None, :]
        quad_facet_quadrature_points = quad_facet_quadrature_points[:, None, None, :]
        tri_facet_quadrature_points  = tri_facet_quadrature_points.repeat(1, *tri_facet_mapping.shape, 1) # [n_quadrature_per_tri_facet, n_tri_facet, dim, dim-1]
        quad_facet_quadrature_points = quad_facet_quadrature_points.repeat(1, *quad_facet_mapping.shape, 1) # [n_quadrature_per_quad_facet, n_quad_facet, dim, dim-1]
        
        tri_facet_quadrature_points  = tri_facet_mapping(tri_facet_quadrature_points)   # [n_quadrature_per_tri_facet, n_tri_facet, dim]
        quad_facet_quadrature_points = quad_facet_mapping(quad_facet_quadrature_points) # [n_quadrature_per_quad_facet, n_quad_facet, dim]

        tri_facet_quadrature_points  = tri_facet_quadrature_points.permute(1, 0, 2)
        quad_facet_quadrature_points = quad_facet_quadrature_points.permute(1, 0, 2)

        n_tri_facet = tri_facet_quadrature_points.shape[0]
        n_quad_facet = quad_facet_quadrature_points.shape[0]

        tri_facet_quadrature_weights = tri_facet_quadrature_weights.repeat(n_tri_facet, 1)
        quad_facet_quadrature_weight = quad_facet_quadrature_weight.repeat(n_quad_facet, 1)

        return (tri_facet_quadrature_weights,
                tri_facet_quadrature_points, 
                quad_facet_quadrature_weight, 
                quad_facet_quadrature_points)
          
    else:
        return (*tri_quadrature(order,dtype,device), *quad_quadrature(order,dtype, device))
      