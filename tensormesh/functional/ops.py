import torch
from typing import Sequence

#################
# Basic Operation
#################

def sym(a:torch.Tensor)->torch.Tensor:
    """

    .. math::

        \\text{sym}(A)_{\\cdots ij} = \\frac{1}{2} (A_{\\cdots i} + A_{\\cdots j})

    Examples
    --------
    >>> x = torch.tensor([1., 2.])
    >>> sym(x)
    tensor([[1.0000, 1.5000],
            [1.5000, 2.0000]])

    >>> x = torch.tensor([1., 2., 3.])
    >>> sym(x)
    tensor([[1.0000, 1.5000, 2.0000],
            [1.5000, 2.0000, 2.5000],
            [2.0000, 2.5000, 3.0000]])

    Parameters
    ----------
    a : torch.Tensor
        :math:`[..., D]`, where :math:`D` is the dimension of the matrix
    Returns
    -------
    torch.Tensor
        :math:`[..., D, D]`, where :math:`D` is the dimension of the matrix
    """
    return 0.5 * (a[..., None] + a[..., None, :])

def skew(x:torch.Tensor, 
         sign:bool=True,
         at_least2d:bool = False)->torch.Tensor:
    r"""
    Compute the skew-symmetric matrix from a vector.

    For 2D:
    
    .. math::

        \text{skew}\left(\begin{bmatrix}
            v_1 \\
            v_2 
        \end{bmatrix}\right) = \begin{cases}
            \begin{bmatrix}
                -v_2 \\
                v_1 
            \end{bmatrix} & \text{if sign=True} \\[1em]
            \begin{bmatrix}
                v_2 \\
                v_1 
            \end{bmatrix} & \text{if sign=False}
        \end{cases}

    For 3D:

    .. math::

        \text{skew}(v) = \begin{cases}
            \begin{bmatrix}
                0 & -v_3 & v_2 \\
                v_3 & 0 & -v_1 \\
                -v_2 & v_1 & 0
            \end{bmatrix} & \text{if sign=True} \\[1em]
            \begin{bmatrix}
                0 & v_3 & v_2 \\
                v_3 & 0 & v_1 \\
                v_2 & v_1 & 0
            \end{bmatrix} & \text{if sign=False}
        \end{cases}

    Examples
    --------
    .. code-block:: python

        >>> x = torch.tensor([1., 2.])
        >>> skew(x)
        tensor([-2.,  1.])
        
        >>> skew(x, sign=False)
        tensor([2., 1.])
        
        >>> skew(x, at_least2d=True)
        tensor([[-2.,  1.]])
        
        >>> x = torch.tensor([1., 2., 3.])
        >>> skew(x)
        tensor([[ 0., -3.,  2.],
               [ 3.,  0., -1.],
               [-2.,  1.,  0.]])
        
        >>> skew(x, sign=False)
        tensor([[0., 3., 2.],
               [3., 0., 1.],
               [2., 1., 0.]])

    Parameters
    ----------
    x : torch.Tensor
        1D Tensor of shape [2] or [3], representing a vector in :math:`\mathbb{R}^2` or :math:`\mathbb{R}^3`
    sign : bool, optional
        If True, use negative signs in skew matrix. Default is True.
    at_least2d : bool, optional
        If True, ensure output is at least 2D for 2D case. Default is False.

    Returns
    -------
    torch.Tensor
        For 2D case:
            - 1D Tensor of shape [2] if at_least2d=False
            - 2D Tensor of shape [1,2] if at_least2d=True
        For 3D case:
            - 2D Tensor of shape [3,3]
            
        The skew-symmetric matrix representation of the input vector.
    """
    assert x.ndim == 1, f"x must be a 1D vector, but got shape {x.shape}"
    dim = x.shape[-1]
    assert dim in (2, 3), f"x vector must be of length 2 or 3, but got {dim}"

    if dim == 2:
        # Out-of-place construction for vmap compatibility
        y0 = -x[1] if sign else x[1]
        y1 = x[0]
        y = torch.stack([y0, y1])

        if at_least2d:
            y = y[None, :]

    elif dim == 3:
        # Out-of-place construction for vmap compatibility
        zero = torch.zeros_like(x[0])
        s = -1.0 if sign else 1.0
        # Row 0: [0, -x[2], x[1]] or [0, x[2], x[1]]
        row0 = torch.stack([zero, s * x[2], x[1]])
        # Row 1: [x[2], 0, -x[0]] or [x[2], 0, x[0]]  
        row1 = torch.stack([x[2], zero, s * x[0]])
        # Row 2: [-x[1], x[0], 0] or [x[1], x[0], 0]
        row2 = torch.stack([s * x[1], x[0], zero])
        y = torch.stack([row0, row1, row2])
    else:
        raise ValueError(f"dimension must be 2 or 3, but got {dim}")
    return y

#################
# Clamp Min Operation
#################

def sqrt(x:torch.Tensor)->torch.Tensor:  
    r"""Square root function that returns 0 for negative inputs.

    This function computes the square root of the input tensor, but clamps negative values to 0 first.
    This avoids NaN values that would occur from taking the square root of negative numbers.

    .. math::

        \sqrt{x} = \begin{cases}
            \sqrt{x} & \text{if } x \geq 0 \\
            0 & \text{if } x < 0
        \end{cases}

    Examples
    --------
    >>> x = torch.tensor([-1.0, 0.0, 4.0])
    >>> sqrt(x)
    tensor([0.0000, 0.0000, 2.0000])


    Parameters
    ----------
    x : torch.Tensor
        :math:`[...]`

    Returns
    -------
    torch.Tensor
        :math:`[...]`
    """
    x = torch.clamp_min(x, 0.)
    return torch.sqrt(x)

def divide(x:torch.Tensor, y:torch.Tensor)->torch.Tensor:
    r"""Safe division function that returns 0 for division by zero.

    This function performs element-wise division of x by y, but returns 0 wherever y is 0.
    This avoids NaN/Inf values that would occur from dividing by zero.

    .. math::

        \frac{x}{y} = \begin{cases}
            \frac{x}{y} & \text{if } y \neq 0 \\
            0 & \text{if } y = 0
        \end{cases}
    
    Examples
    --------
    >>> x = torch.tensor([1.0, 2.0, 3.0])
    >>> y = torch.tensor([2.0, 0.0, 4.0]) 
    >>> divide(x, y)
    tensor([0.5000, 0.0000, 0.7500])

    Parameters
    ----------
    x : torch.Tensor
        :math:`[...]`
    y : torch.Tensor
        :math:`[...]`
    Returns
    -------
    torch.Tensor
        :math:`[...]`
    """
    return torch.where(y == 0., 0., x/y)