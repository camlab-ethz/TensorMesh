import torch



def trace(x):
    """

    .. math::
    
            \\text{trace}(A)_{\\cdots} = \\sum_{i=1}^n A_{\\cdots ii}

    Parameters
    ----------
    x : torch.Tensor 
        :math:`[..., D, D]`, where :math:`D` is the dimension of the matrix

    Returns
    -------
    torch.Tensor
        :math:`[...]` 
    """
    return torch.einsum(f"...ii->...", x)

def dot(a, b, reduce_dim=-1):
    """

    .. math::

        \\text{dot}(A, B)_{\\cdots ab} = \\sum_{i=1}^n A_{\\cdots ai} B_{\\cdots bi}

    Parameters
    ----------
    a : torch.Tensor 
        :math:`[..., B, D]`, where :math:`B` is the number of basis, :math:`D` is the dimension of the matrix
    b : torch.Tensor
        :math:`[..., B, D]`, where :math:`B` is the number of basis, :math:`D` is the dimension of the matrix
    Returns
    -------
    torch.Tensor
        :math:`[..., B, B]`, where :math:`B` is the number of basis
    """
    if reduce_dim == -1:
        return torch.einsum("...ik,...jk->...ij", a, b)
    elif reduce_dim == -2:
        return torch.einsum("...ika,...jkb->...ijab", a, b)
    else:
        raise ValueError(f"reduce_dim must be -1 or -2, but got {reduce_dim}")
    
def ddot(a, b):
    """

    .. math::
    
            \\text{ddot}(A, B)_{\\cdots ab} = \\sum_{i=1}^n A_{\\cdots aij} B_{\\cdots bij}

    Parameters
    ----------
    a : torch.Tensor
        :math:`[..., B, D, D]`, where :math:`B` is the number of basis, :math:`D` is the dimension of the matrix
    b : torch.Tensor   
        :math:`[..., B, D, D]`, where :math:`B` is the number of basis, :math:`D` is the dimension of the matrix     

    Returns
    --------
    torch.Tensor
        :math:`[..., B, B]`, where :math:`B` is the number of basis
    """
    return torch.einsum("...imn,...jmn->...ij", a, b)

def mul(a, b):
    """

    .. math::

        \\text{mul}(A, B)_{\\cdots ij} = \\sum_{i=1}^n A_{\\cdots i} B_{\\cdots j} 

    Parameters
    ----------
    a : torch.Tensor
        :math:`[..., B]`, where :math:`B` is the number of basis
    b : torch.Tensor
        :math:`[..., B]`, where :math:`B` is the number of basis
    Returns
    -------
    torch.Tensor
        [..., n_basis, n_basis]
    """
    return torch.einsum("...i,...j->...ij", a, b)

def eye(value, dim):
    """

    .. math::

        \\text{eye}(v, n)_{\\cdots ij} = \\begin{cases} v_{\\cdots}, & i=j \\\\ 0, & i \\neq j \\end{cases}

    Parameters
    ----------
    value : torch.Tensor
        :math:`[...]`, the filled value of the eye
    dim : int
        :math:`D`, the dimension of the eye

    Returns 
    -------
    torch.Tensor
        :math:`[..., D, D]`
    """
    dims = value.shape
    zeros = torch.zeros_like(value)
    result = torch.stack([torch.stack([zeros if j != i else value for j in range(dim)],-1) for i in range(dim)], -2)
   
    return result

def sym(a):
    """

    .. math::

        \\text{sym}(A)_{\\cdots ij} = \\frac{1}{2} (A_{\\cdots i} + A_{\\cdots j})

    Parameters
    ----------
    a : torch.Tensor
        :math:`[..., D]`, where :math:`D` is the dimension of the matrix
    Returns
    -------
    torch.Tensor
        :math:`[..., D]`, where :math:`D` is the dimension of the matrix
    """
    return 0.5 * (a[..., None] + a[..., None, :])

def vector(x):
    """

    .. math::

        \\text{vector}(A) = \\begin{bmatrix}A_{\\cdots}^0\\ \\vdots \\ A_{\\cdots}^{n_{\\text{row}}-1\end{bmatrix}

    Parameters
    ----------
    x: : List[torch.Tensor]
        tensor list of shape [...]
    Returns
    -------
    torch.Tensor
        :math:`[..., n_{\\text{row}}]`
    """
    return torch.stack(x, -1)

def matrix(x):
    """

    .. math::

        \\text{matrix}(A) = 
        \\begin{bmatrix}
        A_{\\cdots}^{0,0} & \\cdots & A_{\\cdots}^{n_{\\text{col}}-1} \\\\
        \\vdots & \\ddots & \\vdots \\\\
        A_{\\cdots}^{0,n_{\\text{row}}-1} & \\cdots & A_{\\cdots}^{n_{\\text{col}}-1,n_{\\text{row}}-1}
        \\end{bmatrix}


    Parameters
    ----------
        x : List[List[torch.Tensor]]
            tensor list of list of shape [...]
    Returns
    -------
    torch.Tensor
            :math:`[..., n_{\\text{col}}, n_{\\text{row}}]`
    """
    return torch.stack([torch.stack(row, -1) for row in x], -2)

def transpose(x):
    """

    .. math::
    
            \\text{transpose}(A)_{\\cdots ij} = A_{\\cdots ji}  


    Parameters
    ----------
    x : torch.Tensor
        :math:`[..., a, b]`
    Returns
    -------
    torch.Tensor
        :math:`[..., b, a]`
    """
    return torch.einsum("...ij->...ji", x)

def matmul(a,  b):
    """

    .. math::
    
            \\text{matmul}(A, B)_{\\cdots ij} = \\sum_{k=1}^n A_{\\cdots ik} B_{\\cdots kj} 

    Parameters:
    -----------
    a : torch.Tensor
        :math:`[..., a, b]`
    b : torch.Tensor
        :math:`[..., b, c]`
    Returns:
    --------
    torch.Tensor
        :math:`[..., a, c]`
    """
    return torch.einsum("...ij,...jk->...ik", a, b)