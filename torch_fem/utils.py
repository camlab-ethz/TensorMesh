import torch

def is_float(x):
    if  isinstance(x, torch.Tensor):
        return x.dtype == torch.float32 or x.dtype == torch.float64 or x.dtype == torch.float16
    else:
        return x == torch.float32 or x == torch.float64 or x == torch.float16

def trace(tensor):
    return torch.einsum(f"...ii->...", tensor)

def dot(a, b, reduce_dim=-1):
    """
        Parameters:
        -----------
            a [..., n_basis, n_dim]
            b [..., n_basis, n_dim]
        Returns:
        --------
            c [..., n_basis, n_basis]
    """
    if reduce_dim == -1:
        return torch.einsum("...ik,...jk->...ij", a, b)
    elif reduce_dim == -2:
        return torch.einsum("...ika,...jkb->...ijab", a, b)
    else:
        raise ValueError(f"reduce_dim must be -1 or -2, but got {reduce_dim}")
    
def ddot(a, b):
    """
        Parameters:
        -----------
            a [..., n_basis, n_dim, n_dim]
            b [..., n_basis, n_dim, n_dim]
        Returns:
        --------
            c [..., n_basis, n_basis]
    """
    return torch.einsum("...imn,...jmn->...ij", a, b)

def mul(a, b):
    """
        Parameters:
        -----------
            a [..., n_basis]
            b [..., n_basis]
        Returns:
        --------
            c [..., n_basis, n_basis]
    """
    return torch.einsum("...i,...j->...ij", a, b)

def eye(value, dim):
    dims = value.shape
    zeros = torch.zeros_like(value)
    result = torch.stack([torch.stack([zeros if j != i else value for j in range(dim)],-1) for i in range(dim)], -2)
   
    return result

def sym(a):
    """
        Parameters:
        -----------
            a [..., n_basis, n_dim]
        Returns:
        --------
            c [..., n_basis, n_dim, n_dim]
    """
    return 0.5 * (a[..., None] + a[..., None, :])

def vector(x):
    """
        Parameters:
        -----------
            x: List[torch.Tensor(...)]
        Returns:
        --------
            y: torch.Tensor[..., n_row]
    """
    return torch.stack(x, -1)

def matrix(x):
    """
        Parameters:
        -----------
            x: List[List[torch.Tensor(...)]]
        Returns:
        --------
            y: torch.Tensor[..., n_row, n_col]
    """
    return torch.stack([torch.stack(row, -1) for row in x], -2)

def transpose(x):
    """
        Parameters:
        -----------
            x: torch.Tensor[..., a, b]
        Returns:
        --------
            y: torch.Tensor[..., b, a]
    """
    return torch.einsum("...ij->...ji", x)

def matmul(a,  b):
    """
        Parameters:
        -----------
            a: torch.Tensor[..., a, b]
            b: torch.Tensor[..., b, c]
        Returns:
        --------
            c: torch.Tensor[..., a, c]
    """
    return torch.einsum("...ij,...jk->...ik", a, b)