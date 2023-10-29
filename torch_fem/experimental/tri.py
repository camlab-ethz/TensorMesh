

import torch 
import scipy.special

# x1, y1 = 0, 0
# x2, y2 = 1, 0
# x3, y3 = 0, 1

# A = 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
# M = torch.tensor([
#     [y2-y3,  x3-x2, x2*y3 - x3*y2],
#     [y3-y1,  x1-x3, x3*y1 - x1*y3],
#     [y1-y2,  x2-x1, 1 - x2*y3 + x3*y2 - x3*y1 + x1*y3]
# ]) / (2*A)

M = torch.tensor([
    [-1, -1, 1],
    [1, 0, 0],
    [0, 1, 0]
])

def shape_val_p1(local_coordinates):
    """
        Parameters:
        -----------
            local_coordinates:torch.Tensor [...,n_dim]
                n_dim = 2 for triangle
                the local coordinates of the quadrature points
        Returns:
        --------
            phi: torch.Tensor of shape [..., n_basis]
                the base functions
    """
    pass 


def shape_grad_p1(local_coordinates):
    """
        Parameters:
        -----------
            local_coordinates:torch.Tensor [...,n_dim]
                n_dim = 2 for triangle
                the local coordinates of the quadrature points
        Returns:
        --------
            grad_phi: torch.Tensor of shape [..., n_basis, n_dim]
                the gradient of the base functions
    """
    pass


def shape_jac_p1(local_coordinates):
    """
        Parameters:
        -----------
            local_coordinates:torch.Tensor [...,n_dim]
                n_dim = 2 for triangle
                the local coordinates of the quadrature points
        Returns:
        --------
            jac: torch.Tensor of shape [..., n_dim, n_dim]
                the jacobian of the base functions
    """
    pass



def _bernstein_poly(i, n, t):
    """
        Bernstein polynomials
        $$\left(\frac{n}{i}\right)t^i (1-t)^(n-i)$$
        Parameters:
        -----------
            i: int
                the index of the polynomial
            n: int
                the order of the polynomial
            t: torch.Tensor of shape [...]
                the input
        Returns:
        --------
            poly: torch.Tensor of shape [...]
                the output
    """
    return scipy.special.comb(n, i) * t**i * (1 - t)**(n - i)

def _bernstein_poly_grad(i, n, t):
    if i == 0:
        return -n * (1 - t)**(n - 1)
    elif i == n:
        return n * t**(n - 1)
    else:
        return scipy.special.comb(n, i) * ((n - i) * t**(n - i - 1) * (1 - t)**i - i * t**(n - i) * (1 - t)**(i - 1))


def shape_fn(local_coordinates, order:int=1):
    """
        Parameters:
        -----------
            local_coordinates:torch.Tensor [...,n_dim]
                n_dim = 2 for triangle
                the local coordinates of the quadrature points
            order: int
                the order of the base functions
        Returns:
        --------
            phi: torch.Tensor of shape [..., n_basis]
                the base functions
    """
    n_dim = local_coordinates.shape[-1]
    assert n_dim == 2, f"traingle elements are only defined in 2D, but got {n_dim}D"
    
    
    ones  = torch.ones_like(local_coordinates[...,0:1]).to(local_coordinates.device)

    p     = torch.cat([local_coordinates, ones], dim=-1) # [..., 3]
    N_P1  = torch.einsum("ij,...j->...i", M.type(p.dtype).to(p.device),p) # [..., 2]

    l1, l2, l3 = N_P1[...,0], N_P1[...,1], N_P1[...,2]
    shape_funcs = []
    for i in range(order+1):
        for j in range(order+1 - i):
            k = order - i - j
            B = _bernstein_poly(i, order, l1) * _bernstein_poly(j, order - i, l2) * _bernstein_poly(k, order - i - j, l3)
            shape_funcs.append(B)
    
    N_Pn =  torch.stack(shape_funcs, dim=-1) # [..., n_basis]
   
    return N_Pn


def shape_fn_grad(local_coordinates, order=1):
    """
        Parameters:
        -----------
            local_coordinates:torch.Tensor [...,n_dim]
                n_dim = 2 for triangle
                the local coordinates of the quadrature points
            order: int
                the order of the base functions
        Returns:
        --------
            grad_phi: torch.Tensor of shape [..., n_basis, n_dim]
                the gradient of the base functions
    """
    n_dim = local_coordinates.shape[-1]
    assert n_dim == 2, f"Triangle elements are only defined in 2D, but got {n_dim}D"
    ones  = torch.ones_like(local_coordinates[..., 0:1])
    p     = torch.cat([local_coordinates, ones], dim=-1)  # [..., 3]
    N_P1  = torch.matmul(M, p)  # [..., 3]

    l1, l2, l3 = N_P1[..., 0], N_P1[..., 1], N_P1[..., 2]
    shape_grads = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            k = order - i - j
            B_grad_l1 = _bernstein_poly_grad(i, order, l1) * _bernstein_poly(j, order - i, l2) * _bernstein_poly(k, order - i - j, l3)
            B_grad_l2 = _bernstein_poly(i, order, l1) * _bernstein_poly_grad(j, order - i, l2) * _bernstein_poly(k, order - i - j, l3)
            B_grad_l3 = _bernstein_poly(i, order, l1) * _bernstein_poly(j, order - i, l2) * _bernstein_poly_grad(k, order - i - j, l3)
            shape_grad = torch.stack([B_grad_l1, B_grad_l2, B_grad_l3], dim=-1)
            shape_grads.append(shape_grad)
    
    N_Pn_grad = torch.stack(shape_grads, dim=-2)  # [..., n_basis, n_dim]
    return N_Pn_grad


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    
    n = 40
    order = 2
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)

    mask = X + Y <= 1
    x_inner = X[mask]
    y_inner = Y[mask]
    phi = shape_fn(torch.from_numpy(np.stack([x_inner, y_inner],-1)), order).numpy() # [n*(n+1)//2, 3]
    n_basis = phi.shape[-1]

    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = fig.gca()
    for basis_idx  in range(n_basis):
        phi_basis = np.zeros_like(X)
        phi_basis[mask] = phi[:,basis_idx]
        ax.plot_surface(X, Y, phi_basis,
                        linewidth=0, rstride=4, cstride=4, alpha=0.5)

    plt.show()
    
    # import numpy as np
    # import torch
    # import matplotlib.pyplot as plt

    # # Create the grid
    # n = 40
    # order = 2
    # x = np.linspace(0, 1, n)
    # y = np.linspace(0, 1, n)
    # X, Y = np.meshgrid(x, y)

    # # Apply the mask
    # mask = X + Y <= 1
    # x_inner = X[mask]
    # y_inner = Y[mask]

    # # Calculate the shape function values
    # phi = shape_fn(torch.from_numpy(np.stack([x_inner, y_inner],-1)), order).numpy()  # [n*(n+1)//2, 3]
    # n_basis = phi.shape[-1]

    # # Calculate the number of rows and columns for the subplots
    # n_cols = int(np.ceil(np.sqrt(n_basis)))
    # n_rows = int(np.ceil(n_basis / n_cols))

    # # Create the subplots
    # fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))

    # # Flatten the axes array for easy indexing
    # axes = np.array(axes).reshape(-1)

    # # Plot each basis function
    # for i in range(n_basis):
    #     ax = axes[i]
    #     phi_basis = np.zeros_like(X)
    #     phi_basis[mask] = phi[:, i]
    #     c = ax.pcolormesh(X, Y, phi_basis, shading='auto')
    #     ax.set_title(f'Basis {i}')
    #     plt.colorbar(c, ax=ax)

    # # Hide any remaining empty subplots
    # for i in range(n_basis, n_rows * n_cols):
    #     axes[i].axis('off')

    # plt.tight_layout()
    # plt.show()

