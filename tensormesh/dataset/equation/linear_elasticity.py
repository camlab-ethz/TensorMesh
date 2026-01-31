"""
Linear Elasticity Analytical Solutions for Dataset Generation.

This module provides analytical solutions for linear elasticity problems
that can be used to generate training/testing datasets for machine learning.

The linear elasticity equation (Navier-Cauchy equations):
    -∇·σ = f  in Ω
    u = 0     on ∂Ω (Dirichlet boundary condition)

where:
    σ = λ tr(ε) I + 2μ ε  (Hooke's law for isotropic materials)
    ε = (∇u + (∇u)^T) / 2  (infinitesimal strain tensor)
    λ, μ are Lamé parameters
"""

import torch
from typing import Optional, Union, Tuple


class LinearElasticityMultiFrequency:
    r"""
    Multi-frequency linear elasticity equation in 2D, with zero boundary condition.

    Displacement field (satisfies zero Dirichlet BC on [0,1]^2):
    
    .. math::

        u_x(x, y) = \frac{1}{K^2} \sum_{i,j=1}^{K} a^x_{ij} \cdot (i^2 + j^2)^{-r} \sin(\pi ix) \sin(\pi jy)
        
        u_y(x, y) = \frac{1}{K^2} \sum_{i,j=1}^{K} a^y_{ij} \cdot (i^2 + j^2)^{-r} \sin(\pi ix) \sin(\pi jy)

    The corresponding body force is computed via:
    
    .. math::

        \mathbf{f} = -\nabla \cdot \boldsymbol{\sigma} = -(\lambda + \mu) \nabla(\nabla \cdot \mathbf{u}) - \mu \nabla^2 \mathbf{u}

    Parameters
    ----------
    a_x : torch.Tensor, optional
        Coefficients for x-displacement. Shape: [K, K] or [N, K, K]
    a_y : torch.Tensor, optional
        Coefficients for y-displacement. Shape: [K, K] or [N, K, K]
    K : int, optional
        Frequency dimension. Default is 2.
    r : float, optional
        Decay exponent for high frequencies. Default is 0.5.
    E : float, optional
        Young's modulus. Default is 1.0.
    nu : float, optional
        Poisson's ratio. Default is 0.3.
    """
    
    def __init__(
        self, 
        a_x: Optional[torch.Tensor] = None,
        a_y: Optional[torch.Tensor] = None,
        K: int = 2,
        r: float = 0.5,
        E: float = 1.0,
        nu: float = 0.3
    ):
        if a_x is None:
            assert K is not None, "K should be specified if a_x is None"
            a_x = torch.zeros((K, K)).uniform_(-1, 1)
        else:
            K = a_x.shape[-1]
            
        if a_y is None:
            a_y = torch.zeros_like(a_x).uniform_(-1, 1)
        else:
            assert a_y.shape == a_x.shape, f"a_y shape {a_y.shape} must match a_x shape {a_x.shape}"
            
        self.K = K
        self.a_x = a_x
        self.a_y = a_y
        self.r = r
        self.E = E
        self.nu = nu
        
        # Compute Lamé parameters
        self.mu = E / (2 * (1 + nu))
        self.lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    
    def _compute_sin_basis(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute sin basis functions and decay weights."""
        K = self.K
        device = points.device
        dtype = points.dtype
        pi = torch.pi
        
        k_idx = torch.arange(1, K + 1, device=device, dtype=dtype)
        i, j = torch.meshgrid(k_idx, k_idx, indexing="ij")  # [K, K]
        w = (i * i + j * j) ** (-self.r)  # [K, K]
        
        x = points[:, 0]  # [n_points]
        y = points[:, 1]  # [n_points]
        
        sinx = torch.sin(pi * x[:, None] * k_idx[None, :])  # [n_points, K]
        siny = torch.sin(pi * y[:, None] * k_idx[None, :])  # [n_points, K]
        cosx = torch.cos(pi * x[:, None] * k_idx[None, :])  # [n_points, K]
        cosy = torch.cos(pi * y[:, None] * k_idx[None, :])  # [n_points, K]
        
        return sinx, siny, cosx, cosy, w, k_idx
    
    def solution(self, points: torch.Tensor) -> torch.Tensor:
        r"""
        Generate the displacement field at each point.
        
        .. math::
        
            u_x = \frac{1}{K^2} \sum_{i,j} a^x_{ij} (i^2+j^2)^{-r} \sin(\pi i x)\sin(\pi j y)
            
            u_y = \frac{1}{K^2} \sum_{i,j} a^y_{ij} (i^2+j^2)^{-r} \sin(\pi i x)\sin(\pi j y)
        
        Parameters
        ----------
        points : torch.Tensor
            2D tensor of shape [n_points, 2], all points in [0,1]^2
            
        Returns
        -------
        torch.Tensor
            Displacement field of shape [n_points, 2] or [N, n_points, 2]
        """
        assert points.shape[-1] == 2, f"points must have shape [n_points, 2], got {points.shape}"
        assert ((points <= 1) & (points >= 0)).all(), "points must be in [0,1]^2"
        
        K = self.K
        device = points.device
        dtype = points.dtype
        
        sinx, siny, _, _, w, _ = self._compute_sin_basis(points)
        
        if len(self.a_x.shape) == 2:
            # Non-batched case
            Bx = self.a_x.to(device=device, dtype=dtype) * w  # [K, K]
            By = self.a_y.to(device=device, dtype=dtype) * w  # [K, K]
            
            Px = sinx @ Bx  # [n_points, K]
            Py = sinx @ By  # [n_points, K]
            
            ux = (Px * siny).sum(dim=-1)  # [n_points]
            uy = (Py * siny).sum(dim=-1)  # [n_points]
            
            u = torch.stack([ux, uy], dim=-1)  # [n_points, 2]
        else:
            # Batched case
            Bx = self.a_x.to(device=device, dtype=dtype) * w  # [N, K, K]
            By = self.a_y.to(device=device, dtype=dtype) * w  # [N, K, K]
            N = Bx.shape[0]
            
            # Memory-efficient chunked computation
            bytes_per = torch.finfo(dtype).bits // 8
            n_pts = sinx.shape[0]
            target_bytes = 512 * 1024 * 1024
            chunk = max(1, target_bytes // max(1, n_pts * K * bytes_per))
            
            out_x, out_y = [], []
            for s in range(0, N, chunk):
                Bxe = Bx[s:s+chunk]  # [c, K, K]
                Bye = By[s:s+chunk]
                Pxe = torch.matmul(sinx, Bxe)  # [c, n_points, K]
                Pye = torch.matmul(sinx, Bye)
                out_x.append((Pxe * siny[None, :, :]).sum(dim=-1))  # [c, n_points]
                out_y.append((Pye * siny[None, :, :]).sum(dim=-1))
            
            ux = torch.cat(out_x, dim=0)  # [N, n_points]
            uy = torch.cat(out_y, dim=0)  # [N, n_points]
            u = torch.stack([ux, uy], dim=-1)  # [N, n_points, 2]
        
        u = u / (K * K)
        return u
    
    def body_force(self, points: torch.Tensor) -> torch.Tensor:
        r"""
        Generate the body force that produces the displacement field.
        
        For linear elasticity with displacement u:
        
        .. math::
        
            f = -\nabla \cdot \sigma = -(\lambda + \mu) \nabla(\nabla \cdot u) - \mu \nabla^2 u
        
        For our sinusoidal basis:
        
        .. math::
        
            \nabla^2 u_x = -\pi^2(i^2 + j^2) \cdot u_x
            
            \nabla \cdot u = \frac{\partial u_x}{\partial x} + \frac{\partial u_y}{\partial y}
        
        Parameters
        ----------
        points : torch.Tensor
            2D tensor of shape [n_points, 2], all points in [0,1]^2
            
        Returns
        -------
        torch.Tensor
            Body force of shape [n_points, 2] or [N, n_points, 2]
        """
        assert points.shape[-1] == 2, f"points must have shape [n_points, 2], got {points.shape}"
        assert ((points <= 1) & (points >= 0)).all(), "points must be in [0,1]^2"
        
        K = self.K
        device = points.device
        dtype = points.dtype
        pi = torch.pi
        
        sinx, siny, cosx, cosy, w, k_idx = self._compute_sin_basis(points)
        
        # For the body force, we need:
        # f_x = -(\lambda + \mu) * d/dx(div u) - \mu * laplacian(u_x)
        # f_y = -(\lambda + \mu) * d/dy(div u) - \mu * laplacian(u_y)
        
        # For u_x = A sin(pi*i*x) sin(pi*j*y):
        #   d u_x / dx = pi*i * A * cos(pi*i*x) sin(pi*j*y)
        #   d u_x / dy = pi*j * A * sin(pi*i*x) cos(pi*j*y)
        #   laplacian(u_x) = -pi^2 * (i^2 + j^2) * A * sin(pi*i*x) sin(pi*j*y)
        
        lam, mu = self.lam, self.mu
        i, j = torch.meshgrid(k_idx, k_idx, indexing="ij")  # [K, K]
        
        # Laplacian multiplier: -pi^2 * (i^2 + j^2)
        lap_mult = -pi * pi * (i * i + j * j)  # [K, K]
        
        if len(self.a_x.shape) == 2:
            ax = self.a_x.to(device=device, dtype=dtype) * w  # [K, K]
            ay = self.a_y.to(device=device, dtype=dtype) * w
            
            # Laplacian terms: -mu * laplacian(u)
            # lap(u_x) * sin(pi*i*x) * sin(pi*j*y)
            lap_ux_coeff = ax * lap_mult  # [K, K]
            lap_uy_coeff = ay * lap_mult
            
            # -mu * lap(u_x)
            P_lap_x = sinx @ lap_ux_coeff
            term_lap_x = -mu * (P_lap_x * siny).sum(dim=-1)  # [n_points]
            
            P_lap_y = sinx @ lap_uy_coeff
            term_lap_y = -mu * (P_lap_y * siny).sum(dim=-1)
            
            # Divergence term: (\lambda + \mu) * grad(div u)
            # div u = d u_x/dx + d u_y/dy
            #       = pi * sum_ij [ax_ij * i * cos(pi*i*x) sin(pi*j*y) + ay_ij * j * sin(pi*i*x) cos(pi*j*y)]
            
            # d(div u)/dx = pi^2 * sum_ij [-ax_ij * i^2 * sin(pi*i*x) sin(pi*j*y) + ay_ij * i*j * cos(pi*i*x) cos(pi*j*y)]
            # d(div u)/dy = pi^2 * sum_ij [ax_ij * i*j * cos(pi*i*x) cos(pi*j*y) - ay_ij * j^2 * sin(pi*i*x) sin(pi*j*y)]
            
            # Term 1 for f_x: -(\lambda+\mu) * d(div u)/dx
            coeff_div_x_ss = -ax * (i * i)  # sin sin term
            coeff_div_x_cc = ay * (i * j)   # cos cos term
            
            P1 = sinx @ coeff_div_x_ss
            term1_x = (P1 * siny).sum(dim=-1)  # sin sin part
            
            P2 = cosx @ coeff_div_x_cc
            term2_x = (P2 * cosy).sum(dim=-1)  # cos cos part
            
            div_grad_x = pi * pi * (term1_x + term2_x)
            f_x = -(lam + mu) * div_grad_x + term_lap_x
            
            # Term 1 for f_y: -(\lambda+\mu) * d(div u)/dy
            coeff_div_y_cc = ax * (i * j)   # cos cos term
            coeff_div_y_ss = -ay * (j * j)  # sin sin term
            
            P3 = cosx @ coeff_div_y_cc
            term1_y = (P3 * cosy).sum(dim=-1)  # cos cos part
            
            P4 = sinx @ coeff_div_y_ss
            term2_y = (P4 * siny).sum(dim=-1)  # sin sin part
            
            div_grad_y = pi * pi * (term1_y + term2_y)
            f_y = -(lam + mu) * div_grad_y + term_lap_y
            
            f = torch.stack([f_x, f_y], dim=-1)  # [n_points, 2]
        else:
            # Batched case
            ax = self.a_x.to(device=device, dtype=dtype) * w  # [N, K, K]
            ay = self.a_y.to(device=device, dtype=dtype) * w
            N = ax.shape[0]
            
            out_fx, out_fy = [], []
            
            bytes_per = torch.finfo(dtype).bits // 8
            n_pts = sinx.shape[0]
            target_bytes = 512 * 1024 * 1024
            chunk = max(1, target_bytes // max(1, n_pts * K * K * bytes_per))
            
            for s in range(0, N, chunk):
                axe = ax[s:s+chunk]  # [c, K, K]
                aye = ay[s:s+chunk]
                
                # Laplacian terms
                lap_ux = axe * lap_mult  # [c, K, K]
                lap_uy = aye * lap_mult
                
                P_lap_x = torch.matmul(sinx, lap_ux)  # [c, n_pts, K]
                term_lap_x = -mu * (P_lap_x * siny[None, :, :]).sum(dim=-1)
                
                P_lap_y = torch.matmul(sinx, lap_uy)
                term_lap_y = -mu * (P_lap_y * siny[None, :, :]).sum(dim=-1)
                
                # Divergence gradient terms for f_x
                coeff_div_x_ss = -axe * (i * i)
                coeff_div_x_cc = aye * (i * j)
                
                P1 = torch.matmul(sinx, coeff_div_x_ss)
                term1_x = (P1 * siny[None, :, :]).sum(dim=-1)
                
                P2 = torch.matmul(cosx, coeff_div_x_cc)
                term2_x = (P2 * cosy[None, :, :]).sum(dim=-1)
                
                div_grad_x = pi * pi * (term1_x + term2_x)
                f_x = -(lam + mu) * div_grad_x + term_lap_x
                
                # Divergence gradient terms for f_y
                coeff_div_y_cc = axe * (i * j)
                coeff_div_y_ss = -aye * (j * j)
                
                P3 = torch.matmul(cosx, coeff_div_y_cc)
                term1_y = (P3 * cosy[None, :, :]).sum(dim=-1)
                
                P4 = torch.matmul(sinx, coeff_div_y_ss)
                term2_y = (P4 * siny[None, :, :]).sum(dim=-1)
                
                div_grad_y = pi * pi * (term1_y + term2_y)
                f_y = -(lam + mu) * div_grad_y + term_lap_y
                
                out_fx.append(f_x)
                out_fy.append(f_y)
            
            f_x = torch.cat(out_fx, dim=0)  # [N, n_points]
            f_y = torch.cat(out_fy, dim=0)
            f = torch.stack([f_x, f_y], dim=-1)  # [N, n_points, 2]
        
        f = f / (K * K)
        return f
    
    def strain(self, points: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the strain tensor at each point.
        
        .. math::
        
            \varepsilon_{ij} = \frac{1}{2}(\frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i})
        
        Parameters
        ----------
        points : torch.Tensor
            2D tensor of shape [n_points, 2]
            
        Returns
        -------
        torch.Tensor
            Strain tensor of shape [n_points, 2, 2] or [N, n_points, 2, 2]
        """
        assert points.shape[-1] == 2
        assert ((points <= 1) & (points >= 0)).all()
        
        K = self.K
        device = points.device
        dtype = points.dtype
        pi = torch.pi
        
        sinx, siny, cosx, cosy, w, k_idx = self._compute_sin_basis(points)
        i, j = torch.meshgrid(k_idx, k_idx, indexing="ij")
        
        if len(self.a_x.shape) == 2:
            ax = self.a_x.to(device=device, dtype=dtype) * w
            ay = self.a_y.to(device=device, dtype=dtype) * w
            
            # du_x/dx = pi * i * ax * cos(pi*i*x) * sin(pi*j*y)
            coeff_duxdx = pi * i * ax
            P1 = cosx @ coeff_duxdx
            du_x_dx = (P1 * siny).sum(dim=-1)  # [n_points]
            
            # du_x/dy = pi * j * ax * sin(pi*i*x) * cos(pi*j*y)
            coeff_duxdy = pi * j * ax
            P2 = sinx @ coeff_duxdy
            du_x_dy = (P2 * cosy).sum(dim=-1)
            
            # du_y/dx = pi * i * ay * cos(pi*i*x) * sin(pi*j*y)
            coeff_duydx = pi * i * ay
            P3 = cosx @ coeff_duydx
            du_y_dx = (P3 * siny).sum(dim=-1)
            
            # du_y/dy = pi * j * ay * sin(pi*i*x) * cos(pi*j*y)
            coeff_duydy = pi * j * ay
            P4 = sinx @ coeff_duydy
            du_y_dy = (P4 * cosy).sum(dim=-1)
            
            # Strain tensor
            eps_xx = du_x_dx
            eps_yy = du_y_dy
            eps_xy = 0.5 * (du_x_dy + du_y_dx)
            
            strain = torch.stack([
                torch.stack([eps_xx, eps_xy], dim=-1),
                torch.stack([eps_xy, eps_yy], dim=-1)
            ], dim=-2)  # [n_points, 2, 2]
        else:
            # Batched case - similar pattern
            ax = self.a_x.to(device=device, dtype=dtype) * w
            ay = self.a_y.to(device=device, dtype=dtype) * w
            
            coeff_duxdx = pi * i * ax  # [N, K, K]
            coeff_duxdy = pi * j * ax
            coeff_duydx = pi * i * ay
            coeff_duydy = pi * j * ay
            
            P1 = torch.matmul(cosx, coeff_duxdx)
            du_x_dx = (P1 * siny[None, :, :]).sum(dim=-1)  # [N, n_points]
            
            P2 = torch.matmul(sinx, coeff_duxdy)
            du_x_dy = (P2 * cosy[None, :, :]).sum(dim=-1)
            
            P3 = torch.matmul(cosx, coeff_duydx)
            du_y_dx = (P3 * siny[None, :, :]).sum(dim=-1)
            
            P4 = torch.matmul(sinx, coeff_duydy)
            du_y_dy = (P4 * cosy[None, :, :]).sum(dim=-1)
            
            eps_xx = du_x_dx
            eps_yy = du_y_dy
            eps_xy = 0.5 * (du_x_dy + du_y_dx)
            
            strain = torch.stack([
                torch.stack([eps_xx, eps_xy], dim=-1),
                torch.stack([eps_xy, eps_yy], dim=-1)
            ], dim=-2)  # [N, n_points, 2, 2]
        
        strain = strain / (K * K)
        return strain
    
    def stress(self, points: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the stress tensor at each point using Hooke's law.
        
        .. math::
        
            \sigma_{ij} = \lambda \text{tr}(\varepsilon) \delta_{ij} + 2\mu \varepsilon_{ij}
        
        Parameters
        ----------
        points : torch.Tensor
            2D tensor of shape [n_points, 2]
            
        Returns
        -------
        torch.Tensor
            Stress tensor of shape [n_points, 2, 2] or [N, n_points, 2, 2]
        """
        eps = self.strain(points)
        
        # trace(eps)
        tr_eps = eps[..., 0, 0] + eps[..., 1, 1]  # [..., n_points] or [..., N, n_points]
        
        # Identity tensor
        I = torch.eye(2, device=points.device, dtype=points.dtype)
        
        # Expand tr_eps for broadcasting
        if eps.dim() == 3:
            # [n_points, 2, 2]
            sigma = self.lam * tr_eps[:, None, None] * I + 2 * self.mu * eps
        else:
            # [N, n_points, 2, 2]
            sigma = self.lam * tr_eps[:, :, None, None] * I + 2 * self.mu * eps
        
        return sigma


class LinearElasticityMultiFrequency3D:
    r"""
    Multi-frequency linear elasticity equation in 3D, with zero boundary condition.

    Displacement field (satisfies zero Dirichlet BC on [0,1]^3):
    
    .. math::

        u_\alpha(x, y, z) = \frac{1}{K^3} \sum_{i,j,k=1}^{K} a^\alpha_{ijk} \cdot (i^2 + j^2 + k^2)^{-r} 
        \sin(\pi ix) \sin(\pi jy) \sin(\pi kz)

    for α ∈ {x, y, z}.

    Parameters
    ----------
    a_x, a_y, a_z : torch.Tensor, optional
        Coefficients for each displacement component. Shape: [K, K, K] or [N, K, K, K]
    K : int, optional
        Frequency dimension. Default is 2.
    r : float, optional
        Decay exponent. Default is 0.5.
    E : float, optional
        Young's modulus. Default is 1.0.
    nu : float, optional
        Poisson's ratio. Default is 0.3.
    """
    
    def __init__(
        self,
        a_x: Optional[torch.Tensor] = None,
        a_y: Optional[torch.Tensor] = None,
        a_z: Optional[torch.Tensor] = None,
        K: int = 2,
        r: float = 0.5,
        E: float = 1.0,
        nu: float = 0.3
    ):
        if a_x is None:
            assert K is not None
            a_x = torch.zeros((K, K, K)).uniform_(-1, 1)
        else:
            K = a_x.shape[-1]
            
        if a_y is None:
            a_y = torch.zeros_like(a_x).uniform_(-1, 1)
        if a_z is None:
            a_z = torch.zeros_like(a_x).uniform_(-1, 1)
            
        self.K = K
        self.a_x = a_x
        self.a_y = a_y
        self.a_z = a_z
        self.r = r
        self.E = E
        self.nu = nu
        
        self.mu = E / (2 * (1 + nu))
        self.lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    
    def solution(self, points: torch.Tensor) -> torch.Tensor:
        r"""
        Generate the displacement field at each point.
        
        Parameters
        ----------
        points : torch.Tensor
            2D tensor of shape [n_points, 3], all points in [0,1]^3
            
        Returns
        -------
        torch.Tensor
            Displacement field of shape [n_points, 3] or [N, n_points, 3]
        """
        assert points.shape[-1] == 3
        assert ((points <= 1) & (points >= 0)).all()
        
        K = self.K
        device = points.device
        dtype = points.dtype
        pi = torch.pi
        
        k_idx = torch.arange(1, K + 1, device=device, dtype=dtype)
        i, j, k = torch.meshgrid(k_idx, k_idx, k_idx, indexing="ij")
        w = (i * i + j * j + k * k) ** (-self.r)
        
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        sinx = torch.sin(pi * x[:, None] * k_idx[None, :])
        siny = torch.sin(pi * y[:, None] * k_idx[None, :])
        sinz = torch.sin(pi * z[:, None] * k_idx[None, :])
        
        if len(self.a_x.shape) == 3:
            # Non-batched
            Bx = self.a_x.to(device=device, dtype=dtype) * w
            By = self.a_y.to(device=device, dtype=dtype) * w
            Bz = self.a_z.to(device=device, dtype=dtype) * w
            
            ux = torch.einsum('pi,pj,pk,ijk->p', sinx, siny, sinz, Bx)
            uy = torch.einsum('pi,pj,pk,ijk->p', sinx, siny, sinz, By)
            uz = torch.einsum('pi,pj,pk,ijk->p', sinx, siny, sinz, Bz)
            
            u = torch.stack([ux, uy, uz], dim=-1)
        else:
            # Batched
            Bx = self.a_x.to(device=device, dtype=dtype) * w
            By = self.a_y.to(device=device, dtype=dtype) * w
            Bz = self.a_z.to(device=device, dtype=dtype) * w
            N = Bx.shape[0]
            
            n_pts = sinx.shape[0]
            bytes_per = torch.finfo(dtype).bits // 8
            target_bytes = 512 * 1024 * 1024
            chunk = max(1, target_bytes // max(1, n_pts * K * K * K * bytes_per))
            
            out_x, out_y, out_z = [], [], []
            for s in range(0, N, chunk):
                Bxe = Bx[s:s+chunk]
                Bye = By[s:s+chunk]
                Bze = Bz[s:s+chunk]
                
                uxe = torch.einsum('pi,pj,pk,nijk->np', sinx, siny, sinz, Bxe)
                uye = torch.einsum('pi,pj,pk,nijk->np', sinx, siny, sinz, Bye)
                uze = torch.einsum('pi,pj,pk,nijk->np', sinx, siny, sinz, Bze)
                
                out_x.append(uxe)
                out_y.append(uye)
                out_z.append(uze)
            
            ux = torch.cat(out_x, dim=0)
            uy = torch.cat(out_y, dim=0)
            uz = torch.cat(out_z, dim=0)
            u = torch.stack([ux, uy, uz], dim=-1)
        
        u = u / (K ** 3)
        return u
    
    def body_force(self, points: torch.Tensor) -> torch.Tensor:
        r"""
        Generate the body force that produces the displacement field.
        
        Parameters
        ----------
        points : torch.Tensor
            2D tensor of shape [n_points, 3], all points in [0,1]^3
            
        Returns
        -------
        torch.Tensor
            Body force of shape [n_points, 3] or [N, n_points, 3]
        """
        assert points.shape[-1] == 3
        assert ((points <= 1) & (points >= 0)).all()
        
        K = self.K
        device = points.device
        dtype = points.dtype
        pi = torch.pi
        lam, mu = self.lam, self.mu
        
        k_idx = torch.arange(1, K + 1, device=device, dtype=dtype)
        ii, jj, kk = torch.meshgrid(k_idx, k_idx, k_idx, indexing="ij")
        w = (ii * ii + jj * jj + kk * kk) ** (-self.r)
        lap_mult = -pi * pi * (ii * ii + jj * jj + kk * kk)
        
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        sinx = torch.sin(pi * x[:, None] * k_idx[None, :])
        siny = torch.sin(pi * y[:, None] * k_idx[None, :])
        sinz = torch.sin(pi * z[:, None] * k_idx[None, :])
        cosx = torch.cos(pi * x[:, None] * k_idx[None, :])
        cosy = torch.cos(pi * y[:, None] * k_idx[None, :])
        cosz = torch.cos(pi * z[:, None] * k_idx[None, :])
        
        if len(self.a_x.shape) == 3:
            ax = self.a_x.to(device=device, dtype=dtype) * w
            ay = self.a_y.to(device=device, dtype=dtype) * w
            az = self.a_z.to(device=device, dtype=dtype) * w
            
            # Laplacian terms: -mu * laplacian(u)
            lap_ux = torch.einsum('pi,pj,pk,ijk->p', sinx, siny, sinz, ax * lap_mult)
            lap_uy = torch.einsum('pi,pj,pk,ijk->p', sinx, siny, sinz, ay * lap_mult)
            lap_uz = torch.einsum('pi,pj,pk,ijk->p', sinx, siny, sinz, az * lap_mult)
            
            term_lap_x = -mu * lap_ux
            term_lap_y = -mu * lap_uy
            term_lap_z = -mu * lap_uz
            
            # div(u) = d(ux)/dx + d(uy)/dy + d(uz)/dz
            # d(div u)/dx involves:
            # -i^2 ax sss + i*j ay ccs + i*k az csc
            
            # For f_x: -(\lambda+\mu) * d(div u)/dx
            coeff_sss = -ax * (ii * ii)
            coeff_ccs = ay * (ii * jj)
            coeff_csc = az * (ii * kk)
            
            t1 = torch.einsum('pi,pj,pk,ijk->p', sinx, siny, sinz, coeff_sss)
            t2 = torch.einsum('pi,pj,pk,ijk->p', cosx, cosy, sinz, coeff_ccs)
            t3 = torch.einsum('pi,pj,pk,ijk->p', cosx, siny, cosz, coeff_csc)
            
            div_grad_x = pi * pi * (t1 + t2 + t3)
            f_x = -(lam + mu) * div_grad_x + term_lap_x
            
            # For f_y: -(\lambda+\mu) * d(div u)/dy
            coeff_ccs_y = ax * (ii * jj)
            coeff_sss_y = -ay * (jj * jj)
            coeff_scs_y = az * (jj * kk)
            
            t1 = torch.einsum('pi,pj,pk,ijk->p', cosx, cosy, sinz, coeff_ccs_y)
            t2 = torch.einsum('pi,pj,pk,ijk->p', sinx, siny, sinz, coeff_sss_y)
            t3 = torch.einsum('pi,pj,pk,ijk->p', sinx, cosy, cosz, coeff_scs_y)
            
            div_grad_y = pi * pi * (t1 + t2 + t3)
            f_y = -(lam + mu) * div_grad_y + term_lap_y
            
            # For f_z: -(\lambda+\mu) * d(div u)/dz
            coeff_csc_z = ax * (ii * kk)
            coeff_scs_z = ay * (jj * kk)
            coeff_sss_z = -az * (kk * kk)
            
            t1 = torch.einsum('pi,pj,pk,ijk->p', cosx, siny, cosz, coeff_csc_z)
            t2 = torch.einsum('pi,pj,pk,ijk->p', sinx, cosy, cosz, coeff_scs_z)
            t3 = torch.einsum('pi,pj,pk,ijk->p', sinx, siny, sinz, coeff_sss_z)
            
            div_grad_z = pi * pi * (t1 + t2 + t3)
            f_z = -(lam + mu) * div_grad_z + term_lap_z
            
            f = torch.stack([f_x, f_y, f_z], dim=-1)
        else:
            # Batched case
            ax = self.a_x.to(device=device, dtype=dtype) * w
            ay = self.a_y.to(device=device, dtype=dtype) * w
            az = self.a_z.to(device=device, dtype=dtype) * w
            N = ax.shape[0]
            
            out_fx, out_fy, out_fz = [], [], []
            
            n_pts = sinx.shape[0]
            bytes_per = torch.finfo(dtype).bits // 8
            target_bytes = 512 * 1024 * 1024
            chunk = max(1, target_bytes // max(1, n_pts * K * K * K * bytes_per))
            
            for s in range(0, N, chunk):
                axe = ax[s:s+chunk]
                aye = ay[s:s+chunk]
                aze = az[s:s+chunk]
                
                # Laplacian terms
                lap_ux = torch.einsum('pi,pj,pk,nijk->np', sinx, siny, sinz, axe * lap_mult)
                lap_uy = torch.einsum('pi,pj,pk,nijk->np', sinx, siny, sinz, aye * lap_mult)
                lap_uz = torch.einsum('pi,pj,pk,nijk->np', sinx, siny, sinz, aze * lap_mult)
                
                term_lap_x = -mu * lap_ux
                term_lap_y = -mu * lap_uy
                term_lap_z = -mu * lap_uz
                
                # f_x
                coeff_sss = -axe * (ii * ii)
                coeff_ccs = aye * (ii * jj)
                coeff_csc = aze * (ii * kk)
                
                t1 = torch.einsum('pi,pj,pk,nijk->np', sinx, siny, sinz, coeff_sss)
                t2 = torch.einsum('pi,pj,pk,nijk->np', cosx, cosy, sinz, coeff_ccs)
                t3 = torch.einsum('pi,pj,pk,nijk->np', cosx, siny, cosz, coeff_csc)
                
                div_grad_x = pi * pi * (t1 + t2 + t3)
                f_x = -(lam + mu) * div_grad_x + term_lap_x
                
                # f_y
                coeff_ccs_y = axe * (ii * jj)
                coeff_sss_y = -aye * (jj * jj)
                coeff_scs_y = aze * (jj * kk)
                
                t1 = torch.einsum('pi,pj,pk,nijk->np', cosx, cosy, sinz, coeff_ccs_y)
                t2 = torch.einsum('pi,pj,pk,nijk->np', sinx, siny, sinz, coeff_sss_y)
                t3 = torch.einsum('pi,pj,pk,nijk->np', sinx, cosy, cosz, coeff_scs_y)
                
                div_grad_y = pi * pi * (t1 + t2 + t3)
                f_y = -(lam + mu) * div_grad_y + term_lap_y
                
                # f_z
                coeff_csc_z = axe * (ii * kk)
                coeff_scs_z = aye * (jj * kk)
                coeff_sss_z = -aze * (kk * kk)
                
                t1 = torch.einsum('pi,pj,pk,nijk->np', cosx, siny, cosz, coeff_csc_z)
                t2 = torch.einsum('pi,pj,pk,nijk->np', sinx, cosy, cosz, coeff_scs_z)
                t3 = torch.einsum('pi,pj,pk,nijk->np', sinx, siny, sinz, coeff_sss_z)
                
                div_grad_z = pi * pi * (t1 + t2 + t3)
                f_z = -(lam + mu) * div_grad_z + term_lap_z
                
                out_fx.append(f_x)
                out_fy.append(f_y)
                out_fz.append(f_z)
            
            f_x = torch.cat(out_fx, dim=0)
            f_y = torch.cat(out_fy, dim=0)
            f_z = torch.cat(out_fz, dim=0)
            f = torch.stack([f_x, f_y, f_z], dim=-1)
        
        f = f / (K ** 3)
        return f

