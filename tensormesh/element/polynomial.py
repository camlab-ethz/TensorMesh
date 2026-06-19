"""Low-level multivariate polynomial primitives used to build the element
shape functions.

:class:`Polynomial` represents a single multivariate polynomial; :class:`Polynomials`
represents an arbitrarily-shaped batch of them (a "tensor of polynomials"). Both
extend :class:`torch.nn.Module` so coefficients and exponents are buffers that
follow the usual ``.to()`` / ``.cuda()`` / ``.double()`` dance.

This module is exposed as :mod:`tensormesh.element` for advanced users who need
to define new element types; most code should subclass
:class:`~tensormesh.Element` and override its basis / quadrature hooks rather
than calling these classes directly.
"""
import math
from functools import lru_cache, reduce
from typing import List, Tuple, Optional, Type, Sequence, Union

import torch
import torch.nn as nn



class Polynomial(nn.Module):
    r"""
    
    A polynomial class representing multivariate polynomials of the form:

    .. math::

        p(x_1,\ldots,x_n) = \sum_{i=1}^m c_i \prod_{j=1}^n x_j^{e_{ij}}

    where:
    
    - :math:`n` is the number of variables (n_vars)
    - :math:`m` is the number of terms (n_terms) 
    - :math:`c_i` are the coefficients (coef)
    - :math:`e_{ij}` are the exponents (exp)
    - :math:`x_j` are the variables

    For example, the polynomial :math:`x^2y + 2xy^2 + 3` has:

    - n_vars = 2 (x, y)
    - n_terms = 3
    - coef = [1, 2, 3]
    - exp = [[2, 1, 0], [1, 2, 0]]  (rows index variables, columns index terms)
    
    Examples
    --------
    .. code-block:: python

        # Create polynomial with exponents and coefficients
        exp = torch.tensor([[1, 2, 3], [2, 1, 0]]) # [n_vars(2), n_terms(3)]
        coef = torch.tensor([1, 1, 1])             # [n_terms(3)]
        poly = Polynomial(exp, coef)
        
        # Print polynomial
        print(poly)  # xy^2 + x^2y + x^3
        
        # Evaluate polynomial at point x
        x = torch.tensor([2, 3]) # [n_vars]
        print(poly(x))  # 2*3^2 + 2^2*3 + 2^3 = 38
        
        # Take derivatives
        print(poly.deriv(0))  # y^2 + 2xy + 3x^2
        print(poly.deriv(1))  # 2xy + x^2
    """

    _coef:torch.Tensor 
    """Coefficients tensor of shape :math:`[N_t]` where:
    
    * :math:`N_t` = number of terms in polynomial
    """

    _exp:torch.Tensor  
    """Exponents tensor of shape :math:`[N_v, N_t]` where:
    
    * :math:`N_v` = number of variables
    * :math:`N_t` = number of terms in polynomial
    """

    n_vars:int
    """Number of variables in polynomial
    :no-index:
    """

    n_terms:int 
    """Number of terms in polynomial
    :no-index:
    """

    def __init__(self, 
                 exp:torch.Tensor, 
                 coef:Optional[torch.Tensor] = None):
        """Initialize a polynomial with exponents and coefficients.

        Parameters
        ----------
        exp : torch.Tensor
            Exponents tensor of shape :math:`[N_v, N_t]` where:
            
            * :math:`N_v` = number of variables
            * :math:`N_t` = number of terms
        coef : Optional[torch.Tensor], optional
            Coefficients tensor of shape :math:`[N_t]`, by default None.
            If None, coefficients will be initialized as ones.

        Examples
        --------
        .. code-block:: python

            # Create polynomial x^2y + 2xy^2 + 3
            exp = torch.tensor([[2,1,0], [1,2,0]])  # [2 vars, 3 terms]
            coef = torch.tensor([1,2,3])            # [3 terms]
            poly = Polynomial(exp, coef)

        """
        super().__init__()

        assert exp.dim() == 2, f"exp should be of shape [n_vars, n_terms], but got exp of shape {exp.shape}"
        if coef is None:
            coef = torch.ones(exp.shape[1],dtype=exp.dtype, device=exp.device)
        assert (coef.dim() == 1 and 
                coef.shape[0] == exp.shape[1]), (f"coef should be of shape [n_terms], exp should be of shape [n_vars, n_terms], "
                                               f"but got coef of shape {coef.shape}, and exp of shape {exp.shape}")
        assert exp.dtype == coef.dtype, f"exp and coef should have the same dtype, but got {exp.dtype} and {coef.dtype}"
        assert exp.device == coef.device, f"exp and coef should have the same device, but got {exp.device} and {coef.device}"
        self.register_buffer('_coef', coef)
        self.register_buffer('_exp', exp)
        self.n_vars  = exp.shape[0]
        self.n_terms = exp.shape[1]

    @property
    def device(self):
        return self._coef.device

    @property
    def dtype(self):
        return self._coef.dtype

    def _apply(self, fn, recurse: bool = True):
        # ``nn.Module._apply`` is the hook ``.to(dtype)`` / ``.float()`` /
        # ``.cfloat()`` ultimately call. Polynomials describe basis-function
        # forms (real-valued coefficients/exponents); complex behaviour in
        # the assembler enters through external ``point_data`` /
        # ``element_data`` coefficients, NOT the basis itself. Casting
        # ``_exp`` to complex also triggers ``torch.pow(0, complex(0))``
        # producing NaN at quadrature node 0 of every reference element
        # — see ROADMAP item 2. So strip complex requests here.
        def _real_only(t):
            out = fn(t)
            if out.is_complex():
                # Map complex64 / complex128 back to the matching real dtype.
                real_dtype = (
                    torch.float32 if out.dtype == torch.complex64 else torch.float64
                )
                out = out.real.to(real_dtype)
            return out
        return super()._apply(_real_only, recurse=recurse)

    def __len__(self):
        """Get number of terms in polynomial.

        Returns
        -------
        int
            Number of terms in polynomial
        """
        return self.n_terms

    def __getitem__(self, index:Union[int,slice,torch.Tensor]
                    )->Union[Tuple[torch.Tensor, torch.Tensor],
                        'Polynomial',
                        'Polynomials']:
        """Get subset of polynomial terms.

        Parameters
        ----------
        index : Union[int, slice, torch.Tensor]
            Index to select terms:
            
            * int: single term
            * slice: range of terms
            * tensor: boolean or integer indices

        Returns
        -------
        Union[Tuple[torch.Tensor, torch.Tensor], Polynomial, Polynomials]
            * For single term: (coefficient, exponents) tuple
            * For multiple terms: new Polynomial
            * For multiple polynomials: new Polynomials
        """
        _coef = self._coef[index]
        _exp  = self._exp[:, index]
        if _coef.dim() == 0:
            return _coef, _exp
        elif _coef.dim() == 1:
            return Polynomial(_exp, _coef)
        elif _coef.dim() > 1:
            return Polynomials(_exp, _coef)
        else:
            raise NotImplementedError(f"Invalid input shape {index}")

    def __str__(self, max_show_col:int=2):
        assert self.n_vars <= 26, "The number of variables should be less than 26"
        if self.n_vars <= 26:
            VAR_NAME = lambda x: list("xyzabcdefghijklmnopqrstuvw")[x]
        else:
            VAR_NAME = lambda x: f"(x{x})"
        string = []
        for i,c in enumerate(self._coef):
            if c != 0:
                c = c.item()
                _vars = []
                for j, e in enumerate(self._exp[:,i]):
                    if e == 0.:
                        _vars.append("")
                    elif e == 1.:
                        _vars.append(f"{VAR_NAME(j)}")
                    else:
                        _vars.append(f"{VAR_NAME(j)}^{e:.2g}")
                
                if len(_vars) > 0 and c == 1.: 
                    if all([x == "" for x in _vars]):
                        string.append("1")
                    else:
                        string.append(''.join(_vars))
                elif len(_vars) > 0 and c == -1.:
                    if all([x == "" for x in _vars]):
                        string.append("-1")
                    else:
                        string.append('-' + ''.join(_vars))
                elif len(_vars) > 0 and all([x == "" for x in _vars]):
                    string.append(f"{c:.2g}")
                else:
                    string.append(f"{c:.2g}{''.join(_vars)}")
        if max_show_col is not None and len(string) > max_show_col * 2:
            string = string[:max_show_col] + ["..."] + string[-max_show_col:]
              

        if len(string) == 0:
            return "0"
        result = []
        for i, term in enumerate(string):
            if i == 0:
                result.append(term)
            else:
                if term.startswith("-"):
                    result.append(" " + term)
                else:
                    result.append(" + " + term)
        return "".join(result)
    
    def __repr__(self):
        return str(self)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """Evaluate the polynomial at given points.

        Examples
        --------
        .. code-block:: python

            # Create a polynomial p(x,y) = 2x + 3y^2
            coef = torch.tensor([2.0, 3.0])
            exp = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
            poly = Polynomial(exp, coef)
            
            # Evaluate at single point
            x = torch.tensor([1.0, 2.0])  # point (x=1, y=2)
            poly(x)  # 2*1 + 3*2^2 = 14.0
            # tensor(14.)
            
            # Evaluate at multiple points
            x = torch.tensor([[1.0, 2.0], [2.0, 1.0]])  # points (1,2) and (2,1)
            poly(x)  # [2*1 + 3*2^2, 2*2 + 3*1^2] = [14.0, 7.0]
            # tensor([14., 7.])


        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``[n_batch, n_vars]`` or ``[n_vars]``.

        Returns
        -------
        torch.Tensor
            Evaluated polynomial values of shape ``[n_batch]`` or scalar.
        """
        assert x.dtype == self.dtype, f"x and polynomial should have the same dtype, but got {x.dtype} and {self.dtype}"
        assert x.device == self.device, f"x and polynomial should have the same device, but got {x.device} and {self.device}"
        assert x.dim() == 1 or x.dim() == 2, f"x should be of shape [n_batch, n_vars] or [n_vars], but got x of shape {x.shape}"
        assert x.shape[-1] == self.n_vars, f"x should have the same number of variables as the polynomial, but got {x.shape[-1]} and {self.n_vars}"
        x = self.get_exp_terms(x) # [n_batch, n_terms] or [n_terms]
        x = x @ self._coef
        
        return x

    def get_exp_terms(self, x:torch.Tensor)->torch.Tensor:
        r"""Compute exponential terms for polynomial evaluation.

        For a polynomial with terms :math:`c_i x_1^{e_{i1}} x_2^{e_{i2}} \cdots x_n^{e_{in}}`,
        computes the exponential terms :math:`x_1^{e_{i1}} x_2^{e_{i2}} \cdots x_n^{e_{in}}` 
        for each term i.

        Examples
        --------
        .. code-block:: python

            # Create polynomial p(x,y) = 2x^2y + 3xy^2
            exp = torch.tensor([[2,1], [1,2]])  # exponents for each term
            coef = torch.tensor([2.0, 3.0])     # coefficients
            poly = Polynomial(exp, coef)

            # Single point evaluation
            x = torch.tensor([2.0, 3.0])  # point (x=2, y=3)
            terms = poly.get_exp_terms(x)  # [2^2 * 3^1, 2^1 * 3^2]
            # tensor([12., 54.])

            # Multiple point evaluation  
            x = torch.tensor([[2.0, 3.0], [1.0, 2.0]])  # points (2,3) and (1,2)
            terms = poly.get_exp_terms(x)  # [[2^2 * 3^1, 2^1 * 3^2], [1^2 * 2^1, 1^1 * 2^2]]
            # tensor([[12., 54.],
            #         [ 2.,  4.]])

        Parameters
        ----------
        x : torch.Tensor
            Input points tensor of shape:
            * [n_vars] for single point
            * [n_batch, n_vars] for multiple points
            where:
            * n_vars = number of variables
            * n_batch = number of points

        Returns
        -------
        torch.Tensor
            Exponential terms tensor of shape:
            * [n_terms] for single point input
            * [n_batch, n_terms] for multiple point input
            where:
            * n_terms = number of polynomial terms
        """
        assert x.dtype  == self.dtype, f"x and self should have the same dtype, but got {x.dtype} and {self.dtype}"
        assert x.device == self.device, f"x and self should have the same device, but got {x.device} and {self.device}"
        if x.dim() == 1:
            x = torch.pow(x[:, None], self._exp) # [n_vars, n_terms]
            x = torch.prod(x, dim=0)    # [n_terms]
        else:
            assert x.dim() == 2, f"x should be of shape [n_batch, n_vars], but got x of shape {x.shape}"
            # breakpoint()
            x = torch.pow(x[:, :, None], self._exp[None, :, :]) # [n_batch, n_vars, n_terms]
            x = torch.prod(x, dim=1)    # [n_batch, n_terms]
        return x
        
    def deriv(self, var_ind:int=0)->'Polynomial':
        r"""
        Compute the derivative of the polynomial with respect to a variable.

        For a polynomial :math:`p(x,y,z) = ax^ny^mz^k`, the derivative with respect to x is:
        :math:`\frac{\partial p}{\partial x} = nax^{n-1}y^mz^k` if n>0, or 0 if n=0
        
        Examples
        --------
        .. code-block:: python

            # Create polynomial p(x,y) = x^2y + 2xy^2 + 3
            exp = torch.tensor([[2, 1, 0], [1, 2, 0]]) # [n_vars(2), n_terms(3)]
            coef = torch.tensor([1, 2, 3])             # [n_terms(3)]
            poly = Polynomial(exp, coef)

            # Take derivative with respect to x
            dx = poly.deriv(0)  # 2xy + 2y^2
            print(dx)

            # Take derivative with respect to y  
            dy = poly.deriv(1)  # x^2 + 4xy
            print(dy)

            # Evaluate derivatives at point (2,3)
            x = torch.tensor([2.0, 3.0])
            print(dx(x))  # 2*2*3 + 2*3^2 = 30
            print(dy(x))  # 2^2 + 4*2*3 = 28


        Parameters
        ----------
        var_ind: int 
                the index of the variable to be differentiated
    
        Returns
        -------
        deriv_poly: Polynomial
                    the derivative of the polynomial
        """
        assert var_ind < self.n_vars, f"var_ind should be less than {self.n_vars}, but got {var_ind}"
        coef = self._coef.clone()
        exp  = self._exp.clone()
        where_constant = exp[var_ind] == 0
        exp[var_ind] = exp[var_ind] - 1
        exp[var_ind] = torch.clamp_min(exp[var_ind], 0.0)
        mask         = torch.ones_like(coef)
        mask[where_constant] = 0
        coef     = coef * (exp[var_ind] + 1) * mask        
        return self.__class__(exp, coef)
    
    def grad(self)->'Polynomials':
        r"""
        Compute the gradient of the polynomial.

        For a polynomial :math:`p(x_1,\ldots,x_n)`, returns a vector of partial derivatives:
        :math:`\nabla p = [\frac{\partial p}{\partial x_1}, \ldots, \frac{\partial p}{\partial x_n}]`

        For example, given :math:`p(x,y) = ax^ny^m`, the gradient is:
        :math:`\nabla p = [nax^{n-1}y^m, max^ny^{m-1}]`

        Examples
        --------
        .. code-block:: python

            # Create polynomial p(x,y) = x^2y + 2xy^2 + 3
            exp = torch.tensor([[2, 1, 0], [1, 2, 0]]) # [n_vars(2), n_terms(3)]
            coef = torch.tensor([1, 2, 3])             # [n_terms(3)]
            poly = Polynomial(exp, coef)

            # Take gradient
            grad = poly.grad()  # [2xy + 2y^2, x^2 + 4xy]
            print(grad)

            # Evaluate gradient at point (2,3)
            x = torch.tensor([2.0, 3.0])
            print(grad(x))  # [30, 28]

        Parameters
        ----------
        None

        Returns
        -------
        grad_poly: Polynomials
                    the gradient of the polynomial
        """
        # TODO: Could be more efficient
        return Polynomials.stack([self.deriv(i) for i in range(self.n_vars)])

    def reset_coef(self, coef:torch.Tensor):
        """
        Reset the coefficients of the polynomial while keeping the exponents unchanged.

        Parameters
        ----------
        coef: torch.Tensor
            The new coefficients to set. Must match the shape, device and dtype of the existing coefficients.

        Returns
        -------
        self: Polynomial
            Returns self for method chaining.
        """

        assert coef.shape == self._coef.shape, f"reset coef error: expected {self._coef.shape} but got {coef.shape}"
        assert coef.device== self.device, f"reset coef error: expected {self.device} but got {coef.device}"
        assert coef.dtype == self.dtype, f"reset coef error: expected {self.dtype} but got {coef.dtype}"

        self._coef = coef

        return self

    def repeat(self, *args)->'Polynomials':
        """
        Repeat the polynomial along specified dimensions.

        Creates a new Polynomials object by repeating the current polynomial's coefficients and 
        exponents according to the provided repeat dimensions. This is similar to torch.repeat().

        For example:
        
        - repeat(3) creates 3 copies along a new first dimension
        - repeat(2,3) creates a 2x3 grid of copies in new dimensions

        The coefficients and exponents are repeated while preserving the polynomial structure,
        effectively creating multiple independent copies of the same polynomial.

        Examples
        --------
        .. code-block:: python

            # Create a polynomial
            exp = torch.tensor([[2, 1], [1, 0]])  # x^2y + x
            coef = torch.tensor([1.0, 1.0])
            poly = Polynomial(exp, coef)

            # Repeat 3 times along new first dimension
            polys1 = poly.repeat(3)  # Shape: [3, n_vars, n_terms]

            # Create 2x3 grid of copies
            polys2 = poly.repeat(2, 3)  # Shape: [2, 3, n_vars, n_terms]

            # Evaluate repeated polynomials
            x = torch.randn(2, 3, 2)  # [2, 3, n_vars]
            y = polys2(x)  # [2, 3] outputs

        Parameters
        ----------
        *args : int
            The number of repetitions for each new dimension. Similar to torch.repeat() arguments.

        Returns
        -------
        Polynomials
            A new Polynomials object with the repeated structure. The
            shape will be ``[*args, *original_shape]``.
        """

        exps = self._exp[None, :, :].repeat(math.prod(args), 1, 1, 1)
        coef = self._coef[None, :].repeat(math.prod(args), 1)
        exps = exps.reshape(*args, self.n_vars, self.n_terms)
        coef = coef.reshape(*args, self.n_terms)
        return Polynomials(exps, coef)
       
    @classmethod 
    def lin_exp(cls, 
                n_vars:int, 
                dtype:torch.dtype=torch.float32, 
                device:torch.device=torch.device('cpu')
                )->'Polynomial':
        """
        Creates a linear polynomial with n_vars variables.
        Creates a polynomial with ``n_vars+1`` terms:

        * A constant term (all exponents 0)
        * Linear terms for each variable (exponent 1 for that variable, 0 for others)

        For example, with n_vars=2, creates polynomial with terms:
        :math:`1, x, y`

        Examples
        --------
        .. code-block:: python

            # Create linear polynomial with 2 variables (1 + x + y)
            poly = Polynomial.lin_exp(2)
            print(poly)  # 1 + x + y

            # Create linear polynomial with 3 variables (1 + x + y + z)
            poly = Polynomial.lin_exp(3)
            print(poly)  # 1 + x + y + z

            # Specify dtype and device
            poly = Polynomial.lin_exp(2, dtype=torch.float64, device=torch.device('cuda'))
            print(poly.dtype)  # torch.float64
            print(poly.device)  # cuda:0

        Parameters
        ----------
        n_vars: int 
            number of vairables
        dtype: torch.dtype
            the data type of the polynomial
        device: torch.device
            the device of the polynomial

        Returns
        -------
        polynomial: Polynomial n_vars=n_vars n_terms=n_vars+1
        """
        exp = torch.cat([
            torch.zeros(n_vars, 1),
            torch.eye(n_vars)
        ], -1) # [n_vars, n_vars+1]
        exp = exp.type(dtype).to(device)
        return Polynomial(exp)
                
    @classmethod 
    def poly_exp(cls, 
                 n_vars:int, 
                 dim:int,
                 dtype:torch.dtype=torch.float32,
                 device:torch.device=torch.device('cpu')
                 )->'Polynomial':
        r"""
        Creates a polynomial with terms up to a maximum total degree.

        For n_vars variables and maximum degree dim, includes all terms where
        the sum of exponents is less than or equal to dim, sorted by total degree.

        For example, with n_vars=2, dim=2, creates polynomial with terms:
        :math:`1, x, y, x^2, xy, y^2`

        For n_vars=1, creates polynomial with terms up to power dim:
        :math:`1, x, x^2, \ldots, x^{dim}`

        Examples
        --------
        .. code-block:: python

            # Create polynomial with terms up to degree 2 in 2 variables
            poly = Polynomial.poly_exp(2, 2)
            print(poly)  # 1 + x + y + x^2 + xy + y^2

            # Create polynomial with terms up to degree 3 in 1 variable 
            poly = Polynomial.poly_exp(1, 3)
            print(poly)  # 1 + x + x^2 + x^3

            # Create polynomial with terms up to degree 2 in 3 variables
            poly = Polynomial.poly_exp(3, 2)
            print(poly)  # 1 + x + y + z + x^2 + xy + xz + y^2 + yz + z^2

            # Specify dtype and device
            poly = Polynomial.poly_exp(2, 2, dtype=torch.float64, device=torch.device('cuda'))
            print(poly.dtype)  # torch.float64
            print(poly.device)  # cuda:0

        Parameters
        ----------
        n_vars: int 
            number of vairables
        dim: int
            number of maximum dimension 
        dtype: torch.dtype
            the data type of the polynomial
        device: torch.device
            the device of the polynomial
        Returns
        -------
        polynomial: Polynomial n_vars=n_vars 
        """
        if n_vars == 1:
            exp = torch.arange(dim+1)
        else:
            axes = torch.meshgrid(*[torch.arange(dim+1) for _ in range(n_vars)], indexing='xy')
            axes = list(map(lambda x: x.flatten(), axes))
            exp  = torch.stack(axes, 0)
            exp  = exp[:, exp.sum(0) <= dim]
            exp  = exp[:, exp.sum(0).argsort()]

        exp = exp.type(dtype).to(device)
        return Polynomial(exp)

    @classmethod 
    def tens_exp(cls, 
                 n_vars:int, 
                 dim:int,
                 dtype:torch.dtype=torch.float32,  
                 device:torch.device=torch.device('cpu')
                 )->'Polynomial':
        """
        Creates a tensor product polynomial with terms up to maximum degree in each variable.

        For n_vars variables and maximum degree dim, includes all terms where
        each exponent is less than or equal to dim, sorted by total degree.

        For example, with n_vars=2, dim=2, creates polynomial with terms:
        :math:`P(x,y) = a_0 + a_1x + a_2y + a_3x^2 + a_4xy + a_5y^2 + a_6x^2y + a_7xy^2 + a_8x^2y^2`

        For n_vars=1, dim=d, creates polynomial with terms up to power dim:
        :math:`P(x) = a_0 + a_1x + a_2x^2 + ... + a_dx^d`

        Examples
        --------
        .. code-block:: python

            # Create tensor product polynomial with 2 variables up to degree 2
            poly = Polynomial.tens_exp(2, 2)
            print(poly)  # 1 + x + y + x^2 + xy + y^2 + x^2y + xy^2 + x^2y^2
            
            # Create tensor product polynomial with 1 variable up to degree 3
            poly = Polynomial.tens_exp(1, 3)
            print(poly)  # 1 + x + x^2 + x^3
            
            # Specify dtype and device
            poly = Polynomial.tens_exp(2, 2, dtype=torch.float64, device=torch.device('cuda'))
            print(poly.dtype)  # torch.float64
            print(poly.device)  # cuda:0
        
        Parameters
        ----------
        n_vars: int 
            number of vairables
        dim: int
            number of maximum dimension 
        dtype: torch.dtype
            the data type of the polynomial
        device: torch.device
            the device of the polynomial
        Returns
        -------
        polynomial: Polynomial n_vars=n_vars n_terms=(dim+1)**n_vars
        """
        if n_vars == 1:
            exp = torch.arange(dim+1)[None, :] # [1, dim+1]
        else:
            axes = torch.meshgrid(*[torch.arange(dim+1) for _ in range(n_vars)], indexing='xy') 
            axes = list(map(lambda x: x.flatten(), axes))
            exp  = torch.stack(axes, 0) # [n_vars, (dim+1)**n_vars]
            exp  = exp[:, exp.sum(0).argsort()]

        exp = exp.type(dtype).to(device)

        return Polynomial(exp)

    @classmethod 
    def pyr_exp(cls, 
                order:int,
                dtype:torch.dtype=torch.float32,
                device:torch.device=torch.device('cpu')
                )->'Polynomial':
        """
        Creates a pyramid polynomial with terms up to maximum total degree.

        For order n, includes all terms where x+z < n+1 and y+z < n+1, sorted by total degree.

        For example, with order=1, creates polynomial with terms:
        :math:`P(x,y,z) = a_0 + a_1x + a_2y + a_3z`

        With order=2, creates polynomial with terms:
        :math:`P(x,y,z) = a_0 + a_1x + a_2y + a_3z + a_4x^2 + a_5xy + a_6y^2 + a_7xz + a_8yz + a_9z^2`

        Examples
        --------
        .. code-block:: python

            # Create order 1 pyramid polynomial
            poly = Polynomial.pyr_exp(1)
            print(poly)  # 1 + x + y + z
            
            # Create order 2 pyramid polynomial with custom dtype and device
            poly = Polynomial.pyr_exp(2, dtype=torch.float64, device=torch.device('cuda'))
            print(poly.dtype)  # torch.float64
            print(poly.device)  # cuda:0
            
            # Evaluate polynomial at point x
            x = torch.tensor([1.0, 2.0, 0.5], device='cuda', dtype=torch.float64)  # [x,y,z]
            print(poly(x))  # Evaluates polynomial at (1,2,0.5)

        Parameters
        ----------
        order : int
            Maximum polynomial order
        dtype : torch.dtype, optional
            Data type of polynomial coefficients
        device : torch.device, optional
            Device to store polynomial on

        Returns
        -------
        polynomial : Polynomial
            Pyramid polynomial with n_vars=3 and appropriate number of terms
        """
        axes = torch.meshgrid(torch.arange(order+1), 
                              torch.arange(order+1), 
                              torch.arange(order+1), 
                              indexing='xy')
        axes = list(map(lambda x: x.flatten(), axes))
        exp  = torch.stack(axes, 0)
        exp  = exp[:, (exp[0]+exp[2] < order+1) & (exp[1]+exp[2] < order+1)]
        exp  = exp[:, exp.sum(0).argsort()]
        exp  = exp.type(dtype).to(device)
        return Polynomial(exp)
            
    @classmethod 
    def pri_exp(cls, 
                order:int,
                dtype:torch.dtype=torch.float32,
                device:torch.device=torch.device('cpu')
                )->'Polynomial':
        """
        Creates a prismatic polynomial with terms up to maximum total degree.

        For order n, includes all terms where x+y < n+1, sorted by total degree.

        For example, with order=1, creates polynomial with terms:
        :math:`P(x,y,z) = a_0 + a_1x + a_2y + a_3z`

        With order=2, creates polynomial with terms:
        :math:`P(x,y,z) = a_0 + a_1x + a_2y + a_3z + a_4x^2 + a_5xy + a_6y^2 + a_7xz + a_8yz + a_9z^2`

        Examples
        --------
        .. code-block:: python

            # Create prismatic polynomial of order 2
            poly = Polynomial.pri_exp(2)
            print(poly)  # 1 + x + y + z + x^2 + xy + y^2 + xz + yz + z^2
            
            # Create with specific dtype and device
            poly = Polynomial.pri_exp(2, dtype=torch.float64, device=torch.device('cuda'))
            print(poly.dtype)  # torch.float64
            print(poly.device)  # cuda:0
            
            # Evaluate polynomial at point x
            x = torch.tensor([1.0, 2.0, 0.5], device='cuda', dtype=torch.float64)  # [x,y,z]
            print(poly(x))  # Evaluates polynomial at (1,2,0.5)

        Parameters
        ----------
        order : int
            Maximum polynomial order
        dtype : torch.dtype, optional
            Data type of polynomial coefficients
        device : torch.device, optional
            Device to store polynomial on

        Returns
        -------
        polynomial : Polynomial
            Prismatic polynomial with n_vars=3 and appropriate number of terms
        """
        axes = torch.meshgrid(torch.arange(order+1), 
                              torch.arange(order+1), 
                              torch.arange(order+1), 
                              indexing='xy')
        axes = list(map(lambda x: x.flatten(), axes))
        exp  = torch.stack(axes, 0)
        exp  = exp[:, exp[0]+exp[1] < order+1]
        exp  = exp[:, exp.sum(0).argsort()]
        exp  = exp.type(dtype).to(device)
        return Polynomial(exp)

class Polynomials(nn.Module):

    """
    A collection of polynomials that can be evaluated and manipulated together.

    This class represents a batch of polynomials, allowing vectorized operations across multiple polynomials.
    Each polynomial has the same number of variables and terms, but can have different coefficients.

    The polynomials are stored in a batched format with coefficients and exponents tensors:

    - Coefficients tensor shape: [n_poly1, ..., n_terms] 
    - Exponents tensor shape: [n_poly1, ..., n_vars, n_terms]

    Where n_poly1, ... are the batch dimensions.

    
    Examples
    --------
    .. code-block:: python

        # Create a batch of 2 quadratic polynomials in 2 variables
        exp = torch.tensor([[[0,1,0,2,0], [0,0,1,0,2]],
                          [[0,1,0,2,0], [0,0,1,0,2]]])  # [2, 2, 5]
        coef = torch.tensor([[1,2,3,4,5], [2,3,4,5,6]]) # [2, 5]
        polys = Polynomials(exp, coef)

        # Print batch shape and dimensions
        print(polys.shape)     # (2,)
        print(polys.n_vars)    # 2 
        print(polys.n_terms)   # 5

        # Evaluate polynomials at points
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # [2, 2]
        y = polys(x)  # Evaluates both polynomials at their respective points

        # Take derivatives
        d_polys = polys.deriv(0)  # Derivative with respect to first variable
        
        # Convert to individual polynomials
        poly_list = [polys[i] for i in range(len(polys))]

        # Create with specific dtype and device
        exp = exp.cuda().double()
        coef = coef.cuda().double() 
        polys = Polynomials(exp, coef)
        print(polys.dtype)    # torch.float64
        print(polys.device)   # cuda:0

    Parameters
    ----------
    exp : torch.Tensor
        Tensor of exponents with shape [n_poly1, ..., n_vars, n_terms]
    coef : torch.Tensor, optional
        Tensor of coefficients with shape [n_poly1, ..., n_terms]. If None, defaults to ones.

    Attributes
    ----------
    _coef : torch.Tensor
        The coefficients tensor
    _exp : torch.Tensor  
        The exponents tensor
    n_polys : Tuple[int, ...]
        The batch dimensions
    n_vars : int
        Number of variables in each polynomial
    n_terms : int
        Number of terms in each polynomial
    device : torch.device
        The device the tensors are stored on
    dtype : torch.dtype
        The data type of the tensors
    shape : Tuple[int, ...]
        The batch dimensions (same as n_polys)

    """
    _coef:torch.Tensor 
    """Coefficients tensor of shape [n_poly1, ..., n_terms] where:
    
    * n_poly1, ... = batch dimensions for multiple polynomials
    * n_terms = number of terms in each polynomial
    """

    _exp:torch.Tensor  
    """Exponents tensor of shape [n_poly1, ..., n_vars, n_terms] where:
    
    * n_poly1, ... = batch dimensions for multiple polynomials
    * n_vars = number of variables in each polynomial
    * n_terms = number of terms in each polynomial
    """

    n_polys:Tuple[int, ...]
    """Batch dimensions for multiple polynomials
    :no-index:
    """

    n_vars:int
    """Number of variables in each polynomial
    :no-index:
    """

    n_terms:int
    """Number of terms in each polynomial
    :no-index:
    """

    def __init__(self, 
                 exp:torch.Tensor, 
                 coef:Optional[torch.Tensor]=None):
        """Initialize multiple polynomials with exponents and coefficients.

        Parameters
        ----------
        exp : torch.Tensor
            Exponents tensor of shape :math:`[N_1, \ldots, N_v, N_t]` where:
            * :math:`N_1, \ldots` = batch dimensions for multiple polynomials
            * :math:`N_v` = number of variables in each polynomial 
            * :math:`N_t` = number of terms in each polynomial
        coef : Optional[torch.Tensor], optional
            Coefficients tensor of shape :math:`[N_1, \ldots, N_t]` where:
            * :math:`N_1, \ldots` = batch dimensions for multiple polynomials
            * :math:`N_t` = number of terms in each polynomial
            If None, coefficients will be initialized as ones.

        Examples
        --------
        .. code-block:: python

            # Create 2x3 grid of polynomials x^2y + 2xy^2 + 3
            exp = torch.tensor([[[2,1,0], [1,2,0]]]).repeat(2,3,1,1)  # [2,3,2,3]
            coef = torch.tensor([[1,2,3]]).repeat(2,3,1)              # [2,3,3]
            polys = Polynomials(exp, coef)  # 2x3 grid of polynomials

            # Print shape
            print(polys.shape)  # (2,3)

            # Access individual polynomials
            print(polys[0,0])  # x^2y + 2xy^2 + 3
            print(polys[1,2])  # x^2y + 2xy^2 + 3
        """
        super().__init__()
        *n_polys, n_vars, n_terms = exp.shape
        if coef is None:
            coef = torch.ones(*n_polys, n_terms, dtype=exp.dtype, device=exp.device)
        for i in range(len(n_polys)):
            assert coef.shape[i] == n_polys[i]
        assert coef.shape[-1] == n_terms
        assert exp.dtype == coef.dtype, f"exp and coef should have the same dtype, but got {exp.dtype} and {coef.dtype}"
        assert exp.device == coef.device, f"exp and coef should have the same device, but got {exp.device} and {coef.device}"
        # self._coef  = coef
        # self._exp   = exp
        self.register_buffer('_coef', coef)
        self.register_buffer('_exp', exp)
        self.n_polys = n_polys 
        self.n_vars  = n_vars 
        self.n_terms = n_terms

    
    def __len__(self):
        return self.n_polys[0]
    
    def numel(self):
        return math.prod(self.n_polys)

    @property 
    def shape(self):
        """Get batch dimensions of polynomials.
        :no-index:
        """
        return tuple(self.n_polys)
    
    @property 
    def ndim(self):
        return len(self.n_polys)
    
    @property
    def device(self):
        return self._coef.device
    
    @property
    def dtype(self):
        return self._coef.dtype

    def _apply(self, fn, recurse: bool = True):
        # Same complex-stripping override as ``Polynomial._apply`` — keep
        # basis-function buffers real even when the enclosing module is
        # cast to a complex dtype. See ROADMAP item 2 / the Polynomial
        # docstring for the rationale.
        def _real_only(t):
            out = fn(t)
            if out.is_complex():
                real_dtype = (
                    torch.float32 if out.dtype == torch.complex64 else torch.float64
                )
                out = out.real.to(real_dtype)
            return out
        return super()._apply(_real_only, recurse=recurse)

    def dim(self):
        return len(self.n_polys)
    
    def __iter__(self):
        self._iter_index = 0
        return self

    def __next__(self):
        if self._iter_index < len(self):
            result = self[self._iter_index]
            self._iter_index += 1
            return result
        else:
            raise StopIteration

    def __getitem__(self, indices)->Union[Tuple[torch.Tensor,torch.Tensor],
                                    'Polynomial',
                                    'Polynomials']:
        _coef = self._coef[indices]
        _exp  = self._exp[indices]
        if _coef.dim() == 0:
            return _coef, _exp
        elif _coef.dim() == 1:
            return Polynomial(_exp, _coef)
        elif _coef.dim() > 1:
            return Polynomials(_exp, _coef)
        else:
            raise Exception(f"Invalid input shape {indices}")
        
    def __str__(self, max_show_item=2):

        def vector2str(max_show_item=2):
            string = []
            for i in range(len(self)):
                string.append(self[i].__str__(max_show_item))

            if max_show_item is None or len(self) <= max_show_item*2:
                return  "[" + ",\n".join(string) + "]"
            else:
                return "[" + ",\n".join(string[:max_show_item]) + ",\n...\n" + ",\n".join(string[-max_show_item:]) + "]" 

        def matrix2str(max_show_row=2, max_show_col=2):
            matrix = []
            for i in range(self.shape[0]):
                row = []
                for j in range(self.shape[1]):
                    row.append(self[i,j].__str__(max_show_col))
                matrix.append(row)

            if max_show_row is None or self.shape[0] <= max_show_row*2:
                if max_show_col is None or self.shape[1] <= max_show_col*2:
                    return "["+"\n".join(["["+", ".join(row)+"]" for row in matrix])+"]"
                else:
                    return "[" + "\n".join(["["+", ".join(row[:max_show_col])+", ..., "+", ".join(row[-max_show_col:])+"]" for row in matrix]) + "]"

            else:
                if max_show_col is None or self.shape[1] <= max_show_col*2:
                    return "["+"\n".join(["["+", ".join(row)+"]" for row in matrix[:max_show_row]])+"\n...\n"+"\n".join(["["+", ".join(row)+"]" for row in matrix[-max_show_row:]])+"]"
                else:
                    return ("[" +
                            "\n".join(["["+", ".join(row[:max_show_col])+", ..., "+", ".join(row[-max_show_col:])+"]" for row in matrix[:max_show_row]]) + 
                            "\n...\n" + 
                            "\n".join(["["+", ".join(row[:max_show_col])+", ..., "+", ".join(row[-max_show_col:])+"]" for row in matrix[-max_show_row:]]) + 
                            "]")
        
        if self.dim() == 1:
            return vector2str(max_show_item)
        elif self.dim() == 2:
            return matrix2str(max_show_item, max_show_item)
        else:
            return f"PolynomialTensor[{self.shape}] n_terms={self.n_terms} n_vars={self.n_vars}"
    
    def __repr__(self):
        return self.__str__()

    def forward(self, x:torch.Tensor)->torch.Tensor:
        r"""
        Evaluates the polynomial at the given input points.

        For a polynomial :math:`p(x) = \sum_i c_i \prod_j x_j^{e_{ij}}`, computes:

        1. Evaluates each term by raising inputs to exponents
        2. Multiplies terms by coefficients 
        3. Sums the terms

        For example, for polynomial :math:`2x^2y + 3xy^2` with point :math:`[2,3]`:

        .. math::
            2(2^2 \cdot 3^1) + 3(2^1 \cdot 3^2) = 2(12) + 3(18) = 78

        The computation is vectorized across batches and/or multiple polynomials.
        

        Examples
        --------
        .. code-block:: python

            # Single polynomial evaluation
            exp = torch.tensor([[2, 1], [1, 2]])  # x^2y + xy^2
            coef = torch.tensor([2, 3])           # 2x^2y + 3xy^2
            poly = Polynomial(exp, coef)
            x = torch.tensor([2.0, 3.0])          # Point [2,3]
            print(poly(x))                        # 2*(2^2*3) + 3*(2*3^2) = 78.0

            # Batch evaluation
            x_batch = torch.tensor([[2.0, 3.0], [1.0, 2.0]])  # 2 points
            print(poly(x_batch))                  # [78.0, 14.0]

            # Multiple polynomials
            polys = Polynomials.stack([poly, poly])     # 2 copies of same polynomial
            print(polys(x))                       # [78.0, 78.0]

            # Batch + multiple polynomials 
            print(polys(x_batch))                 # [[78.0, 78.0], [14.0, 14.0]]

       
        Parameters
        ----------
        x : torch.Tensor
            Input points of shape ``[n_batch, n_poly1, ..., n_vars]`` or
            ``[n_poly1, ..., n_vars]``.

        Returns
        -------
        torch.Tensor
            Evaluated values of shape ``[n_batch, n_poly1, ...]`` or
            ``[n_poly1, ...]``.
        """

        x = self.get_exp_terms(x)             # [n_batch, n_poly1, ..., n_terms] or [n_poly1, ..., n_terms]
        x = self.apply_coefficient(x)         # [n_batch, n_poly1, ...] or [n_poly1, ...]
        return x
    
    def get_exp_terms(self, x:torch.Tensor)->torch.Tensor:
        r"""
        Evaluates the polynomial terms for each input point by raising to the appropriate exponents.

        For each input point :math:`x` and term with exponents :math:`e`, computes:

        .. math::
            x_0^{e_0} \cdot x_1^{e_1} \cdot \ldots \cdot x_n^{e_n}

        For example, for polynomial :math:`x^2y` with point :math:`[2,3]`, computes:

        .. math::
            2^2 \cdot 3^1 = 12

        The computation is vectorized across batches and/or multiple polynomials.

        Examples
        --------
        .. code-block:: python

            # Single polynomial evaluation
            exp = torch.tensor([[2, 1], [1, 2]])  # x^2y + xy^2
            coef = torch.tensor([2, 3])           # 2x^2y + 3xy^2
            poly = Polynomial(exp, coef)
            x = torch.tensor([2.0, 3.0])          # Point [2,3]
            terms = poly.get_exp_terms(x)         # [12, 18] (2^2*3, 2*3^2)

            # Batch evaluation
            x_batch = torch.tensor([[2.0, 3.0], [1.0, 2.0]])  # 2 points
            terms = poly.get_exp_terms(x_batch)   # [[12, 18], [2, 4]]

            # Multiple polynomials
            polys = Polynomials.stack([poly, poly])     # 2 copies of same polynomial
            terms = polys.get_exp_terms(x)        # [[12, 18], [12, 18]]

            # Batch + multiple polynomials
            terms = polys.get_exp_terms(x_batch)  # [[[12, 18], [12, 18]], 
                                                 #  [[2, 4], [2, 4]]]

        Parameters
        ----------
        x : torch.Tensor
            Input points of shape ``[n_batch, n_poly1, ..., n_vars]`` or
            ``[n_poly1, ..., n_vars]``. Can include a batch dimension.
            ``n_poly1, ...`` are optional polynomial batch dimensions.
            Last dimension must match number of variables.

        Returns
        -------
        torch.Tensor
            Output of shape ``[n_batch, n_poly1, ..., n_terms]`` or
            ``[n_poly1, ..., n_terms]``.
            Evaluated terms for each input point.
            Output has same batch/polynomial dimensions as input.
            Last dimension contains values for each term.
        """
        assert x.device == self.device, f"x and self should have the same device, but got {x.device} and {self.device}"
        assert x.dtype == self.dtype, f"x and self should have the same dtype, but got {x.dtype} and {self.dtype}"
        if x.dim() == self.dim() + 2:
            # shape check
            n_batch, *n_polys, n_vars = x.shape
            assert n_polys == self.n_polys, f"Expected n_polys: {self.n_polys} but got {n_polys}"
            assert n_vars  == self.n_vars, f"Expected n_vars: {self.n_vars} but got {n_vars}"
            x    = torch.pow(x[..., None], self._exp[None, ...])  # [n_batch, n_poly1, ..., n_vars, n_terms]
            x    = torch.prod(x, dim=-2) # [n_batch, n_poly1, ..., n_terms]
        elif x.dim() == self.dim() + 1:
            # shape check 
            *n_polys, n_vars = x.shape 
            assert n_polys == self.n_polys, f"Expected n_polys: {self.n_polys} but got {n_polys}"
            assert n_vars  == self.n_vars, f"Expected n_vars: {self.n_vars} but got {n_vars}"
            x    = torch.pow(x[..., None], self._exp)  # [n_poly1, ..., n_vars, n_terms]
            x    = torch.prod(x, dim=-2) # [n_poly1, ..., n_terms]
        else:
            raise Exception(f"Should be shape of [n_batch, {self.shape}, {self.n_vars}] or [{self.shape}, {self.n_vars}], but got Invalid input shape {x.shape}")
        return x
    
    def apply_coefficient(self,  x:torch.Tensor)->torch.Tensor:
        r"""
        Applies the polynomial coefficients to the evaluated terms.

        For each polynomial with coefficients :math:`c` and evaluated terms :math:`t`, computes:

        .. math::
            \sum_i c_i t_i

        For example, for polynomial :math:`2x^2 + 3y` with evaluated terms :math:`[4,3]`, computes:

        .. math::
            2 \cdot 4 + 3 \cdot 3 = 17

        The computation is vectorized across batches and/or multiple polynomials.

        The coefficients are broadcast to match the batch dimensions of the input.


        Examples
        --------
        .. code-block:: python

            # Single polynomial evaluation
            exp = torch.tensor([[2, 1], [0, 1]])  # x^2, y
            coef = torch.tensor([2, 3])           # 2x^2 + 3y
            poly = Polynomial(exp, coef)
            terms = torch.tensor([4, 3])          # terms = [x^2=4, y=3]
            poly.apply_coefficient(terms)          # 2*4 + 3*3 = 17

            # Batch evaluation
            terms = torch.tensor([[4, 3], [1, 2]]) # 2 points
            poly.apply_coefficient(terms)          # [2*4 + 3*3, 2*1 + 3*2]

            # Multiple polynomials
            exp = torch.tensor([[[2, 1], [0, 1]], # 2x^2 + 3y
                              [[1, 2], [1, 0]]])  # x^2y + xy
            coef = torch.tensor([[2, 3], [1, 1]])
            poly = Polynomials(exp, coef)
            terms = torch.tensor([[4, 3], [12, 2]]) # terms for each polynomial
            poly.apply_coefficient(terms)          # [2*4 + 3*3, 1*12 + 1*2]

        Parameters
        ----------
        x:torch.Tensor
            ND Tensor of shape [n_batch, n_poly1, ..., n_terms] or [n_poly1, ..., n_terms]
            
        Returns
        -------
        torch.Tensor
            ND Tensor of shape [n_batch, n_poly1, ..., n_poly2] or [n_poly1, ..., n_poly2] where:
            
            * n_batch = number of input points (optional)
            * n_poly1, ..., n_poly2 = polynomial dimensions
        """
        assert x.device == self.device, f"x and self should have the same device, but got {x.device} and {self.device}"
        assert x.dtype == self.dtype, f"x and self should have the same dtype, but got {x.dtype} and {self.dtype}"
        assert x.shape[-1] == self.n_terms
        for i in range(self.dim()):
            assert x.shape[-2 - i] == self.n_polys[-1-i], f"Expected {self.n_polys} but got {x.shape[-self.dim()-1:-1]}"


        if x.dim() == self.dim() + 2:
            n_batch, *n_polys, n_terms = x.shape
            x = self._coef.reshape(1, *n_polys, self.n_terms) * x # [n_batch, n_poly1, ..., n_terms]
            x = torch.sum(x, dim=-1)           # [n_batch, n_poly1, ..., n_poly2]
        elif x.dim() == self.dim()+1:
            *n_polys, n_terms = x.shape
            x = self._coef.reshape(*self.shape, self.n_terms) * x # [n_poly1, ..., n_terms]
            x = torch.sum(x, dim=-1) # [n_poly1, ..., n_poly2]
        else:
            raise ValueError(f"Expected input of shape [n_batch, {self.shape}, {self.n_vars}] or [{self.shape}, {self.n_vars}], but got {x.shape}")
        return x
    
    def map(self, x:torch.Tensor)->torch.Tensor:
        r"""
        Evaluates the polynomial at given input points.

        For each polynomial with coefficients c and exponents e, computes:

        .. math::
            \sum_i c_i \prod_j x_j^{e_{ij}}

        For example, for polynomial :math:`2x^2 + 3y` evaluated at point :math:`(2,3)`, computes:

        .. math::
            2 \cdot 2^2 + 3 \cdot 3 = 17

        The computation is vectorized across batches and/or multiple polynomials.
        The input points are broadcast to match the polynomial dimensions.

        Examples
        --------
        .. code-block:: python

            # Single polynomial evaluation
            exp = torch.tensor([[2, 1], [1, 2]])  # x^2y, xy^2
            coef = torch.tensor([2, 3])           # 2x^2y + 3xy^2
            poly = Polynomial(exp, coef)
            x = torch.tensor([2.0, 3.0])          # Point (2,3)
            poly.map(x)                           # 2*(2^2*3) + 3*(2*3^2) = 24 + 54 = 78
            # tensor(78.)

            # Batch evaluation
            x = torch.tensor([[2.0, 3.0], [1.0, 2.0]])  # Two points
            poly.map(x)                                  # Evaluate at both points
            # tensor([78., 14.])

            # Multiple polynomials
            polys = Polynomials.stack([poly, poly])  # Two copies of same polynomial
            x = torch.tensor([2.0, 3.0])       # Single point
            polys.map(x)                       # Evaluate both polynomials
            # tensor([78., 78.])

            # Batch + multiple polynomials
            x = torch.tensor([[2.0, 3.0], [1.0, 2.0]])  # Two points
            polys.map(x)                                 # Evaluate both polys at both points
            # tensor([[78., 78.],
            #         [14., 14.]])

        Parameters
        ----------
        x : torch.Tensor
            Input points tensor of shape [n_batch, n_vars] or [n_vars].
            Each row represents a point to evaluate the polynomial(s) at.

        Returns
        -------
        torch.Tensor
            Evaluated polynomial values. Shape is:
            - [n_batch, n_poly1, ...] if input is [n_batch, n_vars]  
            - [n_poly1, ...] if input is [n_vars]
            Where n_poly1, ... are the polynomial batch dimensions.
        """
        x = self.map_exp_terms(x) # [n_batch, n_poly1, ..., n_terms] or [n_poly1, ..., n_terms]
        x = self.apply_coefficient(x) # [n_batch, n_poly1, ...,] or [n_poly1, ...,]
        return x

    def map_exp_terms(self, x:torch.Tensor)->torch.Tensor:
        r"""
        Evaluates the polynomial terms at given input points by applying exponents.

        For each polynomial with exponents e, computes:

        .. math::
            \prod_j x_j^{e_{ij}}

        For example, for polynomial terms :math:`x^2y, xy^2` evaluated at point :math:`(2,3)`, computes:
        
        .. math::
            [2^2 \cdot 3^1, 2^1 \cdot 3^2] = [12, 18]

        The computation is vectorized across batches and/or multiple polynomials.
        The input points are broadcast to match the polynomial dimensions.


        Examples
        --------
        .. code-block:: python

            # Single polynomial, single point
            exp = torch.tensor([[2, 1], [1, 2]])  # x^2y, xy^2
            poly = Polynomial(exp)
            x = torch.tensor([2.0, 3.0])  # Point (2,3)
            terms = poly.map_exp_terms(x)  # Evaluate terms
            # tensor([12., 18.])  # 2^2 * 3^1, 2^1 * 3^2

            # Single polynomial, batch of points
            x = torch.tensor([[2.0, 3.0], [1.0, 2.0]])  # Two points
            terms = poly.map_exp_terms(x)  # Evaluate terms at each point
            # tensor([[12., 18.],  # Point 1: 2^2 * 3^1, 2^1 * 3^2
            #         [1., 4.]])   # Point 2: 1^2 * 2^1, 1^1 * 2^2

            # Multiple polynomials
            polys = Polynomials.stack([poly, poly])  # Two copies
            x = torch.tensor([2.0, 3.0])       # Single point
            terms = polys.map_exp_terms(x)      # Evaluate terms for both polys
            # tensor([[12., 18.],
            #         [12., 18.]])

            # Batch + multiple polynomials
            x = torch.tensor([[2.0, 3.0], [1.0, 2.0]])  # Two points
            terms = polys.map_exp_terms(x)               # Evaluate at each point
            # tensor([[[12., 18.],
            #          [12., 18.]],
            #         [[1., 4.],
            #          [1., 4.]]])

        Parameters
        ----------
        x : torch.Tensor
            Input points tensor of shape [n_batch, n_vars] or [n_vars].
            Each row represents a point to evaluate the polynomial terms at.

        Returns
        -------
        torch.Tensor
            Evaluated polynomial term values. Shape is:
            - [n_batch, n_poly1, ..., n_terms] if input is [n_batch, n_vars]
            - [n_poly1, ..., n_terms] if input is [n_vars]
            Where n_poly1, ... are the polynomial batch dimensions.
        """
        assert x.device == self.device, f"x and self should have the same device, but got {x.device} and {self.device}"
        assert x.dtype == self.dtype, f"x and self should have the same dtype, but got {x.dtype} and {self.dtype}"
        assert x.shape[-1] == self.n_vars
        if x.dim() == 1:
            x = x.repeat(math.prod(self.n_polys), self.n_vars)
            x = x.reshape(*self.n_polys, self.n_vars)
        elif x.dim() == 2:
            n_batch = x.shape[0]
            x = x[:, None, :].repeat(1, math.prod(self.n_polys), 1)
            x = x.reshape(n_batch, *self.n_polys, self.n_vars)
        else:
            raise Exception(f"Should be shape of [n_batch, {self.n_vars}] or [{self.n_vars}], but got Invalid input shape {x.shape}")
           
        return self.get_exp_terms(x)

    def reshape(self, *args:Sequence[int])->'Polynomials':
        """
        Reshapes the polynomial batch dimensions.

        Creates a new Polynomials object with the coefficients and exponents reshaped to the specified dimensions.
        The total number of elements must remain the same.

        Similar to torch.reshape(), this operation changes the batch dimensions while preserving the polynomial structure.
        The n_vars and n_terms dimensions are preserved at the end.

        Parameters
        ----------
        *args : Sequence[int]
            The new shape dimensions. The product of these dimensions must equal the product of the original batch dimensions.

        Returns
        -------
        Polynomials
            A new Polynomials object with reshaped batch dimensions.

        Examples
        --------
        >>> poly = Polynomials(exp, coef)  # shape (6,)
        >>> reshaped = poly.reshape(2,3)   # shape (2,3)
        """
        exps = self._exp.reshape(*args, self.n_vars, self.n_terms)
        coef = self._coef.reshape(*args, self.n_terms)
        return Polynomials(exps, coef)
    
    def transpose(self, *args:Sequence[int])->'Polynomials':
        """
        Transposes the polynomial batch dimensions.

        Creates a new Polynomials object with the coefficients and exponents transposed according to the specified dimension ordering.
        The n_vars and n_terms dimensions are preserved at the end.

        Similar to torch.transpose(), this operation permutes the batch dimensions while preserving the polynomial structure.

        Parameters
        ----------
        *args : Sequence[int]
            The new ordering of dimensions. Must include all dimensions up to dim().

        Returns
        -------
        Polynomials
            A new Polynomials object with transposed batch dimensions.

        Examples
        --------
        >>> poly = Polynomials(exp, coef)  # shape (2,3)
        >>> transposed = poly.transpose(1,0)  # shape (3,2)
        """
        assert len(args) == self.dim() 
        exps = self._exp.transpose(*args, self.dim(), self.dim() + 1)
        coef = self._coef.transpose(*args, self.dim())
        return Polynomials(exps, coef)
        
    def deriv(self, var_ind:int=0)->'Polynomials':
        r"""
        Compute the derivative of the polynomial with respect to a variable.

        For a polynomial :math:`p(x_1,\ldots,x_n)`, computes :math:`\frac{\partial p}{\partial x_i}` 
        where i is the specified variable index.

        For example, given :math:`p(x,y) = ax^ny^m`, the derivatives are:

        * :math:`\frac{\partial p}{\partial x} = nax^{n-1}y^m`
        * :math:`\frac{\partial p}{\partial y} = max^ny^{m-1}`

        The derivative is computed by:

        1. Decrementing the exponent of the specified variable by 1
        2. Multiplying coefficients by the original exponent
        3. Setting terms with exponent 0 to 0


        Examples
        --------
        .. code-block:: python

            # Single polynomial derivative
            exp = torch.tensor([[2, 1], [1, 2]])  # x^2y + xy^2
            coef = torch.tensor([2, 3])           # 2x^2y + 3xy^2
            poly = Polynomial(exp, coef)
            
            # Derivative with respect to x
            dx = poly.deriv(0)                    # 4xy + 3y^2
            
            # Derivative with respect to y  
            dy = poly.deriv(1)                    # 2x^2 + 6xy

            # Multiple polynomials
            polys = Polynomials.stack([poly, poly])     # [2x^2y + 3xy^2, 2x^2y + 3xy^2]
            dx = polys.deriv(0)                   # [4xy + 3y^2, 4xy + 3y^2]
            dy = polys.deriv(1)                   # [2x^2 + 6xy, 2x^2 + 6xy]

        Parameters
        ----------
            var_ind: int 
                    the index of the variable to be differentiated
        
        Returns
        -------
            deriv_poly: Polynomial
                        the derivative of the polynomial
        """
        assert var_ind < self.n_vars, f"var_ind should be less than {self.n_vars}, but got {var_ind}"
        coef = self._coef.clone() # [n_poly1, ..., n_terms]
        exp  = self._exp.clone()  # [n_poly1, ..., n_vars, n_terms]
        where_constant = exp[..., var_ind, :] == 0 # [n_poly1, ..., n_terms]
        # ``where`` instead of ``clamp_min`` so this works for complex
        # dtypes too (complex ``clamp`` is not supported by PyTorch).
        # Constant terms (exp == 0) are zeroed by ``mask`` below anyway,
        # so the value picked here only has to be a safe non-negative
        # exponent — zero is fine.
        zero  = exp[..., var_ind, :].new_zeros(())
        exp[..., var_ind, :] = torch.where(
            where_constant, zero, exp[..., var_ind, :] - 1
        )
        mask     = torch.ones_like(coef) # [n_poly1, ..., n_terms]
        mask[where_constant] = 0         # [n_poly1, ..., n_terms]
        coef     = coef * (exp[..., var_ind, :] + 1) * mask # [n_poly1, ..., n_terms]
        return self.__class__(exp, coef)
    
    def grad(self)->'Polynomials':
        r"""
        Compute the gradient of the polynomial.

        For a polynomial :math:`p(x_1,\ldots,x_n)`, returns a vector of partial derivatives:
        :math:`\nabla p = [\frac{\partial p}{\partial x_1}, \ldots, \frac{\partial p}{\partial x_n}]`

        For example, given :math:`p(x,y) = ax^ny^m`, the gradient is:
        :math:`\nabla p = [nax^{n-1}y^m, max^ny^{m-1}]`

        The gradient is computed by taking the derivative with respect to each variable:

        1. For each variable i=1...n:
           - Compute :math:`\frac{\partial p}{\partial x_i}` using deriv(i)
        2. Stack the derivatives into a vector

        Examples
        --------
        .. code-block:: python

            # Single polynomial gradient
            exp = torch.tensor([[2, 1], [1, 2]])  # x^2y + xy^2
            coef = torch.tensor([2, 3])           # 2x^2y + 3xy^2
            poly = Polynomial(exp, coef)
            grad = poly.grad()                    # [4xy + 3y^2, 2x^2 + 6xy]

            # Multiple polynomials
            polys = Polynomials.stack([poly, poly])     # [2x^2y + 3xy^2, 2x^2y + 3xy^2]
            grad = polys.grad()                   # [[4xy + 3y^2, 4xy + 3y^2],
                                                 #  [2x^2 + 6xy, 2x^2 + 6xy]]

            # Evaluate gradient at points
            x = torch.tensor([2.0, 3.0])          # Point [2,3]
            grad_vals = grad(x)                   # [[36, 36], [24, 24]]

            # Batch evaluation
            x_batch = torch.tensor([[2.0, 3.0],   # 2 points
                                  [1.0, 2.0]])
            grad_vals = grad(x_batch)             # [[[36, 36], [24, 24]],
                                                 #  [[8, 8], [4, 4]]]

        Parameters
        ----------
        None

        Returns
        -------
        Polynomials
            The gradient of the polynomial, with shape
            ``[n_vars, n_poly1, ..., n_polyn]``, ``n_vars`` variables and
            ``n_terms`` terms each.
        """
        # TODO: could be more efficient
        return Polynomials.stack([self.deriv(i) for i in range(self.n_vars)])

    def reset_coef(self, coef:torch.Tensor):
        """
        Reset the coefficients of the polynomial

        Parameters
        ----------
        coef : torch.Tensor
            Coefficients of shape ``[n_poly, n_terms]``.
        """
        assert coef.shape == self._coef.shape, f"reset coef error: expected {self._coef.shape} but got {coef.shape}"
        assert coef.device== self.device, f"reset coef error: expected {self.device} but got {coef.device}"
        assert coef.dtype == self.dtype, f"reset coef error: expected {self.dtype} but got {coef.dtype}"
        self._coef = coef
        return self

    def repeat(self, *args):
        """
        Repeat the polynomial along specified dimensions.

        Creates a new Polynomials object by repeating the current polynomial's coefficients and 
        exponents according to the provided repeat dimensions. This is similar to torch.repeat().

        For example:
        - repeat(3) creates 3 copies along a new first dimension
        - repeat(2,3) creates a 2x3 grid of copies in new dimensions

        The coefficients and exponents are repeated while preserving the polynomial structure,
        effectively creating multiple independent copies of the same polynomial.

        Parameters
        ----------
        *args : int
            The number of repetitions for each dimension. Must match the number of dimensions
            in the polynomial (self.dim()).

        Returns
        -------
        Polynomials
            A new Polynomials object with the repeated structure. The shape will be 
            [args[0]*original_shape[0], args[1]*original_shape[1], ...].
        """
        assert len(args) == self.dim()
        exps = self._exp.repeat(*args, 1, 1)
        coef = self._coef.repeat(*args, 1)
        return Polynomials(exps, coef)

    @classmethod 
    def stack(cls, polys:Sequence[Union[Polynomial,'Polynomials']])->'Polynomials':
        """
        Stack multiple polynomials into a single Polynomials object.

        Takes a sequence of Polynomial or Polynomials objects and stacks them along a new first dimension.
        All polynomials must have the same number of variables and terms.

        Examples
        --------
        .. code-block:: python

            # Create two simple polynomials: x^2 + y and 2x + y^2
            exp1 = torch.tensor([[2,1], [0,1]])  # Exponents for x^2 + y
            coef1 = torch.tensor([1.0, 1.0])     # Coefficients [1,1]
            p1 = Polynomial(exp1, coef1)
            
            exp2 = torch.tensor([[1,0], [0,2]])  # Exponents for 2x + y^2  
            coef2 = torch.tensor([2.0, 1.0])     # Coefficients [2,1]
            p2 = Polynomial(exp2, coef2)
            
            # Stack the polynomials
            stacked = Polynomials.stack([p1, p2])
            print(stacked.shape)  # (2,)
            
            # Evaluate stacked polynomials at point [2,3]
            x = torch.tensor([2.0, 3.0])
            print(stacked(x))  # [7, 13] = [(2^2 + 3), (2*2 + 3^2)]

        Parameters
        ----------
        polys : Sequence[Union[Polynomial, Polynomials]]
            Sequence of polynomials to stack. All must have same n_vars, n_terms, dtype and device.

        Returns
        -------
        Polynomials
            A new Polynomials object with shape [len(polys), ...] containing the stacked polynomials.

        Examples
        --------
        >>> p1 = Polynomial(exp1, coef1)  # First polynomial
        >>> p2 = Polynomial(exp2, coef2)  # Second polynomial with same structure
        >>> stacked = Polynomials.stack([p1, p2])
        >>> stacked.shape
        (2,)
        """
        # check shape
        for i in range(1, len(polys)):
            assert polys[i].__class__ == polys[0].__class__, "All polynomials must have the same class."
            assert (polys[i].n_vars == polys[0].n_vars and 
                    polys[i].n_terms == polys[0].n_terms), \
                "All polynomials must have the same number of variables and terms."
            assert polys[i].dtype == polys[0].dtype, "All polynomials must have the same dtype."
            assert polys[i].device== polys[0].device, "All polynomials must have the same device."
            if isinstance(polys[0], Polynomials):
                assert polys[i].n_polys == polys[0].n_polys, "All polynomials must have the same number of polynomials."
        
        exps = torch.stack([p._exp for p in polys])
        coef = torch.stack([p._coef for p in polys])
        return cls(exps, coef)



   


