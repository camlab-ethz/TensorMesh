# Torch FEM examples

1. **建模问题**。也就是机器学习的模型框架中包含特定的物理过程：模型中包含了特定的物理计算。因此不 再需要从零开始学习关于物 理问题的基本规律，而只需填补成熟的理论框架中空缺的部分。这样的优势就是可以实现从小型的样本中学习到具有足够泛化能力的模型。举例，比如说我们将FEM里面的本构当作一个learnable parameters，在学习的时候基于可微分的FEM框架，就可以很好地基于样本数据学到足够强泛华能力的模型。
2. **优化问题：**一般地看，可以说最小作用量原理支配了整个物理学。在具体的物理学研究 中，可以利用自动微分计算梯度帮助求解优化问题。
   1. 除此以外，还可以把达到迭代收敛条件视作基本的 计算单元，利用隐函数定理得出伴随变量的回传规则。后面这种做法更加契合微分编程的思想，可以让程序更加高效和省内存。
3. **控制问题**：是指通过调整体系的初始条件和外场等，以实现对于含时物理过程的调控
4. **反向设计**：通过设计合适的物理结构以实 现特定的力场、光场、电磁场等

微分编程的思想，就是定义基本计算单元，对于每个计算单元，利用隐函数定理得出伴随变量的回传规则，从而实现整个differentiable simulation的计算逻辑。



# Benchmark

- 自动微分软件

$$
\begin{array}{|c|c|c|c|}
\hline \text { 编程语言 } & \text { 软件名 } & \text { 开发组织 } & \text { 简介 } \\
\hline \text { Python 等 } & \text { TensorFlow } & \text { Google } & \text { 张量运算性能出色。对工业级别应用支持完善。 } \\
\hline \text { Python 等 } & \text { PyTorch } & \text { Facebook } & \text { 方便高效, 在研究领域流行程度逐渐超越了 Tensorflow。 } \\
\hline \text { Python } & \text { Autograd } & \begin{array}{c}
\text { Dougal } \\
\text { Maclaurin等 }
\end{array} & \begin{array}{c}
\text { 通过实现约两百个可微分的基本单元, 几乎支持对于整个Numpy } \\
\text { 库的自动微分。 }
\end{array} \\
\hline \text { Python } & \text { Jax } & \text { James Bradbury 等 } & \text { Autograd的后代。包括自动矢量化以及对于专用加速硬件的支持等。 } \\
\hline \text { Julia } & \text { Zygote } & \text { FluxML } & \text { Julia语言中最流行的后向模式微分库之一。 } \\
\hline \text { Julia } & \begin{array}{l}
\text { ForwardDiff/ } \\
\text { ReverseDiff }
\end{array} & \text { JuliaDiff } & \begin{array}{c}
\text { ForwardDiff 是性能出色的前向自动微分库。ReverseDiff是后向模式 } \\
\text { 自动微分库,性能比Zygote更好。 }
\end{array} \\
\hline \text { Julia } & \text { NiLang } & \begin{array}{l}
\text { Jin-Guo Liu, } \\
\text { Taine Zhao }
\end{array} & \begin{array}{c}
\text { 在可逆计算编程语言中实现的源代码自动微分, 能够更灵活地 } \\
\text { 平衡内存消耗与计算时间。 }
\end{array} \\
\hline \text { C/Fortran } & \text { Tapenade } & \begin{array}{l}
\text { Laurent Hascoët, } \\
\text { Valérie Pascual }
\end{array} & \begin{array}{l}
\text { 较为古老的微分编程工具之一。它通过源代码转换实现了对于一般程序的前向和 } \\
\text { 后向自动微分。性能出色, 并且在洋流和大气模拟等科学计算中有一些应用。 }
\end{array} \\
\hline \mathrm{C}++ & \text { Adept } & \text { Robin Hogan } & \text { C++语言中性能最好的通用自动微分库之一。包括前向和后向自动微分。 } \\
\hline
\end{array}
$$

- Differentiable Programming for Scientific computing
	- [JAX-CFD](https://github.com/google/jax-cfd): Google
	- [JAX-AM](https://github.com/tianjuxue/jax-am): Northwestern
	- $\Phi$[-flow](https://tum-pbs.github.io/PhiFlow/): TUM
	- [torch-ODE](https://github.com/rtqichen/torchdiffeq): UoT
	- [ADCME](https://github.com/kailaix/ADCME.jl): Stanford, Julia
	- [$\xi$​-torch](https://github.com/xitorch/xitorch/), [paper](https://arxiv.org/pdf/2010.01921.pdf)
	- [SciML](https://sciml.ai/)
	- [dolfimadjoint](https://www.dolfin-adjoint.org/en/latest/)

# Basis

Define basis functions over the reference domain $\left\{\psi_i(\boldsymbol{\xi})\right\}_{i=1}^{N_{\text {nd }}^{\text {el }}}$ for $\boldsymbol{\xi} \in \Omega_{\square}$, where $\Omega_{\square} \subset \mathbb{R}^d$ is the reference/parent element. 

#### Simplex

For this we will use Vandermonde's approach (discussed in lecture and Ch. 6). A basis for a simplex element of order $p$ must contain all multinomial terms of order $p$ in $d$ dimensions $\left\{\xi_1^{\alpha_1} \cdots \xi_d^{\alpha_d} \mid \sum_{i=1}^d \alpha_i \leqslant p\right\}$, so we can write our $N_{\mathrm{nd}}^{\mathrm{el}}$ basis functions as
$$
\psi_i(\boldsymbol{\xi})=\sum_{k=1}^{N_{\mathrm{nd}}^{\mathrm{el}}} \alpha_{i k} \prod_{j=1}^d \xi_j^{\Upsilon_{j k}}
$$
where $\Upsilon \in M_{d, N_{\mathrm{nd}}^{\text {el }}}\left(\mathbb{N}_0\right)$ with entries $\Upsilon_{i j}, i=1, \ldots, d, j=1, \ldots, N_{\mathrm{nd}}^{\mathrm{el}}$, such that $\sum_{i=1}^d \Upsilon_{i j} \leqslant p$ for each $j=1, \ldots, N_{\mathrm{nd}}^{\mathrm{el}}$ that is used to sweep over all $N_{\mathrm{nd}}^{\mathrm{el}}$​​ permissible exponents. For example:

- in the special case of $d=2$ (triangle) and $p=1$, we have

$$
\mathbf{\Upsilon}=\left[\begin{array}{lll}
0 & 1 & 0 \\
0 & 0 & 1
\end{array}\right] \quad \Longrightarrow \quad \psi_i\left(\xi_1, \xi_2\right)=\alpha_{i 1}+\alpha_{i 2} \xi_1+\alpha_{i 3} \xi_2
$$

- in the special case of $d=2$ and $p=2$, we have

$$
\mathbf{\Upsilon}=\left[\begin{array}{llllll}
0 & 1 & 0 & 2 & 1 & 0 \\
0 & 0 & 1 & 0 & 1 & 2
\end{array}\right] \quad \Longrightarrow \quad \psi_i\left(\xi_1, \xi_2\right)=\alpha_{i 1}+\alpha_{i 2} \xi_1+\alpha_{i 3} \xi_2+\alpha_{i 4} \xi_1^2+\alpha_{i 5} \xi_1 \xi_2+\alpha_{i 6} \xi_2^2
$$

- in the special case of $d=3$ (tetrahedron) and $p=1$, we have

$$
\mathbf{\Upsilon}=\left[\begin{array}{llll}
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{array}\right] \quad \Longrightarrow \quad \psi_i\left(\xi_1, \xi_2, \xi_3\right)=\alpha_{i 1}+\alpha_{i 2} \xi_1+\alpha_{i 3} \xi_2+\alpha_{i 4} \xi_3
$$

==$\mathbf{\Upsilon}$ is used to store the permissible exponents.==

For convenience, we can introduce the function $\omega_i(\boldsymbol{\xi}), i=1, \ldots, N_{\mathrm{nd}}^{\mathrm{el}}$ to simplify the expression, (==the column of $\mathbf{\Upsilon}$==)
$$
\omega_i(\boldsymbol{\xi})=\prod_{s=1}^d \xi_s^{\Upsilon_{s i}}
$$
The basis functions can be expressed as $\psi_i(\boldsymbol{\xi})=\sum_{k=1}^{N_{\text {nd }}^{\text {el }}} \alpha_{i k} \omega_k(\boldsymbol{\xi})$.

<font color=red>Next, we need to derive the coefficient $\alpha_{ik}$ based on the properies of the element basis function.</font>

Denote the $N_{\text {nd }}^{\text {el }}$ nodes of the $p$ th order simplex element as $\left\{\hat{\boldsymbol{\xi}}_i\right\}_{i=1}^{N_{\text {nd }}^{\text {el }}}$, where $\hat{\boldsymbol{\xi}}_i=\left(\hat{\xi}_{1 i}, \ldots, \hat{\xi}_{d i}\right)^T$​. The nodal property is
$$
\psi_i\left(\hat{\boldsymbol{\xi}}_j\right)=\delta_{i j},
$$
for $i, j=1, \ldots, N_{\mathrm{nd}}^{\mathrm{el}}$, which leads to
$$
\sum_{k=1}^{N_{\text {nd }}^{\text {el }}} \alpha_{i k} \omega_k\left(\hat{\boldsymbol{\xi}}_j\right)=\delta_{i j}
$$


Let $\hat{V}_{i j}=\omega_j\left(\hat{\boldsymbol{\xi}}_i\right)=\prod_{s=1}^d \hat{\xi}_{s i}^{{\Upsilon}_{s j}}$ be the ==Vandermonde matrix (就是将node coordinates带入从而得到线性方程组的系数矩阵，行是node数目，列也是系数组合数目，也是node数目)== corresponding to the $d$-dimensional, $p$ th order simplex evaluated at $\left\{\hat{\boldsymbol{\xi}}_i\right\}_{i=1}^{N_{\text {nd }}^{\text {el }}}$, then the above constraints can be written in matrix form as $\hat{\boldsymbol{V}} \boldsymbol{\alpha}^T=\boldsymbol{I}_{N_{\mathrm{nd}}^{\text {el }}}$, where $\hat{\boldsymbol{V}}, \boldsymbol{\alpha}$ are the matrices with indices $\hat{V}_{i j}, \alpha_{i j}$, respectively, and $\boldsymbol{I}_{N_{\mathrm{nd}}^{\text {el }}}$ is the $N_{\mathrm{nd}}^{\mathrm{el}} \times N_{\mathrm{nd}}^{\mathrm{el}}$ identity matrix. Once we compute the coefficients, $\boldsymbol{\alpha}=\hat{\boldsymbol{V}}^{-T}$, we substitute this expression into (2) and evaluate at new points $\left\{\tilde{\boldsymbol{\xi}}_i\right\}_{i=1}^m$ where $\tilde{\boldsymbol{\xi}}_i=\left(\tilde{\xi}_{1 i}, \ldots, \tilde{\xi}_{d i}\right)$ to give
$$
\psi_i\left(\tilde{\boldsymbol{\xi}}_j\right)=\sum_{k=1}^{N_{\mathrm{nd}}^{\mathrm{el}}} \alpha_{i k} \omega_k\left(\tilde{\boldsymbol{\xi}}_j\right)=\sum_{k=1}^{N_{\mathrm{nd}}^{\mathrm{el}}}\left(\hat{V}^{-1}\right)_{k i} \omega_k\left(\tilde{\boldsymbol{\xi}}_j\right)=\sum_{k=1}^{N_{\mathrm{nd}}^{\mathrm{el}}}\left(\hat{V}^{-1}\right)_{k i} \tilde{V}_{j k}
$$
where the last expression used the $d$-dimensional, $p$ th order simplex Vandermonde matrix evaluated at $\left\{\tilde{\boldsymbol{\xi}}_i\right\}_{i=1}^m: \tilde{V}_{i j}=\omega_j\left(\tilde{\boldsymbol{\xi}}_i\right)=\prod_{s=1}^d \tilde{\xi}_{s i}^{\Upsilon_{s j}}$. Therefore, if we define $Q_{i j}=\psi_i\left(\tilde{\boldsymbol{\xi}}_j\right)$, we have
$$
\boldsymbol{Q}=\hat{\boldsymbol{V}}^{-T} \tilde{\boldsymbol{V}}^T
$$
where $\boldsymbol{Q}, \tilde{\boldsymbol{V}}$ are the matrices with indices $Q_{i j}, \tilde{V}_{i j}$, respectively.
The partial derivatives of the simplex basis functions are also needed to implement the finite element method. A simple differentiation calculation reveals
$$
\frac{\partial \psi_i}{\partial \xi_j}(\boldsymbol{\xi})=\sum_{k=1}^{N_{\mathrm{nd}}^{\mathrm{el}}} \alpha_{i k} \frac{\partial \omega_k}{\partial \xi_j}(\boldsymbol{\xi})
$$

where the partial derivatives of $\omega_i(\boldsymbol{\xi})$ are
$$
\frac{\partial \omega_i}{\partial \xi_j}(\boldsymbol{\xi})= \begin{cases}0 & \text { if } \Upsilon_{j i}=0 \\ \Upsilon_{j i} \xi_j^{\Upsilon_{j i}-1} \prod_{s=1, s \neq j}^d \xi_s^{\Upsilon_{s i}} & \text { if } \Upsilon_{j i} \neq 0 .\end{cases}
$$

Then, the basis functions evaluated at the points $\left\{\tilde{\boldsymbol{\xi}}_i\right\}_{i=1}^m$, take the form
$$
\frac{\partial \psi_i}{\partial \xi_j}\left(\tilde{\boldsymbol{\xi}}_k\right)=\sum_{l=1}^{N_{\mathrm{nd}}^{\text {el }}} \alpha_{i l} \frac{\partial \omega_l}{\partial \xi_j}\left(\tilde{\boldsymbol{\xi}}_k\right)=\sum_{l=1}^{N_{\mathrm{nd}}^{\text {el }}} \hat{V}_{l i}^{-1} \tilde{W}_{k l j}
$$
where $\tilde{W}_{i j k}$ contains the partial derivatives of the Vandermonde matrix evaluated at $\left\{\tilde{\boldsymbol{\xi}}_i\right\}_{i=1}^{\text {eld }}$, i.e., $W_{i j k}=$ $\frac{\partial \omega_j}{\partial \xi_k}\left(\tilde{\boldsymbol{\xi}}_i\right)$.

<font color=red>**Code Logic:** </font>

1. Implement a function that evaluates the Vandermonde matrix and its derivative corresponding to the $d-$dimensional, $p$th order simplex.
2. Implement a function that evaluates the basis functions and their derivatives for $d$-dimensional simplex of order $p$ given the coordinates of the element nodes $x_k$ and points at which to evaluate the basis $x$ (these will eventually be quadrature points).

- `poly_mltdim.py` `poly_onedim.py`: to define the polynomial space and the basis functions for $p$th order $d$​​-dimensional simplex element.
  - 在代码中，提供了三种不同的nodal polynomial interpolation. Lagrange就是单纯地基于interpolation node = 1来进行构造。而Hermite polynomial 构造的时候更多了一步导数信息，也就是不仅在这些interpolation node满足为1，还要满足在这些interpolation node的导数一致性。而Vandermonde则是一种更通用的方法，其可以allowing for matching higher-order derivatives at the nodes. Polynomial basis can be adjusted for higher continuity across nodes. 如果global continuity parameter is set to zeros. 那么Vandermonde-induced polynomial approach, the resulting polynomial will indeed behave similarly to the Lagrange polynomials in terms of interpolation at the nodes. 但是在推倒系数时相比于polynomial interpolation具有不同的收敛性和稳定性。



# Boundary Integral


![image-20240321124855307](assets/image-20240321124855307.png)

**Space:**

- $(d-1)$-dimensional reference simplex element $\left(\Gamma_{\square}\right)$ 

- each face of the reference element $\left(\partial \Omega_{\square, f}\right)$
- $d$-dimensional reference simplex element $\left(\Omega_{\square}\right)$
- physical element $\left(\Omega_e\right)$

**Mapping:**

- $\left(\boldsymbol{\xi}=\gamma_f(\boldsymbol{r})\right)$​

$$
\begin{array}{rll}
\mathcal{G}_e: \Omega_{\square} & \rightarrow \Omega_e \\
\boldsymbol{\xi} & \mapsto \boldsymbol{x}=\mathcal{G}_e(\boldsymbol{\xi}),
\end{array}
$$

- $\left(\boldsymbol{x}=\mathcal{G}_e(\boldsymbol{\xi})\right)$​

$$
\begin{aligned}
\gamma_f: \Gamma_{\square} & \rightarrow \partial \Omega_{\square, f} \\
\boldsymbol{r} & \mapsto \boldsymbol{\xi}=\gamma_f(\boldsymbol{r}) .
\end{aligned}
$$

- The composition mapping from the $(d-1)$-dimensional reference simplex element $\left(\Gamma_{\square}\right)$ to each face of the physical element $\left(\partial \Omega_{e f}\right)\left(\boldsymbol{x}=\mathcal{F}_{e f}(\boldsymbol{r})=\mathcal{G}_e\left(\gamma_f(\boldsymbol{r})\right)\right)$​.

$$
\begin{aligned}
& \mathcal{F}_{e f}: \Gamma_{\square} \rightarrow \Omega_e \\
& \boldsymbol{r} \quad \mapsto \quad \boldsymbol{x}=\mathcal{F}_{e f}(\boldsymbol{r})=\mathcal{G}_e\left(\gamma_f(\boldsymbol{r})\right) \\
&
\end{aligned}
$$



Use the mapped master element to define basis functions of elements in the physical domain.

- - Define volume and boundary integrals over the physical domain in terms of integrals over the physical domain in terms of integrals over the corresponding reference domain using a change of coordinates (volume) and surface parametrization (boundary).
  - `transf.py`: define a class, and these transfer quantities are defined as the properties.


$$
\begin{align}
\mathcal{G}_e: \Omega_\square &\rightarrow \Omega_e\\
\xi &\rightarrow x=\mathcal{G}_e(\xi)
\end{align}
$$

and let $\mathcal{G}_e^{-1}:\Omega_e\rightarrow\Omega_\square$ denote the inverse mapping, i.e. $\xi=\mathcal{G}^{-1}_e(x)$. In addition, introduce the regular ($d-1$)-dimensional simplex, $\Gamma_\square\subset\mathbb{R}^{d-1}$, that will be used as the reference domain for each face of the reference element $\Omega_\square$. Let
$$
\begin{align*}
\gamma_f:\Gamma_\square &\rightarrow \partial\Omega_{\square,f}\\
r &\rightarrow \xi=\gamma_f(r)
\end{align*}
$$
Finally, for convenience, we introduce the mapping from the reference domain $\Gamma_\square$ to the physical domain:
$$
\begin{align*}
\mathcal{F}_{ef}: \Gamma_\square &\rightarrow\Omega_e\\
r &\rightarrow x=\mathcal{F}_{ef}(r)=\mathcal{G}_e(\gamma_f(r))
\end{align*}
$$
**Application**

From this construction, we can define volume and boundary integrals over the physical domain in terms of integrals over the corresponding reference domain using a change of coordinates (volume) and surface parametrization (boundary). Consider the integrals
$$
I_v=\int_{\Omega_e} \theta d v, \quad I_s=\int_{\partial \Omega_{e f}} \vartheta d s,
$$
where $\theta: \Omega_e \rightarrow \mathbb{R}$ and $\vartheta: \partial \Omega_{e f} \rightarrow \mathbb{R}$. Using the mapping $\mathcal{G}_e$ for the volume integral and $\mathcal{F}_{e f}$ for the boundary integral, they can be transformed to the reference domain $\Omega_{\square}$ and $\Gamma_{\square}$, respectively,
$$
\begin{aligned}
I_v=\int_{\Omega_e} \theta d v & =\int_{\Omega_{\square}} \theta\left(\mathcal{G}_e(\boldsymbol{\xi})\right) g_e(\boldsymbol{\xi}) d \boldsymbol{\xi} \\
I_s=\int_{\partial \Omega_{e f}} \vartheta d s & =\int_{\Gamma_{\square}} \vartheta\left(\mathcal{F}_{e f}(\boldsymbol{r})\right) \sigma_{e f}(\boldsymbol{r}) d \boldsymbol{r}
\end{aligned}
$$
where
$$
\begin{aligned}
\boldsymbol{G}_e(\boldsymbol{\xi}) & =\frac{\partial \mathcal{G}_e}{\partial \boldsymbol{\xi}}(\boldsymbol{\xi}), & g_e(\boldsymbol{\xi})=\operatorname{det}\left(\boldsymbol{G}_e(\boldsymbol{\xi})\right) \\
\boldsymbol{F}_{e f}(\boldsymbol{r}) & =\frac{\partial \mathcal{F}_{e f}}{\partial \boldsymbol{r}}(\boldsymbol{r}), & \sigma_{e f}(\boldsymbol{r})=\sqrt{\operatorname{det}\left(\boldsymbol{F}_{e f}(\boldsymbol{r})^T \boldsymbol{F}_{e f}(\boldsymbol{r})\right)},
\end{aligned}
$$
where the derivative of $\mathcal{F}_{\text {ef }}$ can be expanded as
$$
\frac{\partial \mathcal{F}_{e f}}{\partial \boldsymbol{r}}(\boldsymbol{r})=\frac{\partial \mathcal{G}_e}{\partial \boldsymbol{\xi}}\left(\gamma_f(\boldsymbol{r})\right) \frac{\partial \gamma_f}{\partial \boldsymbol{r}}(\boldsymbol{r})=\boldsymbol{G}_e\left(\gamma_f(\boldsymbol{r})\right) \frac{\partial \gamma_f}{\partial \boldsymbol{r}}(\boldsymbol{r})
$$
Let $\left\{\left(w_i, \tilde{\boldsymbol{\xi}}_i\right)\right\}_{i=1}^{N_{\mathrm{qd}}^{\mathrm{el}}}$ denote a quadrature rule over $\Omega_{\square}$, where $N_{\mathrm{qd}}^{\mathrm{el}}$ is the number of quadrature points, $\left\{w_i\right\}_{i=1}^{N_{\mathrm{qd}}^{\mathrm{el}}}$ are the quadrature weights, and $\left\{\tilde{\boldsymbol{\xi}}_i\right\}_{i=1}^{N_{\text {nd }}^{\text {el }}}$ are the quadrature nodes. Similarly, let $\left\{\left(w_i^f, \tilde{\boldsymbol{r}}_i\right)\right\}_{i=1}^{N_{\mathrm{qd}}^{\mathrm{fc}}}$ denote a quadrature rule over $\Gamma_{\square}$, where $N_{\mathrm{qd}}^{\mathrm{fc}}$ is the number of quadrature points, $\left\{w_i^f\right\}_{i=1}^{N_{\mathrm{qd}}^{\mathrm{fc}}}$ are the quadrature weights, and $\left\{\tilde{\boldsymbol{r}}_i\right\}_{i=1}^{N_{\mathrm{qd}}^{\mathrm{fc}}}$ are the quadrature nodes. Then, integrals over the reference domains are approximated as
$$
\int_{\Omega_{\square}} \gamma(\boldsymbol{\xi}) d \boldsymbol{\xi} \approx \sum_{k=1}^{N_{\mathrm{qd}}^{\mathrm{el}}} w_k \gamma\left(\tilde{\boldsymbol{\xi}}_k\right), \quad \int_{\Gamma_{\square}} \lambda(\boldsymbol{r}) d \boldsymbol{r} \approx \sum_{k=1}^{N_{\mathrm{qd}}^{\mathrm{fc}}} w_k^f \lambda\left(\tilde{\boldsymbol{r}}_k\right) .
$$

Therefore the integrals in (4) over the physical domains are approximated as
$$
I_v \approx \sum_{k=1}^{N_{\mathrm{qd}}^{\mathrm{el}}} w_k \theta\left(\mathcal{G}_e\left(\tilde{\boldsymbol{\xi}}_k\right)\right) g_e\left(\tilde{\boldsymbol{\xi}}_k\right), \quad I_s \approx \sum_{k=1}^{N_{\mathrm{qd}}^{\mathrm{fc}}} w_k^f \vartheta\left(\mathcal{F}_{e f}\left(\tilde{\boldsymbol{r}}_k\right)\right) \sigma_{e f}\left(\tilde{\boldsymbol{r}}_k\right)
$$
<font color=red>Define basis functions over the physical element. we only need to evaluate the basis functions associated with the reference domains $\Omega_{\square}$ and $\Gamma_{\square}$ at the quadrature nodes associated with those domains. The next question is how can we define this mapping</font>

We define the mappings $\mathcal{G}_e$ and $\mathcal{F}_{e f}$ using the basis functions associated with $\Omega_{\square}$ and $\Gamma_{\square}$
$$
\mathcal{G}_e(\boldsymbol{\xi})=\sum_{i=1}^{N_{\text {nd }}^{\text {el }}} \hat{\boldsymbol{x}}_i^e \psi_i(\boldsymbol{\xi}), \quad \mathcal{F}_{e f}(\boldsymbol{r})=\mathcal{G}_e\left(\gamma_f(\boldsymbol{r})\right)
$$
where $\left\{\hat{\boldsymbol{x}}_i^e\right\}_{i=1}^{N_{\text {nd }}^{\text {el }}}$ are the nodes associated with the element $\Omega_e$ and $\gamma_f$ can be constructed analytically based on the geometry of the reference elements $\Gamma_{\square}$ and $\Omega_{\square}$.

==这里的$\boldsymbol{G}_e(\boldsymbol{\xi})$就是Jacobian matrix。这点我们在之前的代码中讨论过。==
$$
J=\left[\begin{array}{ll}
\frac{\partial x}{\partial \xi} & \frac{\partial y}{\partial \xi} \\
\frac{\partial x}{\partial \eta} & \frac{\partial y}{\partial \eta}
\end{array}\right]=\left[\begin{array}{ll}
\sum_{j=1}^4 \frac{\partial N_j(\xi, \eta)}{\partial \xi} x_j & \sum_{j=1}^4 \frac{\partial N_j(\xi, \eta)}{\partial \xi} y_j \\
\sum_{j=1}^4 \frac{\partial N_j(\xi, \eta)}{\partial \eta} x_j & \sum_{j=1}^4 \frac{\partial N_j(\xi, \eta)}{\partial \eta} y_j
\end{array}\right]=\left[\begin{array}{llll}
\frac{\partial N_1(\xi, \eta)}{\partial \xi} & \frac{\partial N_2(\xi, \eta)}{\partial \xi} & \frac{\partial N_3(\xi, \eta)}{\partial \xi} & \frac{\partial N_4(\xi, \eta)}{\partial \xi} \\
\frac{\partial N_1(\xi, \eta)}{\partial \eta} & \frac{\partial N_2(\xi, \eta)}{\partial \eta} & \frac{\partial N_3(\xi, \eta)}{\partial \eta} & \frac{\partial N_4(\xi, \eta)}{\partial \eta}
\end{array}\right]_{2 \times 4}\left[\begin{array}{ll}
x_1 & y_1 \\
x_2 & y_2 \\
x_3 & y_3 \\
x_4 & y_4
\end{array}\right]
$$
**Coding Logic:**

1. Derive the reference volume-to-face mapping $\gamma_f(\boldsymbol{r}), f=1, \ldots, 3$, for the three faces of the master triangle; the reference volume-to-face mapping $\gamma_f(\boldsymbol{r}), f=1, \ldots, 4$​, for the four faces of the master tetrahedra?

2. Derive the expression for the quantities $\boldsymbol{G}_e(\boldsymbol{\xi})[\text{Jacobian}], \boldsymbol{F}_{e f}(\boldsymbol{r})$, for the transformation given above. The expression should be in terms of the element coordinates $\left\{\hat{\boldsymbol{x}}_i^e\right\}_{i=1}^{N_{\text {nd }}^{\text {el }}}$, the basis functions $\left\{\psi_i\right\}_{i=1}^{N_{n d}^{\text {el }}}$, and the reference volume-to-face mapping $\gamma_f(\boldsymbol{r})$​.
3. Implement a function that evaluates all relevant transformation quantities.
