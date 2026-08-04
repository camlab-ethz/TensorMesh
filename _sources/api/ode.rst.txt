tensormesh.ode
==============

.. py:module:: tensormesh.ode

Runge-Kutta methods
-------------------

.. math::

        \begin{array}{c|c}
        \textbf  c & \mathfrak A \\
        \hline
        &\textbf b^\top
        \end{array}
        \quad = \quad 
        \begin{array}{c|ccc}
        c_1 & a_{11} & \cdots & a_{1s}\\
        \vdots & \vdots & \ddots & \vdots \\
        c_s & a_{s1} & \cdots & a_{ss}\\\hline
        & b_1 & \cdots & b_s
        \end{array}
        \qquad 
        \textbf c, \textbf b \in \mathbb R^s,\mathfrak A\in  \mathbb R^{s\times s}


.. math::

    \textbf k_i =\textbf f(t+c_i\tau, \textbf u +\tau \sum_{j=1}^s a_{ij}\textbf k_j)\quad \Psi^{t,t+\tau}\textbf u = \textbf u+\tau\sum_{i=1}^s b_i \textbf  k_i

.. math::

    c_i = \sum_j a_{ij}


Base classes
------------

.. autoclass:: tensormesh.ode.ExplicitRungeKutta
    :members:
    :show-inheritance:

.. autoclass:: tensormesh.ode.ImplicitLinearRungeKutta
    :members:
    :show-inheritance:

Built-in schemes
----------------

.. autoclass:: tensormesh.ode.ExplicitEuler
    :members:
    :show-inheritance:

.. autoclass:: tensormesh.ode.ImplicitLinearEuler
    :members:
    :show-inheritance:

.. autoclass:: tensormesh.ode.MidPointLinearEuler
    :members:
    :show-inheritance:
