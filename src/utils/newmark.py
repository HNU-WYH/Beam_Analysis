from typing import Tuple, Any

import numpy as np
from numpy import ndarray, dtype, floating
from numpy._typing import _64Bit


class NewMark:
    """
    This class implements the Newmark method to solve a second-order ordinary differential equation (ODE)
    of the form:

        M*u''(t) + D*u'(t) + S*u(t) = p(t)

    where:
    - M is the mass matrix,
    - D is the damping matrix (ignored in this project),
    - S is the stiffness matrix, and
    - p(t) is the external force vector.

    References:
    1. [Chapter 8] script1_bending_and_fem.pdf:
        a. Dynamic Bending Equation:

            \mu\ddot{w}+(EIw'')''=q

            where:
            - \mu is the mass density
            - w(x,t) is the curve of deflection
            - \ddot{w} is the 2nd derivative of deflection w.r.t time
            - w'' is the 2nd derivative of deflection w.r.t x

        b. Using Weak Form & Galerkin method, the above is converted to:

            M_e*x(t) + S_e*x(t) = f_e

            details can be found in equation (20)

    2. script2_newmark.pdf:

        Induction & Proof of the Newmark method.

    """

    def __init__(self, tau: float, num_steps: int, beta=0.25, gamma=0.5):
        """
        Initialize the NewMark solver with the given hyperparameters.

        Parameters:
        - tau: Time step size.
        - num_steps: Number of time steps to simulate.
        - beta: Newmark parameter, typically 0.25
        - gamma: Newmark parameter, typically 0.5
        """
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.num_steps = num_steps

    def solve(self, M: np.ndarray, S: np.ndarray, f: np.ndarray, x: np.ndarray, dx: np.ndarray, ddx: np.ndarray) \
            -> (ndarray, ndarray, ndarray):

        """
        The Newmark Method is used to solve the second-order differential equation of motion:

        M * ü(t) + S * u(t) = f(t)

        where M is the mass matrix, S is the stiffness matrix, and f(t) is the external force vector at time t.

        The method integrates the equation over time to calculate displacement (u), velocity (du), and acceleration (ddu).

        :param M: the Mass Matrix (n x n matrix, where n is the number of degrees of freedom)
        :param S: the Stiffness Matrix (n x n matrix)
        :param f: the external force vector that may vary with time (n-dimensional vector or n x m matrix, where m is the number of time steps)
        :param x: initial displacement condition of x ∈ ℝ^n (n-dimensional vector)
        :param dx: initial velocity condition of ẋ ∈ ℝ^n (n-dimensional vector)
        :param ddx: initial acceleration condition of ẍ ∈ ℝ^n (n-dimensional vector)
        :return: u, du, ddu: Displacement, velocity, and acceleration at each time step (n x m matrices)
        """

        # Initialize
        num_dofs = M.shape[0]
        u = np.zeros([num_dofs, self.num_steps])
        du = np.zeros([num_dofs, self.num_steps])
        ddu = np.zeros([num_dofs, self.num_steps])

        # external force
        # if f is static, it is an n-dimensional vector
        # if f is dynamic, it is an n x m matrix, where m is the number of time steps
        f = f.reshape([num_dofs,-1])
        if f.shape[1] == 1:
            f = np.tile(f, (1, self.num_steps))

        # Set initial conditions
        u[:, 0] = x
        du[:, 0] = dx
        ddu[:, 0] = ddx

        # Time integration loop using Newmark method
        for j in range(self.num_steps - 1):
            # Compute predicted displacement and velocity
            u_star = u[:, j] + self.tau * du[:, j] + (0.5 - self.beta) * self.tau ** 2 * ddu[:, j]
            du_star = du[:, j] + (1 - self.gamma) * ddu[:, j] * self.tau

            # Compute the right-hand side
            rhs = f[:, j + 1] - S @ u_star

            # Compute the right-hand side
            ddu[:, j + 1] = np.linalg.solve(S * self.beta * self.tau ** 2 + M, rhs)

            # Update displacement and velocity with the new acceleration
            u[:, j + 1] = u_star + self.beta * ddu[:, j + 1] * self.tau ** 2
            du[:, j + 1] = du_star + self.gamma * ddu[:, j + 1] * self.tau

        return u, du, ddu