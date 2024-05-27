import sympy as sp
import numpy as np
from scipy.integrate import quad
from utils.config import LoadType


class LocalElement:
    @staticmethod
    def _loc_basis_func():
        bas_func = [
            lambda x, h: 1 - 3 * (x / h) ** 2 + 2 * (x / h) ** 3,
            lambda x, h: h * (x / h) * (x / h - 1) ** 2,
            lambda x, h: 3 * (x / h) ** 2 - 2 * (x / h) ** 3,
            lambda x, h: h * (x / h) ** 2 * (x / h - 1)
        ]

        # second derivative of basis function
        bas_func_1st = [
            lambda x, h: -6*x/h**2 + 6*x**2/h**3,
            lambda x, h: (x/h-1)**2 + 2*(x/h)*(x/h-1),
            lambda x, h: 6*x/h**2 - 6*x**2/h**3,
            lambda x, h: 2*(x/h)*(x/h-1) + x**2/h**2
        ]
        return bas_func, bas_func_1st

    @staticmethod
    def _init_local_matrix():
        # symbols
        x, E, I, h, rho = sp.symbols('x E I h rho')

        # basis function
        f1 = 1 - 3 * (x / h) ** 2 + 2 * (x / h) ** 3
        f2 = h * (x / h) * (x / h - 1) ** 2
        f3 = 3 * (x / h) ** 2 - 2 * (x / h) ** 3
        f4 = h * (x / h) ** 2 * (x / h - 1)

        f_list = [f1, f2, f3, f4]

        # initialize the stiffness and mass matrix
        M = sp.Matrix.zeros(4, 4)
        S = sp.Matrix.zeros(4, 4)

        # compute S and M
        for i in range(4):
            for j in range(i, 4):
                M[i, j] = rho * sp.integrate(f_list[i] * f_list[j], (x, 0, h))
                M[j, i] = M[i, j]

                S[i, j] = E * I * sp.integrate(sp.diff(f_list[i], x, 2) * sp.diff(f_list[j], x, 2), (x, 0, h))
                S[j, i] = S[i, j]
        return S, M

    @staticmethod
    def evaluate(E_val, I_val, rho_val, h_val):
        S, M = LocalElement._init_local_matrix()
        S_func = sp.lambdify((sp.symbols('h E I')), S, "numpy")
        M_func = sp.lambdify((sp.symbols('h rho')), M, "numpy")

        S_num = S_func(h_val, E_val, I_val)
        M_num = M_func(h_val, rho_val)
        return np.array(S_num), np.array(M_num)

    @staticmethod
    def equal_force(load, load_type: LoadType, xstart, h):
        """
        Compute the equivalent node force based on the equivalent virtual work principle.

        Parameters:
        force : tuple or function
            The tuple for local position and magnitude of the external action (including force and moment for point loads) or a function defining the distributed load in global coordinate.
        force_type : ForceType
            The type of external force ('point' or 'distributed').
        xstart : float
            The start position of the local element in global coordinates.
        h : float
            The length of the local element.

        Returns:
        np.ndarray
            The equivalent node force vector.
        """
        Pe = np.zeros(4)
        bas_func, bas_func_1st = LocalElement._loc_basis_func()
        if load_type == LoadType.p:
            # here load(x) is a function in global coordinates
            def p_local(x_loc):
                x = x_loc +xstart # convert local coordinate into global obe
                return load(x)

            for i, N in enumerate(bas_func):
                Pe[i], _ = quad(lambda x: p_local(x) * N(x, h), 0, h)

        elif load_type == LoadType.F:
            # here load is the position and value of point force
            pos, f = load
            for i, N in enumerate(bas_func):
                Pe[i] = f * N(pos-xstart, h)

        elif load_type == LoadType.M:
            # here load is the position and value of moment
            pos, m = load
            for i, N in enumerate(bas_func_1st):
                Pe[i] = m * N(pos-xstart, h)
        return Pe


if __name__ == "__main__":
    LocMat = LocalElement()
    print("the Stiffness Matrix:")
    print(np.array(LocMat.S))
    print("\nthe Mass Matrix:")
    print(np.array(LocMat.M))
