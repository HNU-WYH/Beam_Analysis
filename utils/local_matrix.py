import sympy as sp
import numpy as np


class LocalMatrix:
    def __init__(self):
        self.S, self.M = self.local_matrix()

    @staticmethod
    def local_matrix():
        # symbols
        x, h, E, I, n, rho = sp.symbols('x h E I n rho')

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

    def evaluate(self, E_val, I_val, rho_val, h_val):
        S_func = sp.lambdify((sp.symbols('h E I')), self.S, "numpy")
        M_func = sp.lambdify((sp.symbols('h rho')), self.M, "numpy")

        S_num = S_func(h_val, E_val, I_val)
        M_num = M_func(h_val, rho_val)

        return np.array(S_num), np.array(M_num)

if __name__ == "__main__":
    LocMat = LocalMatrix()
    print("the Stiffness Matrix:")
    print(np.array(LocMat.S))
    print("\nthe Mass Matrix:")
    print(np.array(LocMat.M))
