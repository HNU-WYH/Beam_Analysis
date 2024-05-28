import sympy as sp
import numpy as np
from scipy.integrate import quad
from config import LoadType


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

    def __global_basis_func(idx, domain, num_element):

        return

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
        if load_type == LoadType.q:
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

    @staticmethod
    def app_func(x, u, domain, num_elements):
        """
        approximate a function f with basis function phi_i and coordinates u_i such that
        $$
        f = \sum u_iphi_i
        $$

        :param x: the input
        :param u: coefficient of basis function
        :param domain: [start,end]
        :param num_elements:
        :return f: the approximate function with basis functions
        """
        start, end = domain
        element_len = (end - start) / num_elements
        node_list = np.linspace(start, end, num_elements + 1)
        bas_func, _ = LocalElement._loc_basis_func()

        start_idx = int(x/element_len)
        end_idx = start_idx + 1
        start = node_list[start_idx]

        f_val = 0
        f_val += u[start_idx*2] * bas_func[0](x - start, element_len)
        f_val += u[start_idx*2 + 1] * bas_func[1](x - start, element_len)

        if end_idx < len(node_list):
            f_val += u[end_idx*2] * bas_func[2](x - start, element_len)
            f_val += u[end_idx*2 +1] * bas_func[3](x - start, element_len)

        return f_val


def approximation_test():
    import matplotlib.pyplot as plt
    num_elements = 5
    plot_precision = 100
    domain = [0, 2 * np.pi]
    x_values = np.linspace(domain[0], domain[1], plot_precision)
    node_list = np.linspace(domain[0], domain[1], num_elements + 1)

    # Create u as coefficients of the basis functions to approximate sin(x)
    u = np.zeros(2 * len(node_list))
    u[0::2] = np.sin(node_list)  # coefficients for phi_2i-1
    u[1::2] = np.cos(node_list)  # coefficients for phi_2i

    # Get the approximate function
    approx_values = np.array([LocalElement.app_func(x, u, domain, num_elements) for x in x_values])

    # Original function
    original_function = np.sin(x_values)
    x = np.linspace(domain[0], domain[1], 200)
    sin_x = np.sin(x)

    # Plot the original function and the approximation
    plt.plot(x, sin_x, label='Original sin(x)')
    plt.plot(x_values, approx_values, label='Approximation', linestyle='--')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Approximation of sin(x) using basis functions')
    plt.show()


if __name__ == "__main__":
    # LocMat = LocalElement()
    # print("the Stiffness Matrix:")
    # print(np.array(LocMat.S))
    # print("\nthe Mass Matrix:")
    # print(np.array(LocMat.M))
    approximation_test()
