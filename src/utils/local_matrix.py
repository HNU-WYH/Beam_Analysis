import sympy as sp
import numpy as np
from scipy.integrate import quad
from config import LoadType


class LocalElement:
    """
    Properties & Calculation in an local element [node_i,node_{i+1}]
    """
    @staticmethod
    def _loc_basis_func():
        """
        In the finite element methods for beams, each node typically has two basis functions:
            - φ1: representing the displacement
            - φ2: representing the slope, namely the derivative of displacement

        For a given node j:
            - φ1(node_j) = δ_{ij}.

            - φ2'(node_j) = δ_{ij}.

            - where δ_{ij} equals to 1 if i = j, otherwise 0.

        :return:
            bas_func (list): A list of basis functions for node_i & node_{i+1}.
            bas_func_1st (list): A list of the first derivatives of the basis functions
        """
        bas_func = [
            lambda x, h: 1 - 3 * (x / h) ** 2 + 2 * (x / h) ** 3, # First basis function (φ1) of node_i
            lambda x, h: h * (x / h) * (x / h - 1) ** 2,          # Second basis function (φ2) of node_i
            lambda x, h: 3 * (x / h) ** 2 - 2 * (x / h) ** 3,     # First basis function of node_{i+1}
            lambda x, h: h * (x / h) ** 2 * (x / h - 1)           # Second basis function of node_{i+1}
        ]

        # first derivative of basis function
        bas_func_1st = [
            lambda x, h: -6*x/h**2 + 6*x**2/h**3,                 # First derivative of First basis function (φ1') of node_i
            lambda x, h: (x/h-1)**2 + 2*(x/h)*(x/h-1),            # First derivative of Second basis function (φ2') of node_{i+1}
            lambda x, h: 6*x/h**2 - 6*x**2/h**3,                  # First derivative of First basis function of node_i
            lambda x, h: 2*(x/h)*(x/h-1) + x**2/h**2              # First derivative of Second basis function of node_{i+1}
        ]

        return bas_func, bas_func_1st

    @staticmethod
    def _init_local_matrix():
        """
        Initializes the local stiffness and mass matrices for a local one-dimensional element in a beam.

        The stiffness matrix (S) and mass matrix (M) are derived using the basis functions and
        their second derivatives.

        The returned matrices are symbolic and will be numerically evaluated later.

        details & induction of mass & stiffness matrices can be found in script1_bending_and_fem.pdf

        :return: S (sp.Matrix), M (sp.Matrix): The symbolic stiffness matrix & mass matrix.
        """
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
        """
        Numerically evaluates the local stiffness and mass matrices using the provided material and geometric properties.

        Parameters:
            E_val (float): Young's modulus.
            I_val (float): Moment of inertia.
            rho_val (float): Density.
            h_val (float): Length of the element.

        Returns:
            S_num (np.ndarray), M_num (np.ndarray): The numerical stiffness matrix & mass matrix.
        """
        S, M = LocalElement._init_local_matrix()
        S_func = sp.lambdify((sp.symbols('h E I')), S, "numpy")
        M_func = sp.lambdify((sp.symbols('h rho')), M, "numpy")

        S_num = S_func(h_val, E_val, I_val)
        M_num = M_func(h_val, rho_val)
        return np.array(S_num), np.array(M_num)

    @staticmethod
    def equal_force(load, load_type: LoadType, xstart, h):
        """
        When external load is not on the node of elements, an equivalent nodal forces is required  for further computation

        This method converts distributed loads, point forces, and moments into equivalent nodal forces,
        using the equivalent virtual work principle, and then facilitate the following finite element method.

        Parameters:
            load (tuple or function): The load applied to the element. Can be a function (for distributed load) or a tuple (for point load or moment).
            load_type (LoadType): The type of load (e.g., distributed, point force, moment).
            xstart (float): The start position of the element in global coordinates.
            h (float): The length of a local element.

        Returns:
            Pe (np.ndarray): The numerical equivalent nodal force vector.
        """
        Pe = np.zeros(4) # [F1, M1, F2, M2], where F1 is the equivalent point force at first node, M1 is the equivalent moment
        bas_func, bas_func_1st = LocalElement._loc_basis_func()

        # For arbitrary distributed load
        if load_type == LoadType.q:
            # convert global load function load(x) into local one
            # p_local is the distributed load in the current local element
            def p_local(x_loc):
                x = x_loc +xstart
                return load(x)

            for i, N in enumerate(bas_func):
                # integrate the product of basis function and local distributed load from 0 to h.
                product = lambda x: p_local(x) * N(x, h)
                Pe[i], _ = quad(product, 0, h)

        # For arbitrary point load
        elif load_type == LoadType.F:
            # For point load, the variable "load" is no longer a function,
            # but the position and value of point force
            pos, f = load
            for i, N in enumerate(bas_func):
                # equivalent force * unit displacement = actual force * displacement
                Pe[i] = f * N(pos-xstart, h)

        elif load_type == LoadType.M:
            # For Moment, the variable "load" is consist of the position and value of moment
            pos, m = load
            for i, N in enumerate(bas_func_1st):
                # equivalent moment * unit rotation = actual moment * rotation (1st derivative)
                Pe[i] = m * N(pos-xstart, h)
        return Pe

    @staticmethod
    def app_func(x, u, domain, num_elements):
        """
        Approximates a function value f(x) using basis functions and provided coefficients, s.t. f(x) = Σui×φi(x)

        Parameters:
            x (float): The point at which to evaluate the approximation f(x).
            u (np.ndarray): Coefficients of the basis functions.
            domain (list): The [start, end] of the global domain.
            num_elements (int): The number of elements in the domain.

        Returns:
            float: The approximated function value at x.
        """

        start, end = domain
        element_len = (end - start) / num_elements
        node_list = np.linspace(start, end, num_elements + 1) # list of nodes
        bas_func, _ = LocalElement._loc_basis_func()          # local basis function

        # find the element where x locate at
        start_idx = int(x/element_len)                        # the index of left node at the element
        end_idx = start_idx + 1                               # the index of right node at the element
        start = node_list[start_idx]                          # the cooridinate of the start node

        # calculate the approximate value of function by adding values of basis functions
        f_val = 0                                                              # initialization
        f_val += u[start_idx*2] * bas_func[0](x - start, element_len)          # adding value of 1st basis function of 1st node
        f_val += u[start_idx*2 + 1] * bas_func[1](x - start, element_len)      # adding value of 2nd basis function of 1st node

        # when x is located at the right boundary of beam
        # start_node is the last node of the beam
        # we do not have a end node
        # Hence we need to determine whether an end node exists
        if end_idx < len(node_list):
            f_val += u[end_idx*2] * bas_func[2](x - start, element_len)        # adding value of 1st basis function of 2nd node
            f_val += u[end_idx*2 +1] * bas_func[3](x - start, element_len)     # adding value of 2nd basis function of 2nd node

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
