import sympy as sp
import numpy as np
from scipy.integrate import quad
from config import LoadType


class LocalElement:
    """
    Properties & Calculation in an one-dimensional local element [node_i,node_{i+1}]
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

class LocalElement2D:
    """
        Properties & Calculation in a two-dimensional local element of beam [node_i,node_{i+1}] with longitudinal & vertical displacement.

        For 2-dimensional structure, we need additionally consider the longitudinal deformation using the PDE below:

        ρü - (EAu')' = f

        where
        - ρ is the density,
        - E is the young module
        - A is the area of cross-section
        - u is the axial deformation,
        - f is the distributed axial force.

        By using weak formulation & galerkin method, the above PDE becomes the following matrix equation:

        Mä + Sa = q + F(L)e_L - F(0)e_0

        where:
        - u = Σ a_i φ_i
        - M_{ij} = ∫ ρ φ_i φ_j dx
        - S_{ij} = ∫ EA φ_i' φ_j' dx
        - q_i = ∫ f φ_i dx
        - F(L),F(0) is the axial force at x = L and x = 0
        - e_0, e_L is the unit vector

        By rearranging the stiffness, matrix of longitudinal and vertical displacement & slopes,
        we can get the 2-dimensional S & M matrices in a local element:

        Let u1,u2 be the basis function of longitudinal deformation
        Let f1,f2,f3,f4 be the basis function of vertical displacement & slopes

        Then we have the following matrix equation:

        Mä + Sa = q + M_Le_L - M_0e_0 + Q_Le_L - Q_0e_0 + F_Le_L - F_0e_0

        where

        S = [∫ EA u1' u1' dx, ∫ EA u1' u2' dx, ... , 0                , 0                , 0                , 0                , ... ],
            [∫ EA u2' u1' dx, ∫ EA u2' u2' dx, ... , 0                , 0                , 0                , 0                , ... ],
            [⋮              , ⋮              , ⋮   , ⋮                , ⋮                , ⋮                , ⋮                , ... ],
            [0              , 0              , ... , ∫ EI f1'' f1'' dx, ∫ EI f1'' f2'' dx, ∫ EI f1'' f3'' dx, ∫ EI f1'' f4'' dx, ... ],
            [0              , 0              , ... , ∫ EI f2'' f1'' dx, ∫ EI f2'' f2'' dx, ∫ EI f2'' f3'' dx, ∫ EI f2'' f4'' dx, ... ],
            [0              , 0              , ... , ∫ EI f3'' f1'' dx, ∫ EI f3'' f2'' dx, ∫ EI f3'' f3'' dx, ∫ EI f3'' f4'' dx, ... ],
            [0              , 0              , ... , ∫ EI f4'' f1'' dx, ∫ EI f4'' f2'' dx, ∫ EI f4'' f3'' dx, ∫ EI f4'' f4'' dx, ... ]
            [⋮              , ⋮              , ⋮   , ⋮                , ⋮                , ⋮                , ⋮                , ⋮   ],

        M = [∫ ρ u1 u1 dx, ∫ ρ u1 u2 dx , ... , 0              , 0              , 0              , 0              , ... ],
            [∫ ρ u2 u1 dx, ∫ ρ u2 u2 dx , ... , 0              , 0              , 0              , 0              , ... ],
            [⋮           , ⋮            , ... , ⋮              , ⋮              , ⋮              , ⋮              , ... ],
            [0           , 0            , ... , ∫ ρ f1 f1 dx   , ∫ ρ f1 f2 dx   , ∫ ρ f1 f3 dx   , ∫ ρ f1 f4 dx   , ... ],
            [0           , 0            , ... , ∫ ρ f2 f1 dx   , ∫ ρ f2 f2 dx   , ∫ ρ f2 f3 dx   , ∫ ρ f2 f4 dx   , ... ],
            [0           , 0            , ... , ∫ ρ f3 f1 dx   , ∫ ρ f3 f2 dx   , ∫ ρ f3 f3 dx   , ∫ ρ f3 f4 dx   , ... ],
            [0           , 0            , ... , ∫ ρ f4 f1 dx   , ∫ ρ f4 f2 dx   , ∫ ρ f4 f3 dx   , ∫ ρ f4 f4 dx   , ... ]
            [⋮           , ⋮            , ⋮   , ⋮              , ⋮              , ⋮              , ⋮              , ... ],

        Longitudinal & vertical displacements and slopes at the nodes of the beam is:

        x(t) = [v_1(t), ... , v_n(t), w_1(t), ... , w_{2n}(t)]^T = Σa_iu_i + Σa_if_i

        with longitudinal displacement first and vertical displacement second,
        where a is the coefficient,u_i & f_i is the longitudinal & vertical basis function
    """

    @staticmethod
    def _init_local_matrix():
        """

        :return:
        """
        # symbols
        x, E, I, h, A, rho = sp.symbols('x E I h A rho')

        # basis function
        # longitudinal
        u1 = x / h
        u2 = 1 - x / h

        # vertical
        f1 = 1 - 3 * (x / h) ** 2 + 2 * (x / h) ** 3
        f2 = h * (x / h) * (x / h - 1) ** 2
        f3 = 3 * (x / h) ** 2 - 2 * (x / h) ** 3
        f4 = h * (x / h) ** 2 * (x / h - 1)

        f_list = [u1, f1, f2, u2, f3, f4]

        # initialize the stiffness and mass matrix
        M = sp.Matrix.zeros(6, 6)
        S = sp.Matrix.zeros(6, 6)

        # compute S and M
        for i in range(4):
            for j in range(i, 4):
                M[i, j] = rho * sp.integrate(f_list[i] * f_list[j], (x, 0, h))
                M[j, i] = M[i, j]

                S[i, j] = E * I * sp.integrate(sp.diff(f_list[i], x, 2) * sp.diff(f_list[j], x, 2), (x, 0, h))
                S[j, i] = S[i, j]
        return S, M





    @staticmethod
    def evaluate():
        pass


if __name__ == "__main__":
    S,M = LocalElement._init_local_matrix()
    print("the Stiffness Matrix:")
    print(S)
    print("\nthe Mass Matrix:")
    print(M)
