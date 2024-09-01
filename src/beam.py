import numpy as np

from config import LoadType, uniform_load_function
from src.utils.local_matrix import LocalElement, LocalElement2D


class Beam:
    """
    This class represents the preprocessing of a one-dimensional beam structure, with only vertical displacement considered
    assembling the global stiffness and mass matrices by stacking local matrices from each finite element.

    Attributes:
        num_elements (int): The number of finite elements used to discretize the beam.
        num_nodes (int): The number of nodes used to discretize the beam.
        L (float): The total length of the beam.
        E (float): The Young's modulus (elastic modulus) of the beam material.
        rho (float): The density of the beam material.
        I (float): The moment of inertia of the beam's cross-section.
        nodes (np.ndarray): An array of node positions along the length of the beam, evenly spaced between 0 and L.
        element_len (float): The length of each finite element, calculated as L divided by the number of elements.
        elements (list): A list of pairs, where each pair contains the indices of the nodes that form a finite element.
        local_S (np.ndarray): The local stiffness matrix for a single finite element.
        local_M (np.ndarray): The local mass matrix for a single finite element.
        S (np.ndarray): The global stiffness matrix, assembled by stacking local stiffness matrices from all elements.
        M (np.ndarray): The global mass matrix, assembled by stacking local mass matrices from all elements.
    """
    def __init__(self, length, young_module, density, moment_inertia, num_elements):
        """
        Initializes the Beam object with physical & FEM properties, and assembles the global matrices without considering
        the external forces & constraints.

        Parameters:
            length (float): The total length of the beam.
            young_module (float): The Young's modulus (elastic modulus) of the beam material.
            density (float): The density of the beam material.
            moment_inertia (float): The moment of inertia of the beam's cross-section.
            num_elements (int): The number of finite elements used to discretize the beam.
        """
        self.num_elements = num_elements    # Number of finite elements
        self.num_nodes = num_elements + 1   # Number of nodes

        self.E = young_module               # Young's modulus
        self.I = moment_inertia             # Moment of inertia
        self.rho = density                  # Density
        self.L = length                     # Length of the beam

        self.nodes = np.linspace(0, self.L, self.num_nodes)  # positions of all nodes evenly spaced along the beam
        self.element_len = self.L / self.num_elements             # Length of each element

        # Create node pairs (each element connects two nodes)
        self.elements = []
        for i in range(self.num_elements):
            self.elements.append([i, i + 1])

        # Assemble global stiffness and mass matrices
        self.assemble_matrices()

    def local_matrices(self):
        """
        Computes the numerical local stiffness and mass matrices for a single finite element
        using the material and geometric properties of the beam.
        """
        # Evaluate the local stiffness (S) and mass (M) matrices for an element
        self.local_S, self.local_M = LocalElement.evaluate(self.E, self.I, self.rho, self.element_len)

    def assemble_matrices(self):
        """
        Assembles the global stiffness and mass matrices by stacking the local matrices from each element.
        """
        # Calculate local matrices first
        self.local_matrices()

        # Initialize global stiffness (S) and mass (M) matrices
        self.S = np.zeros((2 * self.num_nodes, 2 * self.num_nodes))
        self.M = np.zeros((2 * self.num_nodes, 2 * self.num_nodes))

        # Loop over each element and assemble the global matrices
        for element in self.elements:
            node1 = element[0]
            # Add the local matrix contributions to the global matrices
            self.S[2 * node1:2 * node1 + 4, 2 * node1:2 * node1 + 4] += self.local_S
            self.M[2 * node1:2 * node1 + 4, 2 * node1:2 * node1 + 4] += self.local_M

class Beam2D:
    """
    This class represents the preprocessing of a two-dimensional beam structure, with vertical & longitudinal displacement,
    which can also be used to assembly a two-dimensional framework.

    Attributes:
        num_elements (int): The number of finite elements used to discretize the beam.
        num_nodes (int): The number of nodes used to discretize the beam.
        L (float): The total length of the beam.
        E (float): The Young's modulus (elastic modulus) of the beam material.
        rho (float): The density of the beam material.
        I (float): The moment of inertia of the beam's cross-section.
        A (float): The area of the beam's cross-section.
        angle (float): The angle between the beam and x-ais (from -pi to pi).
        nodes (np.ndarray): An array of node positions along the length of the beam, evenly spaced between 0 and L.
        element_len (float): The length of each finite element, calculated as L divided by the number of elements.
        elements (list): A list of pairs, where each pair contains the indices of the nodes that form a finite element.
        S (np.ndarray): The global stiffness matrix, assembled by stacking local stiffness matrices from all elements.
        M (np.ndarray): The global mass matrix, assembled by stacking local mass matrices from all elements.
        q (nd.ndarray): The nodal force vector without the addition of constraints
        """

    def __init__(self, length, young_module, area, density, moment_inertia, num_elements, angle = 0):
        """
        Initializes the Beam object with physical & FEM properties, and assembles the global matrices without considering
        the external forces & constraints.

        Parameters:
            length (float): The total length of the beam.
            young_module (float): The Young's modulus (elastic modulus) of the beam material.
            area (float): The area of the beam's cross-section.
            density (float): The density of the beam material.
            moment_inertia (float): The moment of inertia of the beam's cross-section.
            num_elements (int): The number of finite elements used to discretize the beam.
            angle (float): The angle between the beam and x-ais (from -pi to pi).
        """
        self.num_elements = num_elements  # Number of finite elements
        self.num_nodes = num_elements + 1  # Number of nodes

        self.E = young_module  # Young's modulus
        self.I = moment_inertia  # Moment of inertia
        self.A = area # area of cross-section
        self.rho = density  # Density
        self.L = length  # Length of the beam
        self.angle = angle

        self.nodes = np.linspace(0, self.L, self.num_nodes)  # positions of all nodes evenly spaced along the beam
        self.element_len = self.L / self.num_elements  # Length of each element

        # Create node pairs (each element connects two nodes)
        self.elements = []
        for i in range(self.num_elements):
            self.elements.append([i, i + 1])

        # Assemble global stiffness and mass matrices
        self.assemble_matrices()

        # Intialize nodal force vector
        self.q = np.zeros((3 * self.num_nodes, 1))

    def assemble_matrices(self):
        """
        Assembles the global stiffness and mass matrices by stacking the local matrices from each element.
        The global stiffness & mass matrices is composed of the following form:

        S = | longitudinal Global S        0       |      M = | longitudinal Global M            0          |
            |           0             vertical S   |          |           0             vertical global M   |
        """
        # Calculate local matrices first
        local_Sl, local_Ml, local_Sv, local_Mv = LocalElement2D.evaluate(self.E, self.I, self.A, self.rho, self.element_len)

        # Initialize global stiffness (S) and mass (M) matrices
        self.S = np.zeros((3 * self.num_nodes, 3 * self.num_nodes))
        self.M = np.zeros((3 * self.num_nodes, 3 * self.num_nodes))

        # the 1st part of global matrices for longitudinal deformation
        S, M = self.__assemble_longitudinal_matrices(local_Sl,local_Ml)
        self.S[0: self.num_nodes, 0: self.num_nodes] = S
        self.M[0: self.num_nodes, 0: self.num_nodes] = M

        # the 2nd party of global matrices for vertical deformation
        S, M = self.__assemble_vertical_matrices(local_Sv, local_Mv)
        self.S[self.num_nodes: 3 * self.num_nodes, self.num_nodes: 3 * self.num_nodes] = S
        self.M[self.num_nodes: 3 * self.num_nodes, self.num_nodes: 3 * self.num_nodes] = M

    def __assemble_vertical_matrices(self, local_Sv, local_Mv):
        Sv = np.zeros((2 * self.num_nodes, 2 * self.num_nodes))
        Mv = np.zeros((2 * self.num_nodes, 2 * self.num_nodes))

        for element in self.elements:
            node1 = element[0]
            # Add the local matrix contributions to the global matrices
            Sv[2 * node1:2 * node1 + 4, 2 * node1:2 * node1 + 4] += local_Sv
            Mv[2 * node1:2 * node1 + 4, 2 * node1:2 * node1 + 4] += local_Mv

        return Sv, Mv

    def __assemble_longitudinal_matrices(self, local_Sl, local_Ml):
        Sl = np.zeros((self.num_nodes, self.num_nodes))
        Ml = np.zeros((self.num_nodes, self.num_nodes))

        for element in self.elements:
            node1 = element[0]
            # Add the local matrix contributions to the global matrices
            Sl[ node1: node1 + 2, node1: node1 + 2] += local_Sl
            Ml[ node1: node1 + 2, node1: node1 + 2] += local_Ml

        return Sl, Ml

    def get_node_pos(self):
        """
        :return: the list of all nodes' coordinates (x_i,y_i) with the first node setting to be (0,0)
        """
        coordinates = []
        for position in self.nodes:
            x_global = position * np.cos(self.angle)
            y_global = position * np.sin(self.angle)
            coordinates.append((x_global,y_global))
        return coordinates


    def add_force(self, load, load_type:LoadType, load_angle = np.pi/2, pos_flag = "idx"):
        """
        Applies a force or distributed load to the beam and updates the global force vector.

        Parameters:
            load (tuple or function): The load to apply. For point loads and moments, this is a tuple
                                      (node index or position, magnitude). For distributed loads, this is a function of position.
            load_type (LoadType): The type of load, which can be a distributed load (q), point force (F), or moment (M).
            load_angle (float): The anticlockwise angle from the beam to the load, 0 means parallel.
            pos_flag: "pos" or "idx". "pos" means the detailed position on the beam, "idx" means the index of the node being applied

        """
        # Initialization
        ql = np.zeros((self.num_nodes, 1))
        qt = np.zeros((2 * self.num_nodes, 1))

        if load_type == LoadType.q:
            # Apply distributed load over each element
            for idx, xstart in enumerate(self.nodes[0:-1]):
                pt, pl = LocalElement2D.equal_force(
                    load,
                    load_type,
                    xstart,
                    self.element_len,
                    load_angle
                )

                ql[idx: idx + 2] += pl[:,None]
                qt[2 * idx:2 * idx + 4] += pt[:,None]

        elif load_type == LoadType.F:
            # Apply point force
            f_pos, f_val = load
            if pos_flag == "idx":
                qt[2 * f_pos] = f_val * np.sin(load_angle)
                ql[f_pos] = f_val * np.cos(load_angle)

            elif pos_flag == "pos":
                idx = int(f_pos / self.element_len)
                # check whether the position of force is beyond the range of beam
                if f_pos > self.L or f_pos < 0:
                    Warning("force applied beyond the beam, ignored", f_pos)

                # the force is directly applied on nodes
                elif f_pos in self.nodes:
                    qt[2 * f_pos] = f_val * np.sin(load_angle)
                    ql[f_pos] = f_val * np.cos(load_angle)

                # if the force is not applied on nodes, computing its equivalent nodal force
                else:
                    pt, pl = LocalElement2D.equal_force(
                        load,
                        load_type,
                        idx * self.element_len,
                        self.element_len,
                        load_angle
                    )

                    ql[idx: idx + 2] += pl[:,None]
                    qt[2 * idx:2 * idx + 4] += pt[:,None]

            else:
                raise ValueError("pos_flag should be either 'idx' or 'pos'")

        elif load_type == LoadType.M:
            # Apply moment
            m_pos, m_val = load
            if pos_flag == "idx":
                qt[2 * m_pos + 1] = m_val

            elif pos_flag == "pos":
                idx = int(m_pos / self.element_len)
                # check whether the position of moment is beyond the range of beam
                if m_pos > self.L or m_pos < 0:
                    raise Warning("moment applied beyond the beam, ignored", m_pos)

                # the moment is directly applied on nodes
                elif m_pos in self.nodes:
                    qt[2 * idx + 1] = m_val

                # if the moment is not applied on nodes, computing its equivalent nodal force
                else:
                    pt, _ = LocalElement2D.equal_force(
                        load,
                        load_type,
                        idx * self.element_len,
                        self.element_len,
                        load_angle
                    )

                    qt[2 * idx:2 * idx + 4] += pt[:,None]

            else:
                raise ValueError("pos_flag should be either 'idx' or 'pos'")

        else:
            raise ValueError("Type of load should be either 'q', 'F', 'M' ")

        self.q[:self.num_nodes] += ql
        self.q[self.num_nodes:] += qt


if __name__ == "__main__":
    # Example of creating a Beam object and printing the global matrices
    beam2D = Beam2D(length=3, young_module=1, area=1, density=1, moment_inertia=1, num_elements=3, angle= -0.5 * np.pi)

    # Print the global stiffness matrix
    print("the global stiffness matrix")
    print(beam2D.S)

    # Print the global mass matrix (rounded to 2 decimal places)
    print("the global mass matrix")
    print(np.round(beam2D.M, 2))

    # Print the coordinate of every node in the beam
    print("the coordinates")
    coordinates = beam2D.get_node_pos()
    print(coordinates)

    # beam2D.add_force((-1,100), LoadType.F, load_angle = np.pi/2, pos_flag = "idx")
    # print(beam2D.q)
    # beam2D.add_force((-1, 100), LoadType.M, load_angle = np.pi/2, pos_flag = "idx")
    # print(beam2D.q)
    beam2D.add_force(lambda x: uniform_load_function(x,value=-1), LoadType.q, load_angle= 0, pos_flag="idx")
    beam2D.add_force(uniform_load_function, LoadType.q, load_angle= -np.pi/2, pos_flag="idx")
    print(beam2D.q)