import numpy as np
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

if __name__ == "__main__":
    # Example of creating a Beam object and printing the global matrices
    beam2D = Beam2D(length=3, young_module=1, area=1, density=1, moment_inertia=1, num_elements=3, angle= -0.25 * np.pi)

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
