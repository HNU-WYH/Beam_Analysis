import numpy as np
import matplotlib.pyplot as plt

from src.beam import Beam2D
from src.utils.local_matrix import LocalElement2D
from src.utils.newmark import NewMark
from config import LoadType, ConstraintType, SolvType, ConnectionType


class FrameworkFEM:
    """
    This class represents the postprocessing of a beam structure, including applying forces, applying constraints,
    and solving the resulting displacement in static & dynamic way.

    Compared with the global mass & stiffness matrices in preprocessing steps, the S & M matrices here are expanded
    to include the constraints, details of which can be found in Chapter 6 of script1_bending_and_fem.pdf.

    Attributes:
        beams (list): List of Beam2D objects in the beam structure
        connections (list): List with connections between beams
        constraints (list): List to store applied constraints with (beam, node, value, constraint_Type)
        forces (list): List to store applied forces with (beam, force, force_type)
        nodes (list): List of corresponding nodes index [[1,...,n], [n+1, ... , m], ... ]
        coordinates (list): list of global coordinates of nodes [(x1,y1),(x2,y2), ..., (x_{n+m},y_{n+m}), ...]
        num_nodes (int): Number of nodes in the framework
        num_elements (int): Number of elements in the framework
        S_global (np.ndarray): Global stiffness Matrix for framework
        M_global (np.ndarray): Global stiffness Matrix for framework
        f_global (np.ndarray): the global equivalent nodal force vector
        stsol (np.ndarray): Static solution vector (to be computed)
        dysol (np.ndarray): Dynamic solution vector (to be computed)
    """

    def __init__(self):
        # framework input properties
        self.beams = []  # List of Beam2D objects in the beam structure
        self.connections = []  # List with connections between beams
        self.constraints = []  # List to store applied constraints with (beam, node, value, constraint_Type)
        self.forces = []  # List to store applied forces with (beam_idx, force, force_type)

        # properties for FEM indexing
        self.nodes = []  # List of corresponding nodes index [[1,...,n], [n+1, ... , m], ... ]
        self.coordinates = []  # list of global coordinates of nodes [(x1,y1),(x2,y2), ..., (x_{n+m},y_{n+m}), ...]
        self.num_nodes = 0
        self.num_elements = 0

        # properties for FEM analysis
        self.S_global = None  # Global stiffness Matrix for framework
        self.M_global = None  # Global stiffness Matrix for framework
        self.f_global = None  # the global equivalent nodal force vector
        self.stsol = None  # Static solution vector (to be computed)
        self.dysol = None  # Dynamic solution vector (to be computed)

    def add_beam(self, beam: Beam2D):
        """
        Add a beam to the framework

        Parameters:
            beam (Beam2D): Beam2D object to be added to the framework
        """
        self.beams.append(beam)

    def add_connection(self, beam1: Beam2D, beam2: Beam2D, connect_node_pair, connection_type: ConnectionType):
        """
        Add a connection between two beams

        Parameters:
            beam1 (Beam2D): First Beam2D object to be connected
            beam2 (Beam2D): Second Beam2D object to be connected
            connect_node_pair (tuple): The node pair of the two beams to be connected `(node of beam1, node of beam2)`.
                The node should be 0 (left end) or 1(right end).
            connection_type (ConnectionType): The type of connection between the beams
        """
        self.connections.append((beam1, beam2, connect_node_pair, connection_type))

    def add_force(self, beam: Beam2D, load, load_type):
        """
        Add a force to the beam

        Parameters:
            beam (Beam2D): Beam2D object to which the force is applied
            load (tuple): The load to apply. For point loads and moments, this is a tuple (position, magnitude).
                For distributed loads, this is a function of position.
            load_type (LoadType): The type of load, which can be a distributed load (q), point force (F), or moment (M).
        """
        self.forces.append((beam, load, load_type))

    def add_constraint(self, beam: Beam2D, node, value, constraint_Type: ConstraintType):
        """
        Add a constraint to the beam

        Parameters:
            beam (Beam2D): Beam2D object to which the constraint is applied
            node (int): The node index to which the constraint is applied
            value (float): The value of the constraint
            constraint_Type (ConstraintType): The type of constraint, which can be a displacement or rotation
        """
        self.constraints.append((beam, node, value, constraint_Type))

    def _generate_global_index(self):
        """
        Generate the global index for the framework
        The global index is a list of lists, where each list contains the indices of the nodes in the corresponding beam.
        The global index is used to assemble the global mass and stiffness matrices for the framework.
        """
        for i in range(len(self.beams)):
            self.nodes.append(list(range(self.num_nodes, self.num_nodes + self.beams[i].num_nodes)))
            self.coordinates.extend(self.beams[i].nodes)
            self.num_nodes += self.beams[i].num_nodes
            self.num_elements += self.beams[i].num_elements

    def __generate_local_coordinates(self, initial_position, beam: Beam2D, beam_index: int):
        """
        Generate the local coordinates for the beam
        The local coordinates are a list of tuples, where each tuple contains the x and y coordinates of the corresponding node.
        The local coordinates are used to assemble the local mass and stiffness matrices for the beam.

        Parameters:
            initial_position (tuple): The initial position of the beam in the global coordinate system
            beam (Beam2D): The Beam2D object for which to generate the local coordinates
        """
        x0, y0 = initial_position
        x1, y1 = beam.nodes[0]
        angle = beam.angle
        local_coordinates = []
        for node in beam.nodes:
            x, y = node
            x_new = x0 + (x - x1) * np.cos(angle) - (y - y1) * np.sin(angle)
            y_new = y0 + (x - x1) * np.sin(angle) + (y - y1) * np.cos(angle)
            local_coordinates.append((x_new, y_new))
        nodes = self.nodes[beam_index]
        self.coordinates[nodes[0] - 1:nodes[-1] - 1] = local_coordinates

    def _generate_global_coordinates(self):
        """
        Generate the global coordinates for the framework
        The global coordinates are a list of tuples, where each tuple contains the x and y coordinates of the corresponding node.
        The global coordinates are used to visualize the solution of the framework.
        """
        pass

    @staticmethod
    def __extend_matrix(matrix1, matrix2) -> np.ndarray:
        """
        Merge two matrices along the diagonal.

        Example:
            M_new = [M_1, 0]
                    [0, M_2]

        Parameters:
            matrix1 (np.ndarray): First matrix to merge
            matrix2 (np.ndarray): Second matrix to merge
        """
        n1, m1 = matrix1.shape
        n2, m2 = matrix2.shape
        M_new = np.zeros((n1 + n2, m1 + m2))
        M_new[:n1, :m1] = matrix1
        M_new[n1:, m1:] = matrix2
        return M_new

    def _assemble_frame_matrices(self):
        self._generate_global_index()
        self._generate_global_coordinates()
        # Assemble the global mass and stiffness matrices for the framework
        self.beams[0].assemble_matrices()
        S = self.beams[0].S
        M = self.beams[0].M
        for beam2d in self.beams[1:]:
            beam2d.assemble_matrices()
            S = self.__extend_matrix(S, beam2d.S)
            M = self.__extend_matrix(M, beam2d.M)
        self.S_global = S
        self.M_global = M
        # Assemble the global constraints matrix

        # expand the global mass and stiffness matrices to include the constraints

        # apply the forces to the global equivalent nodal force vector

        pass

    def solv(self, num_steps=None, tau=None, sol_type=SolvType.STATIC, beta=0.25, gamma=0.5):
        """
        Solves the system of equations for the beam under the applied forces and constraints.

        Parameters:
            num_steps (int): Number of time steps for dynamic solution
            tau (float): Time step for dynamic solution
            sol_type (SolvType): Type of solution, either static or dynamic
            beta (float): Beta parameter for Newmark's method
            gamma (float): Gamma parameter for Newmark's method

        Raises:
            Exception: If the solution type is incorrectly defined
        """
        self._assemble_frame_matrices()

        if sol_type == SolvType.STATIC:
            # Static solution
            self.stsol = np.linalg.solve(self.S_global, self.f_global)

        elif sol_type == SolvType.DYNAMIC:
            # Dynamic solution using Newmark's method
            newmark_solver = NewMark(tau, num_steps, beta, gamma)
            init_condis = np.zeros(self.f_global.shape)
            self.dysol, _, _ = newmark_solver.solve(self.M_global, self.S_global, self.f_global, init_condis,
                                                    init_condis, init_condis)
        else:
            raise Exception("Wrong defined type of solution")

    def visualize(self):
        pass
