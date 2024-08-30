import math

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
        forces (list): List to store applied forces with (beam_idx, force, force_type)
        nodes (list): List of corresponding nodes index [[1,...,n], [n+1, ... , m], ... ]
        coordinates (list): List of global coordinates of nodes [(x1,y1),(x2,y2), ..., (x_{n+m},y_{n+m}), ...]
        num_nodes (int): Number of nodes in the framework
        num_elements (int): Number of elements in the framework
        S_global (np.ndarray): Global stiffness Matrix for framework
        M_global (np.ndarray): Global stiffness Matrix for framework
        q (np.ndarray): Global equivalent nodal force vector
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
        self._activate = False  # flag to activate the FEM analysis
        self._connections_global = []  # List with connections between beams with (global_node_index1,
        # global_node_index2, connection_type)
        self._constraints_global = []  # List to store applied constraints globally with (global_node_index, value,
        # constraint_Type)
        self._force_global = []  # List to store applied forces globally with (global_node_index, force, force_type)
        self.S_global = None  # Global stiffness Matrix for framework
        self.M_global = None  # Global stiffness Matrix for framework
        self.q = None  # Initialize the global equivalent nodal force vector with zeros
        self.stsol = None  # Static solution vector (to be computed)
        self.dysol = None  # Dynamic solution vector (to be computed)

    def add_beam(self, beam: Beam2D):
        """
        Add a beam to the framework

        Parameters:
            beam (Beam2D): Beam2D object to be added to the framework
        """
        if not self._activate:
            self.beams.append(beam)
        else:
            raise Exception("Cannot add beam after activating the FEM analysis")

    def add_connection(self, beam1: Beam2D, beam2: Beam2D, connect_node_pair, connection_type: ConnectionType):
        """
        Add a connection between two beams

        Parameters:
            beam1 (Beam2D): First Beam2D object to be connected
            beam2 (Beam2D): Second Beam2D object to be connected
            connect_node_pair (tuple): The node (local) pair of the two beams to be connected `(node of beam1, node of beam2)`.
                The node could be 0 (left end) or -1 (right end).
            connection_type (ConnectionType): The type of connection between the beams
        """
        if not self._activate:
            self.connections.append((beam1, beam2, connect_node_pair, connection_type))
        else:
            raise Exception("Cannot add connection after activating the FEM analysis")

    def add_force(self, beam: Beam2D, load, load_type):
        """
        Add a force to the beam

        Parameters:
            beam (Beam2D): Beam2D object to which the force is applied
            load (tuple): The load to apply. For point loads and moments, this is a tuple (node (local), magnitude).
                For distributed loads, this is a function of position. Use `0` as the left end and `-1` as the last end.
            load_type (LoadType): The type of load, which can be a distributed load (q), point force (F), or moment (M).
        """
        if not self._activate:
            self.forces.append((beam, load, load_type))
        else:
            raise Exception("Cannot add force after activating the FEM analysis")

    def add_constraint(self, beam: Beam2D, node, value, constraint_Type: ConstraintType):
        """
        Add a constraint to the beam

        Parameters:
            beam (Beam2D): Beam2D object to which the constraint is applied
            node (int): The node index (local) to which the constraint is applied. You can use `-1` as the last node.
            value (float): The value of the constraint
            constraint_Type (ConstraintType): The type of constraint, which can be a displacement or rotation
        """
        if not self._activate:
            self.constraints.append((beam, node, value, constraint_Type))
        else:
            raise Exception("Cannot add constraint after activating the FEM analysis")

    def _converge_global_connections(self):
        """
        Converge the connections to the global index
        """
        for connection in self.connections:
            beam1, beam2, connect_node_pair, connection_type = connection
            node1, node2 = connect_node_pair
            global_node1 = self.nodes[self.beams.index(beam1)][node1]
            global_node2 = self.nodes[self.beams.index(beam2)][node2]
            self._connections_global.append((global_node1, global_node2, connection[3]))

    def _converge_global_constraints(self):
        """
        Converge the constraints to the global index
        """
        for i in range(len(self.constraints)):
            beam, node, value, constraint_Type = self.constraints[i]
            global_node = self.nodes[self.beams.index(beam)][node]
            self._constraints_global.append((global_node, value, constraint_Type))

    def _generate_global_index(self):
        """
        Generate the global index for the framework
        The global index is a list of lists, where each list contains the indices of the nodes in the corresponding beam.
        The global index is used to assemble the global mass and stiffness matrices for the framework.
        """
        for i in range(len(self.beams)):
            self.nodes.append(list(range(self.num_nodes, self.num_nodes + self.beams[i].num_nodes)))
            self.num_nodes += self.beams[i].num_nodes
            self.num_elements += self.beams[i].num_elements

    def __generate_local_coordinates(self, initial_position, beam: Beam2D):
        """
        Generate the local coordinates for the beam
        The local coordinates are a list of tuples, where each tuple contains the x and y coordinates of the corresponding node.
        The local coordinates are used to assemble the local mass and stiffness matrices for the beam.

        Parameters:
            initial_position (tuple): The initial position of the beam in the global coordinate system
            beam (Beam2D): The Beam2D object for which to generate the local coordinates
        """
        x0, y0 = initial_position
        angle = beam.angle
        local_coordinates = []
        nodes = self.nodes[self.beams.index(beam)]
        unit_length = beam.element_len
        for node in range(beam.num_nodes):
            x_new = x0 + node * unit_length * np.cos(angle)
            y_new = y0 + node * unit_length * np.sin(angle)
            local_coordinates.append((x_new, y_new))
        self.coordinates[nodes[0]:nodes[-1]] = local_coordinates
        if len(self.coordinates) > self.num_nodes:
            for i in range(len(self.coordinates) - self.num_nodes):
                self.coordinates.pop()

    def __generate_global_adjacency_connection_list(self):
        """
        Generate the global adjacency connection list for the framework

        Returns:
            dict: A dictionary where the keys are the node indices and the values are lists of the connected node
        """
        connection_pairs = {}
        for index, beam in enumerate(self.beams):
            connection_pairs[self.nodes[index][0]] = []
            connection_pairs[self.nodes[index][-1]] = []
        for connection in self._connections_global:
            node1, node2, connection_type = connection
            connection_pairs[node1].append(node2)
            connection_pairs[node2].append(node1)
        return connection_pairs

    def _get_beam_from_node(self, node) -> Beam2D | None:
        """
        Get the beam object from the node index

        Parameters:
            node (int): The node index

        Returns:
            Beam2D: The Beam2D object corresponding to the node index
        """
        for beam in self.beams:
            if node in self.nodes[self.beams.index(beam)]:
                return beam
        return None

    def _generate_global_coordinates(self):
        """
        This function uses deep first search to generate the global coordinates for the framework.
        """
        # initialize the global coordinates
        self.coordinates = [(0, 0)] * self.num_nodes
        adjacency_list = self.__generate_global_adjacency_connection_list()
        # beam is a dictionary, the key is the beam name, and the value is the set of two nodes that make up the beam
        visited_beams = set()
        visited_nodes = set()

        def dfs(beam: Beam2D):
            if beam in visited_beams:
                return
            visited_beams.add(beam)

            beam_index = self.beams.index(beam)
            # 获取当前beam的两个node
            nodes = [self.nodes[beam_index][0], self.nodes[beam_index][-1]]

            # 对这两个node进行遍历，找到与之相连的其他nodes
            for node in nodes:
                if node in visited_nodes:
                    continue
                visited_nodes.add(node)

                # 获取与当前node相连的所有其他nodes
                connected_nodes = adjacency_list.get(node, [])

                # 通过这些相连的nodes找到所有相关的beams
                for connected_node in connected_nodes:
                    new_beam = self._get_beam_from_node(connected_node)
                    if new_beam is None:
                        raise Exception("Invalid connection")
                    if new_beam not in visited_beams:
                        self.__generate_local_coordinates(self.coordinates[node], new_beam)
                        dfs(new_beam)

        # 从起始beam开始遍历
        self.__generate_local_coordinates((0, 0), self.beams[0])
        dfs(self.beams[0])

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

    def _converge_global_force(self):
        """
        Converge the forces to the global index
        """
        for force in self.forces:
            beam, load, load_type = force
            global_nodes = self.nodes[self.beams.index(beam)]
            if load_type == LoadType.F:
                local_node, magnitude = load
                node = global_nodes[local_node]
                self._force_global.append((node, magnitude, load_type))
            elif load_type == LoadType.M:
                local_node, magnitude = load
                node = global_nodes[local_node]
                self._force_global.append((node, magnitude, load_type))
            elif load_type == LoadType.q:
                # TODO
                pass
            else:
                raise Exception("Invalid load type")

    def _assemble_frame_matrices(self):
        # Assemble the global mass and stiffness matrices for the framework
        self.beams[0].assemble_matrices()
        S = self.beams[0].S
        M = self.beams[0].M
        for beam2d in self.beams[1:]:
            beam2d.assemble_matrices()
            S = self.__extend_matrix(S, beam2d.S)
            M = self.__extend_matrix(M, beam2d.M)

        # Assemble the global constraints matrix by list
        C_list = []

        # add constraints for connection
        for connection in self._connections_global:
            node1, node2, connection_type = connection
            beam1_angle = self._get_beam_from_node(node1).angle
            beam2_angle = self._get_beam_from_node(node2).angle
            # cos(φ1) v1(L1, t) − sin(φ1) w1(L1, t) − cos(φ2) v2(0, t) + sin(φ2) w2(0, t) = 0
            c1 = np.zeros(S.shape[0])
            c1[3 * node1] = math.cos(beam1_angle)
            c1[3 * node1 + 1] = -math.sin(beam1_angle)
            c1[3 * node2] = -math.cos(beam2_angle)
            c1[3 * node2 + 1] = math.sin(beam2_angle)
            C_list.append(c1)
            # sin(φ1) v1(L1, t) + cos(φ1) w1(L1, t) − sin(φ2) v2(0, t) − cos(φ2) w2(0, t) = 0
            c2 = np.zeros(S.shape[0])
            c2[3 * node1] = math.sin(beam1_angle)
            c2[3 * node1 + 1] = math.cos(beam1_angle)
            c2[3 * node2] = -math.sin(beam2_angle)
            c2[3 * node2 + 1] = -math.cos(beam2_angle)
            C_list.append(c2)
            if connection_type == ConnectionType.Fix:  # Fix connection，fixed angle
                # w1′ (L1, t) − w2′ (0, t) = 0
                c3 = np.zeros(S.shape[0])
                c3[3 * node1 + 2] = 1
                c3[3 * node2 + 2] = -1
                C_list.append(c3)

        # add boundary constraints
        for constraint in self._constraints_global:
            node, value, constraint_type = constraint
            if constraint_type == ConstraintType.DISPLACEMENT:
                c1 = np.zeros(S.shape[0])
                c1[3 * node] = 1
                C_list.append(c1)
                c2 = np.zeros(S.shape[0])
                c2[3 * node + 1] = 1
                C_list.append(c2)
            elif constraint_type == ConstraintType.ROTATION:
                c = np.zeros(S.shape[0])
                c[3 * node + 2] = 1
                C_list.append(c)

        # extend the global matrix and add constraints matrix into global stiffness matrix
        self.S_global = self.__extend_matrix(S, np.zeros((len(C_list), len(C_list))))
        self.M_global = self.__extend_matrix(M, np.zeros((len(C_list), len(C_list))))
        for i, c in enumerate(C_list):
            self.S_global[:len(c), S.shape[0] + i] = c[:]
            self.S_global[S.shape[0] + i, :len(c)] = c[:]
        # apply the forces to the global equivalent nodal force vector
        self.q = np.zeros(self.S_global.shape[0])
        for force in self._force_global:
            node, magnitude, load_type = force
            if load_type == LoadType.q:
                raise Exception(
                    "Distributed load not supported, please activate the FEM analysis before applying the load")
            elif load_type == LoadType.F:
                self.q[3 * node] += magnitude
            elif load_type == LoadType.M:
                self.q[3 * node + 1] = magnitude

    def activate(self):
        """
        Activates the FEM analysis for the beam structure and locks the framework for further modifications.
        """
        self._activate = True
        self._generate_global_index()
        self._converge_global_connections()
        self._converge_global_constraints()
        self._generate_global_coordinates()
        self._converge_global_force()

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
        self.activate()
        self._assemble_frame_matrices()

        if sol_type == SolvType.STATIC:
            # Static solution
            self.stsol = np.linalg.solve(self.S_global, self.q)

        elif sol_type == SolvType.DYNAMIC:
            # Dynamic solution using Newmark's method
            newmark_solver = NewMark(tau, num_steps, beta, gamma)
            init_condis = np.zeros(self.q.shape)
            self.dysol, _, _ = newmark_solver.solve(self.M_global, self.S_global, self.q, init_condis,
                                                    init_condis, init_condis)
        else:
            raise Exception("Wrong defined type of solution")

    def visualize(self):
        pass
