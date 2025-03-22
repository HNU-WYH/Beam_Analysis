import math
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from src.beam import Beam2D
from src.utils.eigs import EigenMethod
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
        self.connections = []  # List with connections between beams with (beam1, beam2, connected_node_pair(local idx), connect_type)
        self.constraints = []  # List to store applied constraints with (beam, node, value, constraint_Type)
        self.forces = []  # List to store applied forces with (beam_idx, force, force_type)

        # properties for FEM indexing
        self.nodes = []  # List of corresponding nodes index [[1,...,n], [n+1, ... , m], ... ]
        self.coordinates = []  # list of global coordinates of nodes [(x1,y1),(x2,y2), ..., (x_{n+m},y_{n+m}), ...]
        self.num_nodes = 0
        self.num_elements = 0

        # Properties for FEM preprocessing
        self.S = None
        self.M = None
        self.q = None
        self.ddy0 = None

        # properties for FEM analysis
        self._activate = False  # flag to activate the FEM analysis
        self._connections_global = []  # List with connections between beams with (global_node_index1,
        # global_node_index2, connection_type)
        self._constraints_global = []  # List to store applied constraints globally with (global_node_index, value,
        # constraint_Type)
        self._force_global = []  # List to store applied forces globally with (global_node_index, force, force_type)
        self.S_global = None  # Global stiffness Matrix for framework
        self.M_global = None  # Global stiffness Matrix for framework
        self.q_global = None  # Initialize the global equivalent nodal force vector with zeros
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

    def add_force(self, beam: Beam2D, load, load_type, load_angle = np.pi/2):
        """
        Add a force to the beam, computing the equivalent nodal force vector for the loaded beam

        Parameters:
            beam (Beam2D): Beam2D object to which the force is applied
            load (tuple): The load to apply.
                For point loads and moments, this is a tuple (node (local), magnitude).
                For distributed loads, this is a function of position. Use `0` as the left end and `-1` as the last end.
            load_angle (float): The angle to apply to the force.
            load_type (LoadType): The type of load, which can be a distributed load (q), point force (F), or moment (M).
        """
        if not self._activate:
            self.forces.append((beam, load, load_type))
            # 这里调用了新的add_force方法，直接计算出该beam对应的q，存储在beam.q里
            # 这个q会在之后active的时候，得到整个framework的q
            # list存储的是引用，因此self.beams[idx].q 也会相应改变
            beam.add_force(load,load_type,load_angle)
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

    def _convert_global_connections(self):
        """
        Converge the connections to the global index
        """
        for connection in self.connections:
            beam1, beam2, connect_node_pair, connection_type = connection
            node1, node2 = connect_node_pair
            global_node1 = self.nodes[self.beams.index(beam1)][node1]
            global_node2 = self.nodes[self.beams.index(beam2)][node2]
            self._connections_global.append((global_node1, global_node2, connection[3]))

    def _convert_global_constraints(self):
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

    def __generate_local_coordinates(self, initial_position, beam: Beam2D, reverse: bool):
        """
        Generate the local coordinates for the beam
        The local coordinates are a list of tuples, where each tuple contains the x and y coordinates of the corresponding node.
        The local coordinates are used to assemble the local mass and stiffness matrices for the beam.

        Parameters:
            initial_position (tuple): The initial position of the beam in the global coordinate system
            beam (Beam2D): The Beam2D object for which to generate the local coordinates
            reverse (bool): A flag to indicate whether to generate the local coordinates in reverse order
        """
        x0, y0 = initial_position
        angle = beam.angle
        local_coordinates = []
        nodes = self.nodes[self.beams.index(beam)]
        unit_length = beam.element_len
        if not reverse:
            for node in range(beam.num_nodes):
                x_new = x0 + node * unit_length * np.cos(angle)
                y_new = y0 + node * unit_length * np.sin(angle)
                local_coordinates.append((x_new, y_new))
        else:
            # inverse the direction, from right to left
            for node in range(beam.num_nodes):
                x_new = x0 - node * unit_length * np.cos(angle)
                y_new = y0 - node * unit_length * np.sin(angle)
                local_coordinates.append((x_new, y_new))
            local_coordinates.reverse()
        self.coordinates[nodes[0]:nodes[-1] + 1] = local_coordinates

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
            try:
                connection_pairs[node1].append(node2)
                connection_pairs[node2].append(node1)
            except KeyError:
                raise ValueError(f"Failed to connect node {node2} and {node1}. "
                                 f"Only beams' head or tail nodes can be connected.")
        return connection_pairs

    def _get_beam_from_node(self, node) -> Beam2D | None:
        """
        Get the beam object from the node index

        Parameters:
            node (int): The node index

        Returns:
            Beam2D: The Beam2D object corresponding to the node index
        """

        # 这里我稍微改了一下逻辑，不用根据beam object来寻找它的index
        # 反正都是遍历，这样应该会快一点
        for beam_idx in range(len(self.beams)):
            if node in self.nodes[beam_idx]:
                return self.beams[beam_idx]
        return None

    def _generate_global_coordinates(self):
        """
        This function uses deep first search to generate the global coordinates for the framework.
        """
        # initialize the global coordinates & get connections between global nodes
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
                        is_reverse = False if connected_node == self.nodes[self.beams.index(new_beam)][0] else True
                        self.__generate_local_coordinates(self.coordinates[node], new_beam, is_reverse)
                        dfs(new_beam)

        # 从起始beam开始遍历
        self.__generate_local_coordinates((0, 0), self.beams[0], False)
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

    def _convert_global_force(self):
        """
        Converge the forces to the global index
        倒也没有问题，只是我觉得可以先弄出beam的受力q，再叠加就好了
        因此这里都删掉了，直接在assemble里面，把 concat q就好了
        """
        pass

    def assemble_frame_matrices(self):
        if self._activate:
            return
        self.__activate()

        # Assemble the global mass and stiffness matrices for the framework
        self.beams[0].assemble_matrices()
        S = self.beams[0].S
        M = self.beams[0].M
        q = self.beams[0].q
        for beam2d in self.beams[1:]:
            beam2d.assemble_matrices()
            S = self.__extend_matrix(S, beam2d.S)
            M = self.__extend_matrix(M, beam2d.M)
            q = np.concatenate((q, beam2d.q), axis=0)

        # Assemble the global constraints matrix by list
        C_list = []
        a_list = []

        # add constraints for connection
        # 这一段逻辑有点奇怪，之前我们把local转为了global，这里又转回去了，
        for connection in self._connections_global:
            node1, node2, connection_type = connection
            beam1 = self._get_beam_from_node(node1)
            beam2 = self._get_beam_from_node(node2)
            beam1_start = self.nodes[self.beams.index(beam1)][0]
            beam2_start = self.nodes[self.beams.index(beam2)][0]
            node1_local = node1 - beam1_start
            node2_local = node2 - beam2_start

            # cos(φ1) v1(L1, t) − sin(φ1) w1(L1, t) − cos(φ2) v2(0, t) + sin(φ2) w2(0, t) = 0
            c1 = np.zeros(S.shape[0])
            c1[3 * beam1_start + node1_local] = math.cos(beam1.angle)
            c1[3 * beam1_start + beam1.num_nodes + 2 * node1_local] = -math.sin(beam1.angle)
            c1[3 * beam2_start + node2_local] = -math.cos(beam2.angle)
            c1[3 * beam2_start + beam2.num_nodes + 2 * node2_local] = math.sin(beam2.angle)
            C_list.append(c1)
            a_list.append(0)

            # sin(φ1) v1(L1, t) + cos(φ1) w1(L1, t) − sin(φ2) v2(0, t) − cos(φ2) w2(0, t) = 0
            c2 = np.zeros(S.shape[0])
            c2[3 * beam1_start + node1_local] = math.sin(beam1.angle)
            c2[3 * beam1_start + beam1.num_nodes + 2 * node1_local] = math.cos(beam1.angle)
            c2[3 * beam2_start + node2_local] = -math.sin(beam2.angle)
            c2[3 * beam2_start + beam2.num_nodes + 2 * node2_local] = -math.cos(beam2.angle)
            C_list.append(c2)
            a_list.append(0)

            if connection_type == ConnectionType.Fix:  # Fix connection，fixed angle
                # w1′ (L1, t) − w2′ (0, t) = 0
                c3 = np.zeros(S.shape[0])
                c3[3 * beam1_start + beam1.num_nodes + 2 * node1_local + 1] = 1
                c3[3 * beam2_start + beam2.num_nodes + 2 * node2_local + 1] = -1
                C_list.append(c3)
                a_list.append(0)

        # add boundary constraints
        for constraint in self._constraints_global:
            node, value, constraint_type = constraint
            beam = self._get_beam_from_node(node)
            beam_start = self.nodes[self.beams.index(beam)][0]
            node_local = node - beam_start
            if constraint_type == ConstraintType.AXIAL:
                # v1(0, t) = value (usually 0)
                c1 = np.zeros(S.shape[0])
                c1[3 * beam_start + node_local] = 1
                C_list.append(c1)
                a_list.append(value)

            elif constraint_type == ConstraintType.DISPLACEMENT:
                # w1(0, t) = 0
                c2 = np.zeros(S.shape[0])
                c2[3 * beam_start + beam.num_nodes + 2 * node_local] = 1
                C_list.append(c2)
                a_list.append(value)

            elif constraint_type == ConstraintType.ROTATION:
                # w1′ (0,t) = 0
                c3 = np.zeros(S.shape[0])
                c3[3 * beam_start + beam.num_nodes + 2 * node_local + 1] = 1
                C_list.append(c3)
                a_list.append(value)

        # get the matrices before add constraints & forces
        self.S = S
        self.M = M
        self.q = q

        # extend the global matrix and add constraints matrix into global stiffness matrix
        self.S_global = self.__extend_matrix(S, np.zeros((len(C_list), len(C_list))))
        self.M_global = self.__extend_matrix(M, np.zeros((len(C_list), len(C_list))))
        self.q_global = np.concatenate([q, np.array(a_list)])

        for i, c in enumerate(C_list):
            self.S_global[:len(c), S.shape[0] + i] = c[:]
            self.S_global[S.shape[0] + i, :len(c)] = c[:]

    def __activate(self):
        """
        Activates the FEM analysis for the beam structure and locks the framework for further modifications.
        """
        self._activate = True
        self._generate_global_index()
        self._convert_global_connections()
        self._convert_global_constraints()
        self._generate_global_coordinates()
        self._convert_global_force()

    def initial_acceleration(self, x0):
        if np.isscalar(x0):
            x0 = np.full(shape=self.q.shape, fill_value=x0)
        ddx0 = np.linalg.solve(self.M, self.q - self.S @ x0 )
        return ddx0

    def solv(self, tau=None, num_steps=None, x0 = 0, dx0 = 0, sol_type=SolvType.STATIC, beta=0.25, gamma=0.5):
        """
        Solves the system of equations for the beam under the applied forces and constraints.


        Parameters:
            num_steps (int): Number of time steps for dynamic solution
            tau (float): Time step for dynamic solution
            sol_type (SolvType): Type of solution, either static, dynamic or eigen
            beta (float): Beta parameter for Newmark's method
            gamma (float): Gamma parameter for Newmark's method
            x0 (float): Initial position of the beam
            dx0 (float): Initial velocity of the beam

        Raises:
            Exception: If the solution type is incorrectly defined
        """
        N, K = self.S.shape[0], self.S_global.shape[0] - self.S.shape[0]

        if sol_type == SolvType.STATIC:
            # Static solution
            self.stsol = np.linalg.solve(self.S_global, self.q_global)

        elif sol_type == SolvType.DYNAMIC or sol_type == SolvType.EIGEN:
            # calculating the initial condition
            if np.isscalar(x0) :
                x0 = np.full(shape=self.q.shape, fill_value=x0)
            if np.isscalar(dx0):
                dx0 = np.full(shape=self.q.shape, fill_value=dx0)

            if sol_type == SolvType.DYNAMIC:
                # calculating the initial acceleration
                ddx0 = self.initial_acceleration(x0)

                # expand initial conditions
                x0 = np.concatenate([x0, np.zeros(K)])
                dx0 = np.concatenate([dx0, np.zeros(K)])
                ddx0 = np.concatenate([ddx0, np.zeros(K)])

                # Dynamic solution using Newmark's method
                newmark_solver = NewMark(tau, num_steps, beta, gamma)
                self.dysol, _, _ = newmark_solver.solve(self.M_global, self.S_global, self.q_global, x0, dx0, ddx0)

            else:
                # Dynamic solution using Eigenvalue method without external forces

                eigen = EigenMethod(N, K)
                res = eigen.solve(self.M_global, self.S_global, x0, dx0)
                self.dysol = np.array([res(t) for t in np.arange(0, tau * num_steps, tau)])

        else:
            raise Exception("Wrong defined type of solution")

    def visualize(self, sol_type=SolvType.STATIC, title: str = 'Deformed Structure', save_flag = False):
        x = [coord[0] for coord in self.coordinates]
        y = [coord[1] for coord in self.coordinates]

        if sol_type == SolvType.STATIC:
            x_st = x.copy()
            y_st = y.copy()
            for beam in self.beams:
                angle = beam.angle
                nodes = self.nodes[self.beams.index(beam)]
                beam_start = nodes[0]
                nodes_local = [node - beam_start for node in nodes]
                w = self.stsol[3 * beam_start + beam.num_nodes: 3 * beam_start + beam.num_nodes + 2 * nodes_local[-1] + 1: 2]
                v = self.stsol[3 * beam_start: 3 * beam_start + nodes_local[-1] + 1]
                # x = x0 + w * sin(angle) - v * cos(angle)
                x_st[nodes[0]:nodes[-1] + 1] += np.sin(angle) * w - np.cos(angle) * v
                # y = y0 - w * cos(angle) - v * sin(angle)
                y_st[nodes[0]:nodes[-1] + 1] -= np.cos(angle) * w + np.sin(angle) * v

            # plot the original beam (self.coordinates) and the static deformed beam (x, y)
            plt.plot(x, y, 'b-', label='Original Beam')
            plt.plot(x_st, y_st, 'r-', label='Deformed Beam')
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')
            plt.title(title)
            plt.legend()
            plt.show()

        elif sol_type == SolvType.DYNAMIC or sol_type == SolvType.EIGEN:
            self.dysol = self.dysol.T
            import matplotlib
            matplotlib.use("WebAgg", force=True)
            x_d = np.zeros((len(x), self.dysol.shape[1]))
            y_d = np.zeros((len(y), self.dysol.shape[1]))
            for i in range(self.dysol.shape[1]):
                x_d[:, i] = x
                y_d[:, i] = y
            for beam in self.beams:
                angle = beam.angle
                nodes = self.nodes[self.beams.index(beam)]
                beam_start = nodes[0]
                nodes_local = [node - beam_start for node in nodes]
                w_d = self.dysol[
                      3 * beam_start + beam.num_nodes: 3 * beam_start + beam.num_nodes + 2 * nodes_local[-1] + 1: 2, :]
                v_d = self.dysol[3 * beam_start: 3 * beam_start + nodes_local[-1] + 1, :]
                # x = x0 + w * sin(angle) - v * cos(angle)
                dx_d = np.sin(angle) * w_d - np.cos(angle) * v_d
                # y = y0 - w * cos(angle) - v * sin(angle)
                dy_d = -np.cos(angle) * w_d - np.sin(angle) * v_d
                # transform the complex values to real values
                dx_d = np.real(dx_d)
                dy_d = np.real(dy_d)
                x_d[nodes[0]:nodes[-1] + 1,] += dx_d[:,]
                y_d[nodes[0]:nodes[-1] + 1,] += dy_d[:,]

            # Create a figure for the dynamic animation
            fig, ax = plt.subplots()
            line, = ax.plot([], [], 'r-', label='Dynamic Solution')

            ax.plot(x, y, 'b-', label='Original Beam')
            ax.set_xlim(np.min(x_d), np.max(x_d))
            ax.set_ylim(np.min(y_d), np.max(y_d))
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(title)
            ax.legend(loc="upper left")

            # Update function for animation
            def update(frame):
                line.set_data(x_d[:, frame], y_d[:, frame])
                return line,

            # Create animation
            ani = animation.FuncAnimation(fig, update, frames=self.dysol.shape[1], blit=True)


            # Save the animation as a GIF
            if save_flag:
                os.makedirs(r"./output", exist_ok=True)
                ani.save(r'./output/dynamic_solution.gif', writer='imagemagick', fps=30)

            plt.show()

