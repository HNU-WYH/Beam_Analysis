import numpy as np
import matplotlib.pyplot as plt

from src.beam import Beam2D
from src.utils.local_matrix import LocalElement2D
from src.utils.newmark import NewMark
from config import LoadType, ConstraintType, SolvType, ConnectionType


class FrameworkFEM:
    def __init__(self):
        # framework composition
        self.beams = []  # List of Beam2D objects in the beam structure
        self.nodes = []  # List of corresponding nodes index [[1,...,n], [n+1, ... , m], ... ]
        self.coordinates = []  # list of global coordinates of nodes [(x1,y1),(x2,y2), ..., (x_{n+m},y_{n+m}), ...]
        self.constraints = []  # List to store applied constraints with (node_idx, cons_type)
        self.num_nodes = 0
        self.num_elements = 0
        self.connections = {ConnectionType.Fix: [],
                            ConnectionType.Hinge: []}  # Dictionary with connections between beams

        # properties for FEM analysis
        self.S_global = None  # Global stiffness Matrix for framework
        self.M_global = None  # Global stiffness Matrix for framework
        self.forces = None  # the global equivalent nodal force vector
        self.stsol = None  # Static solution vector (to be computed)
        self.dysol = None  # Dynamic solution vector (to be computed)

    def add_beam(self, beam: Beam2D):
        # add beam2D object to framework
        self.beams.append(beam)

        # allocate the global node index to every nodes in added beam
        self.nodes.append([x + self.num_nodes for x in np.arange(beam.num_nodes)])
        self.num_nodes += beam.num_nodes
        self.num_elements += beam.num_elements

    def add_connection(self, beam1: Beam2D, beam2: Beam2D, connect_node_pair, connection_type: ConnectionType):
        pass

    def apply_force(self, beam: Beam2D, load, load_type):
        pass

    def add_constraint(self, beam: Beam2D, node, value, constraint_Type: ConstraintType):
        pass

    def visualize(self):
        pass

    def _assemble_frame_matrices(self):
        pass

    def solve(self, num_steps=None, tau=None, sol_type=SolvType.STATIC, beta=0.25, gamma=0.5):
        pass
