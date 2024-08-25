import numpy as np
import matplotlib.pyplot as plt

from src.beam import beam2D, Beam2D
from src.utils.local_matrix import LocalElement2D
from src.utils.newmark import NewMark
from config import LoadType, ConstraintType, SolvType


class FrameworkFEM:
    def __init__(self):
        pass

    def add_beam(self, beam: Beam2D, joint_coordinate, connection_type):
        pass

    def apply_force(self):
        pass

    def add_constraint(self):
        pass

    def visualize(self):
        pass

    def assemble_frame_matrices(self):
        pass

    def solve(self):
        pass
