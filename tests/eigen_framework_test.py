import math
import unittest

import numpy as np
from numpy.lib.function_base import angle

from config import LoadType, ConstraintType, SolvType, ConnectionType
from src.beam import Beam2D
from src.fem_frame import FrameworkFEM


def portal_frame_test():
    # Initialize two simple beam with 50 nodes and length 5.0
    length = 3.0
    num_elements = 30
    E, I, rho, A = 210 * 10 ** 9, 1 * 10 ** (-6), 7800, 10 ** (-4)

    # Initialize the beam
    beam_1 = Beam2D(length, E, A, rho, I, num_elements, angle=math.pi / 2)
    beam_2 = Beam2D(length, E, A, rho, I, num_elements)
    beam_3 = Beam2D(length, E, A, rho, I, num_elements, angle=- math.pi / 2)

    # Initialize FEM Framework model
    frame_work = FrameworkFEM()

    # Add beams to the framework
    frame_work.add_beam(beam_1)
    frame_work.add_beam(beam_2)
    frame_work.add_beam(beam_3)

    # Add connections between beams
    frame_work.add_connection(beam_1, beam_2, (-1, 0), ConnectionType.Fix)
    frame_work.add_connection(beam_2, beam_3, (-1, 0), ConnectionType.Fix)

    # create initial conditions
    # the initial position & speed of every node
    y0 = 0
    dy0 = 1

    # no external force, since eigenvalue method can only be used in homogenous case

    # Add constraints
    frame_work.add_constraint(beam_1, 0, 0, ConstraintType.DISPLACEMENT)
    frame_work.add_constraint(beam_1, 0, 0, ConstraintType.AXIAL)
    frame_work.add_constraint(beam_1, 0, 0, ConstraintType.ROTATION)

    frame_work.add_constraint(beam_3, -1, 0, ConstraintType.DISPLACEMENT)
    frame_work.add_constraint(beam_3, -1, 0, ConstraintType.AXIAL)
    frame_work.add_constraint(beam_3, -1, 0, ConstraintType.ROTATION)

    # assemble the global matrices
    frame_work.assemble_frame_matrices()

    # Solve the static system
    frame_work.solv(tau=0.1, num_steps=1000, x0 = y0, dx0 = dy0, sol_type=SolvType.DYNAMIC)
    dysol1 = frame_work.dysol

    frame_work.solv(tau=0.1, num_steps=1000, x0 = y0, dx0 = dy0, sol_type=SolvType.EIGEN)
    dysol2 = frame_work.dysol



if __name__ == '__main__':
    portal_frame_test()