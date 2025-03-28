import math
import unittest

from config import LoadType, ConstraintType, SolvType, ConnectionType
from src.beam import Beam2D
from src.fem_frame import FrameworkFEM


def get_beam():
    # Initialize two simple beam with 50 nodes and length 5.0
    length = 5.0
    num_elements = 100
    E, I, rho, A = 210 * 10 ** 9, 36.92 * 10 ** (-6), 42.3, 5383 * 10 ** (-6)

    # Initialize the beam
    beam_1 = Beam2D(length, E, A, rho, I, num_elements, angle=math.pi / 2)
    beam_2 = Beam2D(length, E, A, rho, I, num_elements)
    beam_3 = Beam2D(length, E, A, rho, I, num_elements, angle=-math.pi / 2)

    # Initialize FEM Framework model
    frame_work = FrameworkFEM()

    # Add beams to the framework
    frame_work.add_beam(beam_1)
    frame_work.add_beam(beam_2)
    frame_work.add_beam(beam_3)


    return frame_work, beam_1, beam_2, beam_3


def add_connection(frame_work: FrameworkFEM, beam_1: Beam2D, beam_2: Beam2D, beam_3: Beam2D, _type: str):
    # Add connections
    if _type == 'fix':
        frame_work.add_connection(beam_1, beam_2, (-1, 0), ConnectionType.Fix)
        frame_work.add_connection(beam_2, beam_3, (-1, 0), ConnectionType.Fix)
    if _type == 'hinged':
        frame_work.add_connection(beam_1, beam_2, (-1, 0), ConnectionType.Hinge)
        frame_work.add_connection(beam_2, beam_3, (-1, 0), ConnectionType.Hinge)


def add_constraint(frame_work: FrameworkFEM, beam_1: Beam2D, beam_2: Beam2D, beam_3: Beam2D):
    # Add constraints
    frame_work.add_constraint(beam_1, 0, 0, ConstraintType.DISPLACEMENT)
    frame_work.add_constraint(beam_1, 0, 0, ConstraintType.AXIAL)
    frame_work.add_constraint(beam_1, 0, 0, ConstraintType.ROTATION)
    frame_work.add_constraint(beam_3, -1, 0, ConstraintType.DISPLACEMENT)
    frame_work.add_constraint(beam_3, -1, 0, ConstraintType.AXIAL)
    frame_work.add_constraint(beam_3, -1, 0, ConstraintType.ROTATION)


def activate(frame_work: FrameworkFEM, sol_type: SolvType = SolvType.DYNAMIC, x0 = 0., dx0 = 0., title = None):
    # assemble the global matrices
    frame_work.assemble_frame_matrices()

    # Solve the static system
    frame_work.solv()

    # Solve the dynamic system
    frame_work.solv(tau=0.001, num_steps=200, sol_type=sol_type, x0 = x0, dx0 = dx0)

    # Visualize the solution
    frame_work.visualize(title=title)
    # frame_work.visualize(sol_type=sol_type, title=title)


class fix_joint(unittest.TestCase):

    def test_concentrated_force(self):
        frame_work, beam_1, beam_2, beam_3 = get_beam()
        add_connection(frame_work, beam_1, beam_2, beam_3, 'fix')
        add_constraint(frame_work, beam_1, beam_2, beam_3)
        frame_work.add_force(beam_1, (-1, 1000000), LoadType.F)
        activate(frame_work, title='Fix Joint with Concentrated force')

    def test_distributed_force(self):
        frame_work, beam_1, beam_2, beam_3 = get_beam()
        add_connection(frame_work, beam_1, beam_2, beam_3, 'fix')
        add_constraint(frame_work, beam_1, beam_2, beam_3)
        q = lambda x: 100000
        frame_work.add_force(beam_1, q, LoadType.q)
        activate(frame_work, title='Fix Joint with Distributed force')

    def test_free_vibration(self):
        frame_work, beam_1, beam_2, beam_3 = get_beam()
        add_connection(frame_work, beam_1, beam_2, beam_3, 'fix')
        add_constraint(frame_work, beam_1, beam_2, beam_3)
        activate(frame_work, sol_type=SolvType.DYNAMIC, x0=0.1, dx0=0, title='Fix Joint Free Vibration')

class hinged_joint(unittest.TestCase):

    def test_concentrated_force(self):
        frame_work, beam_1, beam_2, beam_3 = get_beam()
        add_connection(frame_work, beam_1, beam_2, beam_3, 'hinged')
        add_constraint(frame_work, beam_1, beam_2, beam_3)
        frame_work.add_force(beam_1, (-1, 500000), LoadType.F)
        activate(frame_work, title='Hinged Joint with Concentrated force')

    def test_distributed_force(self):
        frame_work, beam_1, beam_2, beam_3 = get_beam()
        add_connection(frame_work, beam_1, beam_2, beam_3, 'hinged')
        add_constraint(frame_work, beam_1, beam_2, beam_3)
        q = lambda x: 100000
        frame_work.add_force(beam_1, q, LoadType.q)
        activate(frame_work, title='Hinged Joint with Distributed force')

    def test_free_vibration(self):
        frame_work, beam_1, beam_2, beam_3 = get_beam()
        add_connection(frame_work, beam_1, beam_2, beam_3, 'hinged')
        add_constraint(frame_work, beam_1, beam_2, beam_3)
        activate(frame_work, sol_type=SolvType.EIGEN, x0=1, dx0=0, title='Hinged Joint Free Vibration')
