import unittest

from config import LoadType, ConstraintType, SolvType, ConnectionType
from src.beam import Beam2D
from src.fem_frame import FrameworkFEM


def get_beam():
    # Initialize two simple beam with 50 nodes and length 5.0
    length = 5.0
    num_elements = 50
    E, I, rho, A = 210 * 10 ** 9, 1 * 10 ** (-6), 7800, 10 ** (-4)

    # Initialize the beam
    beam_1 = Beam2D(length, E, A, rho, I, num_elements)

    # Initialize FEM Framework model
    frame_work = FrameworkFEM()

    # Add beams to the framework
    frame_work.add_beam(beam_1)

    return frame_work, beam_1


def add_constraint(frame_work: FrameworkFEM, beam_1: Beam2D, _type: str):
    # Add constraints
    if _type == 'cantilever':
        frame_work.add_constraint(beam_1, 0, 0, ConstraintType.DISPLACEMENT)
        frame_work.add_constraint(beam_1, 0, 0, ConstraintType.AXIAL)
        frame_work.add_constraint(beam_1, 0, 0, ConstraintType.ROTATION)
    if _type == 'simple':
        frame_work.add_constraint(beam_1, 0, 0, ConstraintType.DISPLACEMENT)
        frame_work.add_constraint(beam_1, 0, 0, ConstraintType.AXIAL)
        frame_work.add_constraint(beam_1, -1, 0, ConstraintType.DISPLACEMENT)


def activate(frame_work: FrameworkFEM):
    # assemble the global matrices
    frame_work.assemble_frame_matrices()

    # Solve the static system
    frame_work.solv()

    # Solve the dynamic system
    frame_work.solv(tau=0.1, num_steps=200, sol_type=SolvType.DYNAMIC)

    # Visualize the solution
    frame_work.visualize()
    frame_work.visualize(sol_type=SolvType.DYNAMIC)


class cantilevel_beam(unittest.TestCase):

    def test_concentrated_force(self):
        frame_work, beam_1 = get_beam()
        add_constraint(frame_work, beam_1, 'cantilever')
        frame_work.add_force(beam_1, (-1, 50), LoadType.F)
        activate(frame_work)

    def test_distributed_force(self):
        frame_work, beam_1 = get_beam()
        add_constraint(frame_work, beam_1, 'cantilever')
        q = lambda x: 10
        frame_work.add_force(beam_1, q, LoadType.q)
        activate(frame_work)

    def test_free_vibration(self):
        frame_work, beam_1 = get_beam()
        add_constraint(frame_work, beam_1, 'cantilever')
        activate(frame_work)

class simple_supported_beam(unittest.TestCase):

        def test_concentrated_force(self):
            frame_work, beam_1 = get_beam()
            add_constraint(frame_work, beam_1, 'simple')
            frame_work.add_force(beam_1, (-1, 50), LoadType.F)
            activate(frame_work)

        def test_distributed_force(self):
            frame_work, beam_1 = get_beam()
            add_constraint(frame_work, beam_1, 'simple')
            q = lambda x: 10
            frame_work.add_force(beam_1, q, LoadType.q)
            activate(frame_work)

        def test_free_vibration(self):
            frame_work, beam_1 = get_beam()
            add_constraint(frame_work, beam_1, 'simple')
            activate(frame_work)
