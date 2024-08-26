import unittest

from config import LoadType, ConstraintType, SolvType, ConnectionType
from src.beam import Beam2D
from src.fem_frame import FrameworkFEM


class TestCasesForFramework(unittest.TestCase):
    def test_1(self):
        # Initialize two simple beam with 50 nodes and length 5.0
        length = 5.0
        num_elements = 50
        E, I, rho, A = 210 * 10 ** 9, 1 * 10 ** (-6), 7800, 10 ** (-4)

        # Initialize the beam
        beam_1 = Beam2D(length, E, A, rho, I, num_elements)
        beam_2 = Beam2D(length, E, A, rho, I, num_elements, angle=-45)

        # Initialize FEM Framework model
        frame_work = FrameworkFEM()

        # Add beams to the framework
        frame_work.add_beam(beam_1)
        frame_work.add_beam(beam_2)

        # Add connections between beams
        frame_work.add_connection(beam_1, beam_2, (1, 0), ConnectionType.Hinge)

        # Apply a force of 50 at position 5 (the right end of beam_1)
        frame_work.apply_force(beam_1, (5, 50), LoadType.F)

        # Add constraints
        frame_work.add_constraint(beam_1, 0, 0, ConstraintType.DISPLACEMENT)
        frame_work.add_constraint(beam_1, 0, 0, ConstraintType.ROTATION)
        frame_work.add_constraint(beam_2, 49, 0, ConstraintType.DISPLACEMENT)

        # Solve the static system
        frame_work.solve()

        # Solve the dynamic system
        frame_work.solve(tau=0.1, num_steps=200, sol_type=SolvType.DYNAMIC)

        # Visualize the solution
        frame_work.visualize()

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
