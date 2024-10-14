import math
import unittest

import numpy as np
from numpy.lib.function_base import angle

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
        beam_2 = Beam2D(length, E, A, rho, I, num_elements, angle=-math.pi / 4)

        # Initialize FEM Framework model
        frame_work = FrameworkFEM()

        # Add beams to the framework
        frame_work.add_beam(beam_1)
        frame_work.add_beam(beam_2)

        # Add connections between beams
        frame_work.add_connection(beam_1, beam_2, (-1, 0), ConnectionType.Hinge)

        # Apply a force of 50 at position -1 (node) (the right end of beam_1)
        frame_work.add_force(beam_1, (-1, 500000), LoadType.F)

        # Add constraints
        frame_work.add_constraint(beam_1, 0, 0, ConstraintType.DISPLACEMENT)
        frame_work.add_constraint(beam_1, 0, 0, ConstraintType.AXIAL)
        frame_work.add_constraint(beam_1, 0, 0, ConstraintType.ROTATION)
        frame_work.add_constraint(beam_2, -1, 0, ConstraintType.DISPLACEMENT)
        frame_work.add_constraint(beam_2, -1, 0, ConstraintType.AXIAL)

        # assemble the global matrices
        frame_work.assemble_frame_matrices()

        # Solve the static system
        frame_work.solv()

        # Solve the dynamic system
        frame_work.solv(tau=0.1, num_steps=200, sol_type=SolvType.DYNAMIC)

        # Visualize the solution
        frame_work.visualize()
        frame_work.visualize(sol_type=SolvType.DYNAMIC)

    def portal_frame_test(self):
        # Initialize two simple beam with 50 nodes and length 5.0
        length = 3.0
        num_elements = 30
        E, I, rho, A = 210 * 10 ** 9, 1 * 10 ** (-6), 7800, 10 ** (-4)

        # Initialize the beam
        beam_1 = Beam2D(length, E, A, rho, I, num_elements, angle = math.pi / 2)
        beam_2 = Beam2D(length, E, A, rho, I, num_elements)
        beam_3 = Beam2D(length, E, A, rho, I, num_elements, angle= - math.pi / 2)

        # Initialize FEM Framework model
        frame_work = FrameworkFEM()

        # Add beams to the framework
        frame_work.add_beam(beam_1)
        frame_work.add_beam(beam_2)
        frame_work.add_beam(beam_3)

        # Add connections between beams
        frame_work.add_connection(beam_1, beam_2, (-1, 0), ConnectionType.Fix)
        frame_work.add_connection(beam_2, beam_3, (-1, 0), ConnectionType.Fix)

        # Apply a force of 50 at position -1 (node) (the right end of beam_1)
        frame_work.add_force(beam_2, (0, 500000), LoadType.F, load_angle = 0)

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
        frame_work.solv()
        frame_work.solv(tau=0.1, num_steps=200, sol_type=SolvType.DYNAMIC)

        frame_work.visualize(SolvType.DYNAMIC)



    def Single_beam(self):
        # Initialize two simple beam with 50 nodes and length 5.0
        length = 3.0
        num_elements = 30
        E, I, rho, A = 210 * 10 ** 9, 1 * 10 ** (-6), 7800, 10 ** (-4)

        # Initialize the beam
        beam_1 = Beam2D(length, E, A, rho, I, num_elements, angle=math.pi / 2)

        # Initialize FEM Framework model
        frame_work = FrameworkFEM()

        # Add beams to the framework
        frame_work.add_beam(beam_1)

        # Apply a force of 50 at position -1 (node) (the right end of beam_1)
        frame_work.add_force(beam_1, (-1, 500000), LoadType.F, load_angle= np.pi/4)

        # Add constraints
        frame_work.add_constraint(beam_1, 0, 0, ConstraintType.DISPLACEMENT)
        frame_work.add_constraint(beam_1, 0, 0, ConstraintType.AXIAL)
        frame_work.add_constraint(beam_1, 0, 0, ConstraintType.ROTATION)

        # assemble the global matrices
        frame_work.assemble_frame_matrices()

        # Solve the static system
        frame_work.solv()
        print("coordinates: ", frame_work.coordinates)
        frame_work.visualize()

    def test_2(self):
        # Initialize two simple beam with 50 nodes and length 5.0
        length = 5.0
        num_elements = 50
        E, I, rho, A = 210 * 10 ** 9, 1 * 10 ** (-6), 7800, 10 ** (-4)

        # Initialize the beam
        beam_1 = Beam2D(length, E, A, rho, I, num_elements, angle=math.pi / 6)
        beam_2 = Beam2D(length, E, A, rho, I, num_elements, angle=-math.pi / 7)

        # Initialize FEM Framework model
        frame_work = FrameworkFEM()

        # Add beams to the framework
        frame_work.add_beam(beam_1)
        frame_work.add_beam(beam_2)

        # Add connections between beams
        frame_work.add_connection(beam_1, beam_2, (-1, 0), ConnectionType.Fix)

        # Apply a force of 5000 at position -1 (node) (the right end of beam_1)
        frame_work.add_force(beam_1, (-1, -5000), LoadType.F)

        # Add constraints
        frame_work.add_constraint(beam_1, 0, 0, ConstraintType.DISPLACEMENT)
        frame_work.add_constraint(beam_1, 0, 0, ConstraintType.ROTATION)
        frame_work.add_constraint(beam_1, 0, 0, ConstraintType.AXIAL)

        # assemble the global matrices
        frame_work.assemble_frame_matrices()

        # Solve the static system
        frame_work.solv()

        # Solve the dynamic system
        frame_work.solv(tau=0.1, num_steps=200, sol_type=SolvType.DYNAMIC)

        # Visualize the solution
        frame_work.visualize()
        frame_work.visualize(sol_type=SolvType.DYNAMIC)

    def test_3(self):
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

        # Apply a force of 50 at position -1 (node) (the right end of beam_1)
        frame_work.add_force(beam_1, (-1, 5000), LoadType.F, load_angle= 0)

        # Add constraints
        frame_work.add_constraint(beam_1, 0, 0, ConstraintType.DISPLACEMENT)
        frame_work.add_constraint(beam_1, 0, 0, ConstraintType.ROTATION)
        frame_work.add_constraint(beam_1, 0, 0, ConstraintType.AXIAL)

        # assemble the global matrices
        frame_work.assemble_frame_matrices()

        # Solve the static system
        frame_work.solv()

        # Solve the dynamic system
        frame_work.solv(tau=0.1, num_steps=200, sol_type=SolvType.DYNAMIC)

        # Visualize the solution
        frame_work.visualize()
        frame_work.visualize(sol_type=SolvType.DYNAMIC)


if __name__ == '__main__':
    unittest.main()
