import math
import matplotlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from config import LoadType, ConstraintType, SolvType, ConnectionType
from src.beam import Beam2D
from src.fem_frame import FrameworkFEM

def portal_frame_staic_test():
    # Initialize two simple beam with 50 nodes and length 5.0
    length = 5.0
    num_elements = 100
    E, I, rho, A = 210 * 10 ** 9, 36.92 * 10 ** (-6), 42.3, 5383 * 10 ** (-6)

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
    frame_work.add_connection(beam_1, beam_2, (-1, 0), ConnectionType.Hinge)
    frame_work.add_connection(beam_2, beam_3, (-1, 0), ConnectionType.Hinge)

    # Apply a force of 50 at position -1 (node) (the right end of beam_1)
    frame_work.add_force(beam_1, (-1, 100000), LoadType.F)

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
    frame_work.visualize()

    return frame_work.stsol[:3*frame_work.num_nodes]

def portal_frame_free_vibration_test(x0):
    # Initialize two simple beam with 50 nodes and length 5.0
    length = 5.0
    num_elements = 100
    E, I, rho, A = 210 * 10 ** 9, 36.92 * 10 ** (-6), 42.3, 5383 * 10 ** (-6)

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

    # Add constraints
    frame_work.add_constraint(beam_1, 0, 0, ConstraintType.DISPLACEMENT)
    frame_work.add_constraint(beam_1, 0, 0, ConstraintType.AXIAL)
    frame_work.add_constraint(beam_1, 0, 0, ConstraintType.ROTATION)

    frame_work.add_constraint(beam_3, -1, 0, ConstraintType.DISPLACEMENT)
    frame_work.add_constraint(beam_3, -1, 0, ConstraintType.AXIAL)
    frame_work.add_constraint(beam_3, -1, 0, ConstraintType.ROTATION)

    # assemble the global matrices
    frame_work.assemble_frame_matrices()

    frame_work.solv(tau=0.001, num_steps=1000, x0= x0, dx0=0, sol_type=SolvType.EIGEN)
    eigen_sol = frame_work.dysol.T

    frame_work.solv(tau=0.001, num_steps=1000, x0=x0, dx0=0, sol_type=SolvType.DYNAMIC)
    fem_sol = frame_work.dysol.T

    # Visualization
    matplotlib.use("WebAgg", force=True)
    x = [coord[0] for coord in frame_work.coordinates]
    y = [coord[1] for coord in frame_work.coordinates]

    # Prepare data for plotting
    x_d_fem = np.zeros((len(x), fem_sol.shape[1]))
    x_d_eigen = np.zeros((len(x), fem_sol.shape[1]))
    y_d_fem = np.zeros((len(y), fem_sol.shape[1]))
    y_d_eigen = np.zeros((len(y), eigen_sol.shape[1]))

    for i in range(fem_sol.shape[1]):
        x_d_fem[:, i] = x
        y_d_fem[:, i] = y

        x_d_eigen[:, i] = x
        y_d_eigen[:, i] = y

    # compute deformed x & y in fem
    for beam in frame_work.beams:
        angle = beam.angle
        nodes = frame_work.nodes[frame_work.beams.index(beam)]
        beam_start = nodes[0]
        nodes_local = [node - beam_start for node in nodes]

        w_d1 = fem_sol[3 * beam_start + beam.num_nodes: 3 * beam_start + beam.num_nodes + 2 * nodes_local[-1] + 1: 2, :]
        v_d1 = fem_sol[3 * beam_start: 3 * beam_start + nodes_local[-1] + 1, :]

        w_d2 = eigen_sol[3 * beam_start + beam.num_nodes: 3 * beam_start + beam.num_nodes + 2 * nodes_local[-1] + 1: 2, :]
        v_d2 = eigen_sol[3 * beam_start: 3 * beam_start + nodes_local[-1] + 1, :]

        # x = x0 + w * sin(angle) - v * cos(angle)
        dx_d1 = np.sin(angle) * w_d1 - np.cos(angle) * v_d1
        dx_d2 = np.sin(angle) * w_d2 - np.cos(angle) * v_d2

        # y = y0 - w * cos(angle) - v * sin(angle)
        dy_d1 = -np.cos(angle) * w_d1 - np.sin(angle) * v_d1
        dy_d2 = -np.cos(angle) * w_d2 - np.sin(angle) * v_d2

        # transform the complex values to real values
        dx_d1 = np.real(dx_d1)
        dy_d1 = np.real(dy_d1)
        dx_d2 = np.real(dx_d2)
        dy_d2 = np.real(dy_d2)

        x_d_fem[nodes[0]:nodes[-1] + 1, ] += dx_d1[:, ]
        y_d_fem[nodes[0]:nodes[-1] + 1, ] += dy_d1[:, ]

        x_d_eigen[nodes[0]:nodes[-1] + 1, ] += dx_d2[:, ]
        y_d_eigen[nodes[0]:nodes[-1] + 1, ] += dy_d2[:, ]

    # Create a figure for the dynamic animation
    fig, ax = plt.subplots()
    line1, = ax.plot([], [], 'r-', label='Dynamic Solution (FEM)')
    line2, = ax.plot([], [], 'b-', label='Dynamic Solution (Eigen)')

    ax.plot(x, y, 'k-', label='Original Beam')
    ax.set_xlim(np.min(x_d_fem), np.max(x_d_fem))
    ax.set_ylim(np.min(y_d_fem), np.max(y_d_fem))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Comparison of FEM and Eigen Dynamic Solutions Over Time')
    ax.legend(loc="upper left")

    # Update function for animation
    def update(frame):
        line1.set_data(x_d_fem[:, frame], y_d_fem[:, frame])
        line2.set_data(x_d_eigen[:, frame], y_d_eigen[:, frame])
        return line1, line2

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=fem_sol.shape[1], blit=True)

    # Save the animation as a GIF
    ani.save(r'../output/dynamic_solution_final2.gif', writer='imagemagick', fps=30)

    # Display the animation
    plt.show()


if __name__ == "__main__":
    # Run the test
    x0 = portal_frame_staic_test()
    portal_frame_free_vibration_test(x0)