import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src.beam import Beam
from src.fem_beam import FEM, LoadType, ConstraintType, SolvType


def test_dynamic_eigen():
    matplotlib.use("WebAgg", force=True)

    # Initialize a simple beam with 5 nodes and length 10
    length = 5.0
    num_elements = 5
    N = 2 * (num_elements + 1)
    E, I, rho = 210 * 10 ** 9, 1 * 10 ** (-6), 7800

    # Initialize the beam
    beam = Beam(length, E, rho, I, num_elements)

    # Initialize FEM model
    fem = FEM(beam)

    # Add constraints: displacement at node 0 is 0, rotation at node 4 is 0
    fem.add_constraint(0, 0, ConstraintType.DISPLACEMENT)
    fem.add_constraint(0, 0, ConstraintType.ROTATION)

    # create initial conditions
    # the initial position & speed of every node
    y0 = 0
    dy0 = 1

    # no external force, since eigenvalue method can only be used in homogenous case

    # Solve the dynamic system in fem method
    fem.solv(tau = 0.01, num_steps = 100, x0 = y0, dx0 = dy0, sol_type= SolvType.DYNAMIC)
    dysol1 = fem.dysol
    dysol1_vis = dysol1.T[:,:N:2]


    # Solve the dynamic system in eigen method
    fem.solv(tau=0.01, num_steps=100, x0=y0, dx0=dy0, sol_type=SolvType.EIGEN)
    dysol2 = fem.dysol
    dysol2_vis = dysol2[:,:N:2]


    # Create a figure and axis for the animation
    fig, ax = plt.subplots()
    line1, = ax.plot([], [], 'r-', label='Dynamic Solution (FEM)')
    line2, = ax.plot([], [], 'b-', label='Dynamic Solution (Eigen)')
    ax.set_xlim(0, dysol2_vis.shape[1] - 1)
    ax.set_ylim(min(np.min(dysol1_vis), np.min(dysol2_vis)), max(np.max(dysol1_vis), np.max(dysol2_vis)))
    ax.set_xlabel('Node')
    ax.set_ylabel('Displacement')
    ax.set_title('Comparison of FEM and Eigen Solutions Over Time')
    ax.legend(loc="upper left")

    # Update function for animation
    def update(frame):
        line1.set_data(range(dysol1_vis.shape[1]), dysol1_vis[frame, :])
        line2.set_data(range(dysol2_vis.shape[1]), dysol2_vis[frame, :])
        return line1, line2

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=min(dysol1_vis.shape[0], dysol2_vis.shape[0]), blit=True, repeat=False)

    # Save the animation as a GIF (optional)
    ani.save(r'..\output\eigen_vs_fem_solution3.gif', writer='imagemagick', fps=30)

    # Display the animation
    plt.show()


if __name__ == "__main__":
    # Run the test
    test_dynamic_eigen()