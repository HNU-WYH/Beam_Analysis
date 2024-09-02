import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src.beam import Beam
from src.fem_beam import FEM, LoadType, ConstraintType, SolvType


def test_dynamic_fem():
    matplotlib.use("WebAgg", force=True)

    # Initialize a simple beam with 5 nodes and length 10
    length = 5.0
    num_elements = 50
    E, I, rho = 210 * 10 ** 9, 1 * 10 ** (-6), 7800

    # Initialize the beam
    beam = Beam(length, E, rho, I, num_elements)

    # Initialize FEM model
    fem = FEM(beam)

    # Apply a force of 100 at position 5
    fem.apply_force((5, 50), LoadType.F)

    # Add constraints: displacement at node 0 is 0, rotation at node 4 is 0
    fem.add_constraint(0, 0, ConstraintType.DISPLACEMENT)
    fem.add_constraint(0, 0, ConstraintType.ROTATION)

    # Solve the static system
    fem.solv()

    # Get the static solution
    stsol = fem.stsol[0:2 * (num_elements + 1):2]

    # Solve the dynamic system
    fem.solv(tau=0.1, num_steps=200, sol_type=SolvType.DYNAMIC)

    # Get the dynamic solution
    dysol = fem.dysol[0:2 * (num_elements + 1):2, :]

    # Create a figure and axis for the animation
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'r-', label='Dynamic Solution')
    ax.plot(stsol, 'b-', label='Static Solution')
    ax.set_xlim(0, len(stsol) - 1)
    ax.set_ylim(np.min(dysol), np.max(dysol))
    ax.set_xlabel('Node')
    ax.set_ylabel('Displacement')
    ax.set_title('Dynamic Solution Over Time')
    ax.legend(loc = "upper left")

    # Update function for animation
    def update(frame):
        line.set_data(range(len(stsol)), dysol[:, frame])
        return line,

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=dysol.shape[1], blit=True)

    # Save the animation as a GIF
    ani.save(r'..\output\fem_static_dynamic_solution.gif', writer='imagemagick', fps=30)

    # Display the animation
    plt.show()


if __name__ == "__main__":
    # Run the test
    test_dynamic_fem()
