import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src.beam import Beam
from src.fem import FEM, LoadType, ConstraintType, SolvType
from src.utils.eigs import EigenMethod


def test_dynamic_eigen():
    matplotlib.use("WebAgg", force=True)

    # Initialize a simple beam with 5 nodes and length 10
    length = 5.0
    num_elements = 50
    E, I, rho = 210 * 10 ** 9, 1 * 10 ** (-6), 7800

    # Initialize the beam
    beam = Beam(length, E, rho, I, num_elements)

    # Initialize FEM model
    fem = FEM(beam)

    # Add constraints: displacement at node 0 is 0, rotation at node 4 is 0
    fem.add_constraint(0, 0, ConstraintType.DISPLACEMENT)
    fem.add_constraint(0, 0, ConstraintType.ROTATION)

    # Solve the static system
    fem._apply_constraint()

    N = beam.S.shape[0]
    K = fem.S.shape[0] - beam.S.shape[0]

    eigen = EigenMethod(N,K)
    res =  eigen.solve(fem.M, fem.S, np.zeros(N)[:,None], np.ones(N)[:,None])

    # Get the dynamic solution
    dysol = np.array([res(t) for t in np.arange(0,10,0.1)])[:,0,:N:2]

    # Create a figure and axis for the animation
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'r-', label='Dynamic Solution')
    ax.set_xlim(0, dysol.shape[1] - 1)
    ax.set_ylim(np.min(dysol), np.max(dysol))
    ax.set_xlabel('Node')
    ax.set_ylabel('Displacement')
    ax.set_title('Dynamic Solution Over Time')
    ax.legend(loc="upper left")

    # Update function for animation
    def update(frame):
        line.set_data(range(dysol.shape[1]), dysol[frame, :])
        return line,

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=dysol.shape[0], blit=True, repeat=False)

    # Save the animation as a GIF (optional)
    ani.save(r'..\output\dynamic_solution2.gif', writer='imagemagick', fps=30)

    # Display the animation
    plt.show()


if __name__ == "__main__":
    # Run the test
    test_dynamic_eigen()