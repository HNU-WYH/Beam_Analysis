import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src.beam import Beam
from src.fem import FEM, LoadType, ConstraintType, SolvType
from src.utils.eigs import EigenMethod
from src.utils.newmark import NewMark


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
    fem._apply_constraint()

    # dimension for original & expanded S, M matrices
    N = beam.S.shape[0]
    K = fem.S.shape[0] - beam.S.shape[0]

    # create initial conditions
    y0 = np.zeros(fem.new_q.shape)
    dy0 = np.concatenate((np.ones(N),np.zeros(K)))


    # Solve the dynamic system in fem method
    newmark_solver = NewMark(tau = 0.01, num_steps = 1000, beta = 0.25, gamma = 0.5)
    dysol1, _, _ = newmark_solver.solve(fem.M, fem.S, y0, y0, dy0, y0)
    dysol1 = dysol1.T[:,:N:2]

    # Solve the static system in eigen method
    eigen = EigenMethod(N,K)
    y0 = np.zeros(N)[:, None]
    dy0 = np.ones(N)[:, None]
    res = eigen.solve(fem.M, fem.S, y0, dy0)
    dysol2 = np.array([res(t) for t in np.arange(0,100,0.01)])[:,0,:N:2]

    # Create a figure and axis for the animation
    fig, ax = plt.subplots()
    line1, = ax.plot([], [], 'r-', label='Dynamic Solution (FEM)')
    line2, = ax.plot([], [], 'b-', label='Dynamic Solution (Eigen)')
    ax.set_xlim(0, dysol2.shape[1] - 1)
    ax.set_ylim(min(np.min(dysol1), np.min(dysol2)), max(np.max(dysol1), np.max(dysol2)))
    ax.set_xlabel('Node')
    ax.set_ylabel('Displacement')
    ax.set_title('Comparison of FEM and Eigen Solutions Over Time')
    ax.legend(loc="upper left")

    # Update function for animation
    def update(frame):
        line1.set_data(range(dysol1.shape[1]), dysol1[frame, :])
        line2.set_data(range(dysol2.shape[1]), dysol2[frame, :])
        return line1, line2

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=min(dysol1.shape[0], dysol2.shape[0]), blit=True, repeat=False)

    # Save the animation as a GIF (optional)
    ani.save(r'..\output\eigen_vs_fem_solution.gif', writer='imagemagick', fps=30)

    # Display the animation
    plt.show()


if __name__ == "__main__":
    # Run the test
    test_dynamic_eigen()