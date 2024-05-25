import numpy as np
import matplotlib.pyplot as plt
from utils.newmark import NewMark


def test_newmark():
    # System parameters
    M = np.array([[1]])  # Mass matrix
    S = np.array([[1]])  # Stiffness matrix
    num_steps = 100
    tau = 0.1
    beta = 0.25
    gamma = 0.5

    # Initial conditions
    x0 = np.array([1])  # Initial displacement
    dx0 = np.array([0])  # Initial velocity
    ddx0 = np.array([0])  # Initial acceleration

    # External force (zero in this case)
    f = np.zeros((1, num_steps))

    # Create NewMark solver
    newmark_solver = NewMark(tau, num_steps, beta, gamma)

    # Solve the system
    u, du, ddu = newmark_solver.solve(M, S, f, x0, dx0, ddx0)

    # Plot results
    time = np.linspace(0, (num_steps - 1) * tau, num_steps)
    plt.plot(time, u[0, :], label='Displacement')
    plt.plot(time, du[0, :], label='Velocity')
    plt.plot(time, ddu[0, :], label='Acceleration')
    plt.xlabel('Time')
    plt.ylabel('Response')
    plt.legend()
    plt.title('Newmark Method for SDOF System')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    test_newmark()
