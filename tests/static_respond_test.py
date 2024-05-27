import matplotlib.pyplot as plt
from src.beam import Beam
from src.fem import FEM, LoadType, ConstraintType, SolvType


def test_static_fem():
    # Initialize a simple beam with 5 nodes and length 10
    length = 5.0
    num_elements = 50
    E, I, rho = 210 * 10 ** 9, 1 * 10 ** (-6), 7800

    # Initialize the beam
    beam = Beam(length, E, rho, I, num_elements)

    fem = FEM(beam)

    # Apply a force of 100 at position 5
    fem.apply_force((5, 100), LoadType.F)

    # Add constraints: displacement at node 0 is 0, rotation at node 4 is 0
    fem.add_constraint(0, 0, ConstraintType.DISPLACEMENT)
    fem.add_constraint(0, 0, ConstraintType.ROTATION)

    # Solve the system
    fem.solv()
    stsol = fem.stsol[0:2 * (num_elements + 1):2]

    # Plot the static solution
    plt.plot(stsol, label='Static Solution')
    plt.xlabel('Node')
    plt.ylabel('Displacement')
    plt.title('Static Solution')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Run the test
    test_static_fem()