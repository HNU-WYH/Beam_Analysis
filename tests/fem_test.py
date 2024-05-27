from src.beam import Beam
from src.fem import FEM
import matplotlib.pyplot as plt
from utils.config import LoadType, ConstraintType, SolvType


def test_fem():
    # Initialize a simple beam with 5 nodes and length 10
    length = 5.0
    num_elements = 5
    E, I, rho = 1, 1, 1


    # Initialize the beam
    beam = Beam(length, E, rho, I, num_elements)

    fem = FEM(beam)

    # Apply a force of 100 at position 5
    fem.apply_force((5, 100), LoadType.F)

    # Add constraints: displacement at node 0 is 0, rotation at node 4 is 0
    fem.add_constraint(0, 0, ConstraintType.displacement)
    fem.add_constraint(0, 0, ConstraintType.rotation)

    # Solve the system
    fem.solv()

    # Output the solution
    plt.plot(fem.stsol[0:2*(num_elements+1):2])
    plt.show()

    fem.solv(tau=0.1, num_steps=10000, soltype=SolvType.dynamic)
    plt.plot(fem.dysol[0:2*(num_elements+1):2,-1])
    plt.show()

# Run the test
test_fem()