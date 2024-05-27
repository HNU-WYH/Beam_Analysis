from src.beam import Beam
from src.fem import FEM
from utils.local_matrix import LocalElement
from utils.config import LoadType


def uniform_load_function(x):
    return 1.0  # Constant distributed load of 1000 N/m


def triangular_load_function(x):
    return x


def partial_uniform_load_function(x):
    if x <= 0.5:
        return 1
    else:
        return 0

def test_equal_force():
    # Define beam parameters
    length = 1.0
    num_elements = 10
    E, I, rho = 1, 1, 1

    # Initialize the beam
    beam = Beam(length, E, rho, I, num_elements)

    # Initialize FEM
    fem = FEM(beam)

    # Define a distributed load function
    fem.apply_force(uniform_load_function, LoadType.p)
    print("\nEquivalent nodal forces for uniform load\n",fem.q)
    print("Equivalent nodal forces for triangular distributed load\n",
          LocalElement.equal_force(triangular_load_function, LoadType.p, 0, beam.element_len))
    print("Equivalent nodal forces for partial uniform load\n",
          LocalElement.equal_force(partial_uniform_load_function, LoadType.p, 0, beam.element_len))

if __name__ == '__main__':
    test_equal_force()
