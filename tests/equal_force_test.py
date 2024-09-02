from src.beam import Beam
from src.fem_beam import FEM
from src.utils.local_matrix import LocalElement
from src.utils.local_matrix import LoadType
from config import uniform_load_function, partial_uniform_load_function, triangular_load_function


def test_equal_force():
    # Define beam parameters
    length = 1.0
    num_elements = 10
    E, I, rho = 1, 1, 1

    # Initialize the beam
    beam = Beam(length, E, rho, I, num_elements)

    # Initialize FEM
    fem = FEM(beam)

    print("\nEquivalent nodal forces for triangular distributed load\n",
          LocalElement.equal_force(uniform_load_function, LoadType.q, 0, beam.element_len))
    print("Equivalent nodal forces for triangular distributed load\n",
          LocalElement.equal_force(triangular_load_function, LoadType.q, 0, beam.element_len))
    print("Equivalent nodal forces for partial uniform load\n",
          LocalElement.equal_force(partial_uniform_load_function, LoadType.q, 0, beam.element_len))

    fem.apply_force(uniform_load_function, LoadType.q)
    print("\nGlobal nodal forces for uniform load\n", fem.q)


if __name__ == '__main__':
    test_equal_force()
