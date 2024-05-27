import numpy as np
from src.utils import LocalElement


class Beam:
    def __init__(self, length, young_module, density, moment_inertia, num_elements):
        self.num_elements = num_elements
        self.num_nodes = num_elements + 1

        self.E = young_module
        self.I = moment_inertia
        self.rho = density
        self.L = length

        self.nodes = np.linspace(0, self.L, self.num_nodes)
        self.element_len = self.L / self.num_elements
        self.elements = []
        for i in range(self.num_elements):
            self.elements.append([i, i + 1])
        self.assemble_matrices()

    def local_matrices(self):
        self.local_S, self.local_M = LocalElement.evaluate(self.E, self.I, self.rho, self.element_len)

    def assemble_matrices(self):
        self.local_matrices()
        self.S = np.zeros((2 * self.num_nodes, 2 * self.num_nodes))
        self.M = np.zeros((2 * self.num_nodes, 2 * self.num_nodes))

        for element in self.elements:
            node1 = element[0]
            self.S[2 * node1:2 * node1 + 4, 2 * node1:2 * node1 + 4] += self.local_S
            self.M[2 * node1:2 * node1 + 4, 2 * node1:2 * node1 + 4] += self.local_M


if __name__ == "__main__":
    beam = Beam(length=3, young_module=1, density=1, moment_inertia=1, num_elements=3)
    print("the global stiffness matrix")
    print(beam.S)
    print("the global mass matrix")
    print(np.round(beam.M, 2))
