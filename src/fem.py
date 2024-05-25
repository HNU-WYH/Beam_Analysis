import numpy as np

class FEM:
    def __inti__(self, h, tau, E, I, rho):
        self.h = h # Grid length
        self.tau = tau # Time step
        self.E = E # Young's modulus
        self.I = I  # Moment of inertia
        self.rho = rho  # Mass density