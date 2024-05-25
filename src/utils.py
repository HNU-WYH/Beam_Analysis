import numpy as np


class NewMarkBeta:
    '''
    solving $M\ddot x + S x = f$
    '''

    def __inti__(self, tau: float, num_steps: int, beta=0.25, gamma=0.5):
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.num_steps = num_steps

    def solve(self, M: np.ndarray, S: np.ndarray, f: np.ndarray, x: np.ndarray, ddx: np.ndarray):
        '''
        M: the Mass Matrix
        S: the Stiffness Matrix
        x: initial condition of x\in\mathbb R^n
        ddx: intial condition of \ddot x\in\mathbb R^n
        '''
        u = np.zeros([x.shape, self.num_steps])
        ddu = np.zeros([ddx.shape, self.num_steps])
        ddu[:, 0] = ddu
        u[:, 0] = x

        for j in range(self.num_steps - 1):
            u_star = u[:, j] + (0.5 - self.beta) * self.tau ** 2 * ddu[:, j]
            # solve the equation of $(S\beta h_j^2)\ddot{x}_{j+1}=f-M\ddot{x}_j-Sx_j^*$
            # A = S\beta h_j^2
            # sol = ddx_{j+1}
            # b =f-M\ddot{x}_j-Sx_j^*
            rhs = f - M @ ddu[:, -1] - S @ u_star
            ddu[:, j + 1] = np.linalg.solve(S * self.beta * self.tau ** 2, rhs)
            u[:, j + 1] = u_star + self.beta * ddu[:, j + 1] * self.tau ** 2

        return u, ddu
