import numpy as np


class NewMark:
    '''
    solving $M\ddot x + S x = f$
    '''

    def __init__(self, tau: float, num_steps: int, beta=0.25, gamma=0.5):
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.num_steps = num_steps

    def solve(self, M: np.ndarray, S: np.ndarray, f: np.ndarray, x: np.ndarray, dx: np.ndarray, ddx: np.ndarray):
        '''
        M: the Mass Matrix
        S: the Stiffness Matrix
        f: the external force varies with time
        x: initial condition of x\in\mathbb R^n
        dx: intial condition of \dot x\in\mathbb R^n
        ddx: intial condition of \ddot x\in\mathbb R^n

        Newmark Method:
        solve the equation of $(S\beta h_j^2)\ddot{x}_{j+1}=f-M\ddot{x}_j-Sx_j^*$
        A = S\beta h_j^2
        sol = ddx_{j+1}
        b =f-M\ddot{x}_j-Sx_j^*
        '''
        # Initialize
        num_dofs = x.shape[0]
        u = np.zeros([num_dofs, self.num_steps])
        du = np.zeros([num_dofs, self.num_steps])
        ddu = np.zeros([num_dofs, self.num_steps])
        is_f_dynamic = len(f.shape) == 2

        u[:, 0] = x
        du[:, 0] = dx
        ddu[:, 0] = ddx

        for j in range(self.num_steps - 1):
            u_star = u[:, j] + self.tau * du[:, j] + (0.5 - self.beta) * self.tau ** 2 * ddu[:, j]
            du_star = du[:, j] + (1 - self.gamma) * ddu[:, j] * self.tau
            if is_f_dynamic:
                rhs = f[:, j + 1] - S @ u_star
            else:
                rhs = f - S @ u_star
            ddu[:, j + 1] = np.linalg.solve(S * self.beta * self.tau ** 2 + M, rhs)
            u[:, j + 1] = u_star + self.beta * ddu[:, j + 1] * self.tau ** 2
            du[:, j + 1] = du_star + self.gamma * ddu[:, j + 1] * self.tau
        return u, du, ddu
