from typing import Callable

import numpy as np


class EigenMethod:
    """
    The (numerical) eigenvalue method is used to solve the homogeneous DAE of the form:

        [ M  0 ] * [ ÿ ]  + [ S    C ]  * [ ÿ ] = 0
        [ 0  0 ]   [ ü ]    [ C^T  0 ]    [ ü ]
          M_e                  S_e
    where:
    - M is the mass matrix, which is N*N symmetric positive definite.
    - S is the stiffness matrix, which is N*N symmetric positive semi-definite.
    - C is the constraints matrix, which is N*K and has linearly independent columns.

    Let A = S_e^{-1}M_e \in R^{(N+K)×(N+K)}, there exist N-K linearly independent eigenvectors [y_k, u_k]^T, s.t.:

    - A[y_k, u_k]^T = λ_k[y_k, u_k]^T with λ_k>0

    - y_j^T M y_k = 0 for j ≠ k

    Then the numerical solution for the DAE are in the form:
             N-K
    [y(t)] =  Σ  (α_k cos(w_kt) + β_k/w_k sin(w_kt)) [y_k]  with w_k = 1/sqrt(λ_k)
    [u(t)]    1                                      [u_k]

    where
    - α_k = (y_k^T M y(0)) / (y_k^T M y_k)

    - β_k = (y_k^T M \dot y(0)) / (y_k^T M y_k)

    - y(0) & \dot y(0) are the initial condition for displacement & slope

    References:
    1. Details & Proof can be found in script5_ev_method_numerical_back.pdf

    2. The analytic version for eigenvalue method instead of a numerical one can be found in script4_ev_method_analytic.pdf,
       which is not shown here.

    """

    def __init__(self,N: int, K: int):
        self.N = N
        self.K = K
        self.M = None
        self.A = None
        self.Me = None
        self.Se = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.wave_modes = None
        self.alphas = None
        self.betas = None
        self.omegas = None
        self.sol = None

    def solve(self, Me: np.ndarray, Se: np.ndarray, y0: np.ndarray, dy0: np.ndarray)-> Callable:
        if Me.shape != Se.shape or Me.shape[0]!=Me.shape[1] or Me.shape[0]!=self.K + self.N:
            raise ValueError("The shape of stiffness & mass matrices is incompatible")

        self.Me, self.Se = Me, Se
        self.M = self.Me[:self.N,:self.N]
        self.A = np.linalg.inv(self.Se) @ self.Me

        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.A)

        # Filter positive eigenvalues and corresponding eigenvectors
        positive_idx = self.eigenvalues > 1e-8  # filtering near-zero or negative eigenvalues
        self.eigenvalues = self.eigenvalues[positive_idx]
        self.eigenvectors = self.eigenvectors[:, positive_idx]

        self.wave_modes = [self.eigenvectors[:self.N, i:i + 1] for i in range(len(self.eigenvalues))]

        self.alphas = [(mode.T @ self.M @ y0)/(mode.T @ self.M @ mode)[0][0] for mode in self.wave_modes]
        self.betas =  [(mode.T @ self.M @ dy0)/(mode.T @ self.M @ mode)[0][0] for mode in self.wave_modes]
        self.omegas = [ 1./np.sqrt(x) for x in self.eigenvalues]

        def sol(t:float):
            res = 0
            for i, y_k in enumerate(self.eigenvectors.T):
                coefficient = self.alphas[i] * np.cos(t * self.omegas[i]) + self.betas[i]/self.omegas[i] * np.sin(t * self.omegas[i])
                res += coefficient * y_k
            return res

        self.sol = sol
        return sol




