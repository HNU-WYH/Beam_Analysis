import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import matplotlib

from .beam import Beam
from .utils.eigs import EigenMethod

from .utils.local_matrix import LocalElement
from .utils.newmark import NewMark
from .config import LoadType, ConstraintType, SolvType

class FEM:
    """
    This class represents the postprocessing of a beam structure, including applying forces, applying constraints,
    and solving the resulting displacement in static & dynamic way.

    Compared with the global mass & stiffness matrices in preprocessing steps, the S & M matrices here are expanded
    to include the constraints, details of which can be found in Chapter 6 of script1_bending_and_fem.pdf.

    Attributes:
        S (np.ndarray): The expanded global stiffness matrix.
        M (np.ndarray): The expanded global mass matrix.
        stsol (np.ndarray): The displacement vector for the static case.
        dysol (np.ndarray): The displacement vector for the dynamic case.
        beam (Beam): The beam object in preprocessing steps, imported from beam.py
        constraints (list): A list of constraints applied to the beam.
        q (np.ndarray): The equivalent global nodal force vector, representing applied loads.
        new_q (np.ndarray): The global equivalent nodal force vector with constraints, namely [q,a].

    """
    def __init__(self, beam: Beam):
        """
        Initializes the FEM object with the beam object, setting up the initial state for the analysis.

        Parameters:
            beam (Beam): The beam object from beam.py, containing the structure and properties of the beam.
        """
        self.S = None                           # Global stiffness matrix (to be set after applying constraints)
        self.M = None                           # Global mass matrix (to be set after applying constraints)
        self.stsol = None                       # Static solution vector (to be computed)
        self.dysol = None                       # Dynamic solution vector (to be computed)
        self.beam = beam                        # Beam object containing the beam structure
        self.constraints = []                   # List to store applied constraints
        self.q = np.zeros(2 * beam.num_nodes)   # Initialize the global equivalent nodal force vector with zeros
        self.new_q = None                       # The global equivalent nodal force vector with constraints
        self._activate = False                  # whether the constraints has been added to the stiffness matrix

    def apply_force(self, load, load_type):
        """
        Applies a force or distributed load to the beam and updates the global force vector.

        Parameters:
            load (tuple or function): The load to apply. For point loads and moments, this is a tuple
                                      (position, magnitude). For distributed loads, this is a function of position.
            load_type (LoadType): The type of load, which can be a distributed load (q), point force (F), or moment (M).
        """
        if load_type == LoadType.q:
            # Apply distributed load over each element
            for idx, xstart in enumerate(self.beam.nodes[0:-1]):
                self.q[2 * idx:2 * idx + 4] += LocalElement.equal_force(
                    load,
                    load_type,
                    xstart,
                    self.beam.element_len
                )

        elif load_type == LoadType.F:
            # Apply point force
            f_pos, f_val = load
            idx = int(f_pos / self.beam.element_len)

            # check whether the position of force is beyond the range of beam
            if f_pos > self.beam.L or f_pos < 0:
                Warning("force applied beyond the beam", f_pos)

            # the force is directly applied on nodes
            elif f_pos in self.beam.nodes:
                self.q[2 * idx] = f_val

            # if the force is not applied on nodes, computing its equivalent nodal force
            else:
                self.q[2 * idx:2 * idx + 4] += LocalElement.equal_force(
                    load,
                    load_type,
                    idx * self.beam.element_len,
                    self.beam.element_len
                )

        elif load_type == LoadType.M:
            # Apply moment
            m_pos, m_val = load
            idx = int(m_pos / self.beam.element_len)

            # check whether the position of moment is beyond the range of beam
            if m_pos > self.beam.L or m_pos < 0:
                raise Warning("moment applied beyond the beam", m_pos)

            # the moment is directly applied on nodes
            elif m_pos in self.beam.nodes:
                self.q[2 * idx + 1] = m_val

            # if the moment is not applied on nodes, computing its equivalent nodal force
            else:
                self.q[2 * idx:2 * idx + 4] += LocalElement.equal_force(
                    load,
                    load_type,
                    idx * self.beam.element_len,
                    self.beam.element_len
                )

    def add_constraint(self, node, value, constraint_Type: ConstraintType):
        """
        Storing given constraints of the beam and corresponding constraint types.
        For simplifying the program, the constraints must be applied on nodes instead of middle of element.

        Parameters:
            node (int): The node index where the constraint is applied.
            value (float): The specific forced displacement of the constraint (e.g., displacement or rotation value, normally set to be 0).
            constraint_Type (ConstraintType): The type of constraint, either rotation or displacement.
        """
        self.constraints.append((node, value, constraint_Type))

    def _apply_constraint(self):
        """
        The 'add_constraint' method only append the given constraints to our list of constraints (self.constraints).
        No expansion of global mass & stiffness matrices is made.

        This method '__apply_constraint' is the internal method to actually apply the constraints to the global matrices and expand them accordingly,
        which will be called in the method 'solve'.

        Details of how to expand the stiffness & mass matrices based on given constraints can be found in Chapter 6 of script1_bending_and_fem.pdf.
        """

        num_constraints = len(self.constraints)                     # Number of constraints
        original_size = self.beam.S.shape[0]                        # Original size of the global matrices
        expand_size = num_constraints + original_size               # New size after expansion

        # create the expanded matrix
        self.S = np.zeros((expand_size, expand_size))               # initialize the expanded stiffness matrix
        self.M = np.zeros((expand_size, expand_size))               # initialize the expanded mass matrix
        self.new_q = np.zeros(expand_size)                          # initialize the expanded global force vector

        # copy original S,M,q into expanded matrix
        self.S[0:original_size,0:original_size] = self.beam.S
        self.M[0:original_size, 0:original_size] = self.beam.M
        self.new_q[0:original_size] = self.q

        # Apply constraints
        for i, (node, value, constraint_type) in enumerate(self.constraints):
            constraint_idx  = original_size+i
            self.new_q[constraint_idx] = value

            if constraint_type == ConstraintType.ROTATION:
                self.S[constraint_idx, 2 * node + 1] = 1
                self.S[2 * node + 1, constraint_idx] = 1
            elif constraint_type == ConstraintType.DISPLACEMENT:
                self.S[constraint_idx, 2 * node] = 1
                self.S[2 * node, constraint_idx] = 1
            else:
                Warning("wrong type of constraint", constraint_type)

    def initial_acceleration(self, x0 = 0):
        if np.isscalar(x0):
            x0 = np.full(shape=self.q.shape, fill_value=x0)
        ddx0 = np.linalg.solve(self.beam.M, self.q - self.beam.S @ x0 )
        return ddx0

    def solv(self, tau=None, num_steps=None, x0 = 0, dx0 = 0, sol_type=SolvType.STATIC,  beta=0.25, gamma=0.5):
        """
        Solves the system of equations for the beam under the applied forces and constraints.

        Parameters:
            num_steps (int, optional): Number of time steps for dynamic analysis.
            tau (float, optional): Time step size for dynamic analysis.
            sol_type (SolvType): Type of solution, either STATIC or DYNAMIC.
            beta (float, optional): Newmark beta parameter for dynamic analysis.
            gamma (float, optional): Newmark gamma parameter for dynamic analysis.

        Raises:
            Exception: If the solution type is incorrectly defined.
        """
        if not self._activate:
            self._activate = True
            self._apply_constraint()  # Apply constraints to the global matrices

        if sol_type == SolvType.STATIC:
            # Static solution
            self.stsol = np.linalg.solve(self.S, self.new_q)

        elif sol_type == SolvType.DYNAMIC or sol_type == SolvType.EIGEN:
            # calculating the initial condition
            if np.isscalar(x0) :
                x0 = np.full(shape=self.q.shape, fill_value=x0)
            if np.isscalar(dx0):
                dx0 = np.full(shape=self.q.shape, fill_value=dx0)

            if sol_type == SolvType.DYNAMIC:
                # calculating the initial acceleration
                ddx0 = self.initial_acceleration(x0)

                # expand initial conditions
                x0 = np.concatenate([x0,np.zeros(len(self.constraints))])
                dx0 = np.concatenate([dx0,np.zeros(len(self.constraints))])
                ddx0 = np.concatenate([ddx0,np.zeros(len(self.constraints))])

                # Dynamic solution using Newmark's method
                newmark_solver = NewMark(tau, num_steps, beta, gamma)
                self.dysol, _, _ = newmark_solver.solve(self.M, self.S, self.new_q, x0, dx0, ddx0)

            else:
                N, K = self.beam.S.shape[0], len(self.constraints)
                eigen = EigenMethod(N, K)
                res = eigen.solve(self.M, self.S, x0, dx0) # a function that require t and return the result in t
                self.dysol = np.array([res(t) for t in np.arange(0, tau * num_steps, tau)])

        else:
            raise Exception("Wrong defined type of solution")

    def assess_dis(self, coefficient, x_values):
        y_values = [LocalElement.app_func(x, coefficient, [0, self.beam.L], self.beam.num_elements) for x in x_values]
        return y_values


    def visualize(self, sol_type=SolvType.STATIC, granularity = 100, title='Displacement', written = False):
        """
        Visualizes the displacement of the beam.

        Parameters:
            sol_type (SolvType): Type of solution, either STATIC, DYNAMIC, or EIGEN.
            title (str): Title for the plot or animation.
            granularity: The number of interpolation points on the solution
            written: The flag determining whether we plot the figure
        """
        matplotlib.use('webagg')
        x_values = np.linspace(0, self.beam.L, granularity)

        if sol_type == SolvType.STATIC:
            # Static solution - plot a static graph
            y_values = self.assess_dis(self.stsol, x_values)

            plt.figure()
            plt.plot(x_values, y_values, label='Static Displacement')
            plt.xlabel('Position along the beam (x)')
            plt.ylabel('Displacement')
            plt.title(title)
            plt.legend()
            plt.grid(True)
            plt.show()

        elif sol_type in [SolvType.DYNAMIC, SolvType.EIGEN]:
            # Dynamic solution - create an animation
            fig, ax = plt.subplots()
            line, = ax.plot([], [], 'r-', label='Dynamic Solution')

            # Create a figure and axis for the animation
            y_static = self.assess_dis(self.stsol, x_values)

            ax.plot(x_values, y_static, 'b-', label='Static Solution')
            ax.set_xlim(0, self.beam.L)
            ax.set_ylim(min(np.min(self.dysol[:,:2 * (self.beam.num_elements + 1)]),
                            np.min(self.stsol[:2 * (self.beam.num_elements + 1)])),
                        max(np.max(self.dysol[:,:2 * (self.beam.num_elements + 1)]),
                            np.max(self.stsol[:2 * (self.beam.num_elements + 1)])))
            ax.set_xlabel('Position along the beam (x)')
            ax.set_ylabel('Displacement')
            ax.set_title('Dynamic Solution Over Time')
            ax.legend(loc="upper left")

            def update(frame):
                y_dynamic = self.assess_dis(self.dysol[frame], x_values)
                line.set_data(x_values, y_dynamic)
                return line,

            ani = animation.FuncAnimation(fig, update, frames=self.dysol.shape[0], blit=True)

            if written:
                ani.save(r'../output/fem_static_dynamic_solution.gif', writer='imagemagick', fps=30)

            plt.show()