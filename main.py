"""
Example of a portal frame with fixed connections to demonstrate how to run the project.
This script performs a static analysis of the structure by applying multiple types of loads,
setting up supports and connections, and solving for the nodal displacements.

Other examples like cantilever beam, simply-supported beam and portal frame with hinged connections
are available in the directory "./analysis"
"""

import math
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Import enum types and FEM modules
from config import LoadType, ConstraintType, SolvType, ConnectionType
from src.beam import Beam2D
from src.fem_frame import FrameworkFEM

# Define the input parameters
length = 5.0  # Length of each beam (m)
num_elements = 100  # Number of elements per beam
E = 210e9  # Young's modulus in Pascals
I = 36.92e-6  # Moment of inertia in m^4
rho = 42.3  # Density in appropriate units (e.g., kg/m^3)
A = 5383e-6  # Cross-sectional area in m^2
F_value = 1000000  # Magnitude of the point force load (N)
M_value = 100000  # Magnitude of the bending moment (N·m)
q_value = 100000  # Magnitude of the distributed force (N/m)

# Static analysis of fixed portal framework
def run_static_analysis(length, num_elements, E, I, rho, A,
                        F_value, M_value, q_value,
                        connection=(-1, 0),
                        constraint_type=ConstraintType,
                        connection_type=ConnectionType,
                        load_type =LoadType):
    """
    Run static analysis for a portal frame.

    This function builds a portal frame consisting of three beams:
    - A vertical beam (beam_1)
    - Two inclined beams (beam_2 and beam_3)

    The beams are connected together with fixed connections, and multiple loads are applied
    on beam_1 at its right end (last node). The loads include a point force, a bending moment,
    and a distributed force. The structure is supported (i.e., constrained) at the left end of
    beam_1 and the right end of beam_3.

    Parameters:
        length (float): Length of each beam.
        num_elements (int): Number of elements per beam.
        E (float): Young's modulus.
        I (float): Moment of inertia.
        rho (float): Density.
        A (float): Cross-sectional area.

        F_value (float): Magnitude of the point force load (applied on beam_1).
        M_value (float): Magnitude of the bending moment load (applied on beam_1).
        q_value (float): Magnitude of the distributed force load (applied on beam_1).

        connection (tuple): A tuple indicating the node indices used for connecting beams.
                            Example: (-1, 0) means connecting the last node of one beam to the
                            first node of the adjacent beam.

        constraint_type (enum): Constraint types to be applied, including:
            - ConstraintType.DISPLACEMENT (fix the transverse displacement)
            - ConstraintType.AXIAL (fix the axial displacement)
            - ConstraintType.ROTATION (fix the rotation)

        connection_type (enum): Connection type between beams, including:
            - ConnectionType.Fix (fixed connection)
            - ConnectionType.Hinge (hinged connection)

        load_type (enum): TypeS of load for the force applied, including:
            - LoadType.F for point force,
            - LoadType.M for moment,
            - LoadType.q for distributed force.

    Returns:
        numpy.ndarray: The initial nodal displacement vector,
        which can be used as the starting point for subsequent dynamic analysis.
    """
    # Create beams:
    # - beam_1 is vertical (angle = 90° or π/2 radians)
    # - beam_2 is horizontal (default angle = 0)
    # - beam_3 is inclined in the opposite direction (angle = -π/2 radians)
    beam_1 = Beam2D(length, E, A, rho, I, num_elements, angle=math.pi / 2)
    beam_2 = Beam2D(length, E, A, rho, I, num_elements)
    beam_3 = Beam2D(length, E, A, rho, I, num_elements, angle=-math.pi / 2)

    # Initialize the FEM framework model.
    frame_work = FrameworkFEM()

    # Add beams to the framework.
    frame_work.add_beam(beam_1)
    frame_work.add_beam(beam_2)
    frame_work.add_beam(beam_3)

    # Add connections between beams.
    # The 'connection' tuple specifies which nodes to connect.
    frame_work.add_connection(beam_1, beam_2, connection, connection_type.Fix)
    frame_work.add_connection(beam_2, beam_3, connection, connection_type.Fix)

    # Apply loads on beam_1 at its right end (last node):
    # - A point force load using LoadType.F.
    # - A bending moment load using LoadType.M.
    # - A distributed force load using LoadType.q.
    frame_work.add_force(beam_1, (-1, F_value), load_type.F)
    frame_work.add_force(beam_1, (-1, M_value), load_type.M)
    frame_work.add_force(beam_1, lambda x: q_value, load_type.q)

    # Add constraints (supports) at the beam ends:
    # - Fix the left end of beam_1 (node index 0) in displacement, axial, and rotation.
    # - Fix the right end of beam_3 (node index -1) in displacement, axial, and rotation.
    frame_work.add_constraint(beam_1, 0, 0, constraint_type.DISPLACEMENT)
    frame_work.add_constraint(beam_1, 0, 0, constraint_type.AXIAL)
    frame_work.add_constraint(beam_1, 0, 0, constraint_type.ROTATION)

    frame_work.add_constraint(beam_3, -1, 0, constraint_type.DISPLACEMENT)
    frame_work.add_constraint(beam_3, -1, 0, constraint_type.AXIAL)
    frame_work.add_constraint(beam_3, -1, 0, constraint_type.ROTATION)

    # Assemble the global stiffness and mass matrices for the entire structure.
    frame_work.assemble_frame_matrices()

    # Solve the static equilibrium problem and visualize the deformed structure.
    frame_work.solv()
    frame_work.visualize()

    # Return the nodal displacement vector
    # This vector serves as the initial condition for dynamic analysis.
    return frame_work.stsol[:3 * frame_work.num_nodes]


print("Start Static analysis completed.")
initial_displacement = run_static_analysis(length, num_elements, E, I, rho, A,
                                           F_value, M_value, q_value)
print("Static analysis completed.")

def run_dynamic_analysis(length, num_elements, E, I, rho, A,
                         tau, num_steps, x0 = 0, dx0 = 0,
                         external_force = None,
                         constraint_type=ConstraintType,
                         connection_type=ConnectionType,
                         load_type=LoadType,
                         solve_type: SolvType = SolvType.DYNAMIC,
                         save_flag = True):
    """
    Run dynamic analysis for a portal frame with fixed connection.

    Parameters:
        length (float): Length of each beam.
        num_elements (int): Number of elements per beam.
        E (float): Young's modulus.
        I (float): Moment of inertia.
        rho (float): Density.
        A (float): Cross-sectional area.
        tau (float): Time step size for dynamic analysis.
        num_steps (int): Number of time steps to simulate.
        x0 (float or numpy.ndarray): Initial nodal displacement vector.
        dx0 (float or numpy.ndarray): Initial nodal speed vector.

        external_force (tuple or None): Tuple (F_value, M_value, q_value) specifying the external
            loads to be applied on beam_1, where:
                - F_value: Magnitude of point force (N)
                - M_value: Magnitude of bending moment (N·m)
                - q_value: Magnitude of distributed force (N/m)
            If None, no external force is applied.

        constraint_type (enum): Constraint type for supports. Options include:
            - ConstraintType.DISPLACEMENT (fix transverse displacement)
            - ConstraintType.AXIAL (fix axial displacement)
            - ConstraintType.ROTATION (fix rotation)

        connection_type (enum): Connection type between beams. Options include:
            - ConnectionType.Fix (fixed connection)
            - ConnectionType.Hinge (hinged connection)

        load_type (enum): Load type for the applied forces. Options include:
            - LoadType.F for point force,
            - LoadType.M for moment,
            - LoadType.q for distributed force.

        solve_type (SolvType): The dynamic solver type to be used. Options include:
            - SolvType.DYNAMIC: Newmark method
            - SolvType.EIGEN: Eigenvalue method (only available for homogenuous boundary conditions)

        save_flag (bool): whether to save the output vibration animation
    Returns:
        None. The function visualizes the dynamic response via an animation.
    """
    # Create beams:
    # beam_1: vertical beam (angle = π/2)
    # beam_2: horizontal beam (angle = 0, default)
    # beam_3: inclined in the opposite direction (angle = -π/2)
    beam_1 = Beam2D(length, E, A, rho, I, num_elements, angle=math.pi / 2)
    beam_2 = Beam2D(length, E, A, rho, I, num_elements)
    beam_3 = Beam2D(length, E, A, rho, I, num_elements, angle=-math.pi / 2)

    # Initialize the FEM framework model and add the beams.
    frame_work = FrameworkFEM()
    frame_work.add_beam(beam_1)
    frame_work.add_beam(beam_2)
    frame_work.add_beam(beam_3)

    # Add connections between beams:
    # The tuple (-1, 0) indicates that the last node of one beam connects to the first node of the next.
    frame_work.add_connection(beam_1, beam_2, (-1, 0), connection_type.Fix)
    frame_work.add_connection(beam_2, beam_3, (-1, 0), connection_type.Fix)

    # Add supports (constraints):
    # - Fix the left end of beam_1 (node index 0)
    # - Fix the right end of beam_3 (node index -1)
    frame_work.add_constraint(beam_1, 0, 0, constraint_type.DISPLACEMENT)
    frame_work.add_constraint(beam_1, 0, 0, constraint_type.AXIAL)
    frame_work.add_constraint(beam_1, 0, 0, constraint_type.ROTATION)
    frame_work.add_constraint(beam_3, -1, 0, constraint_type.DISPLACEMENT)
    frame_work.add_constraint(beam_3, -1, 0, constraint_type.AXIAL)
    frame_work.add_constraint(beam_3, -1, 0, constraint_type.ROTATION)

    # If an external force is provided, apply it on beam_1 at its right end.
    if external_force is not None:
        F_value, M_value, q_value = external_force
        frame_work.add_force(beam_1, (-1, F_value), load_type.F)
        frame_work.add_force(beam_1, (-1, M_value), load_type.M)
        # For distributed force, we use a function to define the value of load along the beam.
        frame_work.add_force(beam_1, lambda x: q_value, load_type.q)

    # Assemble the global stiffness and mass matrices for the structure.
    frame_work.assemble_frame_matrices()

    # Solve the dynamic problem using the specified solver type.
    # SolvType.DYNAMIC uses the Newmark method
    # SolveType.EIGEN uses the eigenvalue method (only suitable in the case of free vibration)
    frame_work.solv(tau=tau, num_steps=num_steps, x0=x0, dx0=dx0, sol_type=solve_type)

    # Visualize the dynamic response.
    import matplotlib
    matplotlib.use("WebAgg", force=True)
    frame_work.visualize(SolvType.DYNAMIC, save_flag=save_flag)


Fd_value = 1000000  # Magnitude of the point force load (N)
Md_value = 100000   # Magnitude of the bending moment (N·m)
qd_value = 100000   # Magnitude of the distributed force (N/m)
tau = 0.001       # Time step size for dynamic analysis (s)
num_steps = 500   # Number of time steps for dynamic analysis
external_force = (Fd_value, Md_value, qd_value)

# if the input external_force is None, then is free vibration
# here we use the free vibration case by setting the initial position
# feel free to change it to the forced vibration case, by setting "external_force=external_force"
print("Starting dynamic analysis...")
run_dynamic_analysis(length, num_elements, E, I, rho, A,
                     tau, num_steps, x0 = initial_displacement, dx0 = 0,
                     external_force = None,
                     solve_type=SolvType.DYNAMIC,
                     save_flag = False)
print("Dynamic analysis completed.")