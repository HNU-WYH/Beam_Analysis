import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

"""
BEAM PARAMETERS
"""

# Parameters
L = 10 # Length of the beam
n = 25  # Number of elements

# Constants
E = 200e9  # Young's modulus (Pa)
I = 1  # Beam moment of inertia (m^4)
rho = 7800.  # Density (kg/m^3)

# Other info
n_nodes = n + 1
x_nodes = np.linspace(0, L, n_nodes)  # Node positions
h = L / n  # Length of each element

# Element connectivity
elements = []
for i in range(n):
    elements.append([i, i + 1])  # Node indices of each element, [start, end]

"""
STIFNESS AND MASS MATRICES (Local and Global)
"""

# LOCAL MATRICES
def local_matrices(E_val, I_val, rho_val, h_val):
    # Define symbols
    E, I, h, rho, x = sp.symbols('E I h rho x')

    # Basis functions as symbolic expressions
    b1 = 1 - 3 * (x / h) ** 2 + 2 * (x / h) ** 3
    b2 = h * (x / h) * (x / h - 1) ** 2
    b3 = 3 * (x / h) ** 2 - 2 * (x / h) ** 3
    b4 = h * (x / h) ** 2 * (x / h - 1)

    b_functions = [b1, b2, b3, b4]

    # Initialize the stiffness and mass matrices
    Stiffness = sp.Matrix.zeros(4, 4)
    Mass = sp.Matrix.zeros(4, 4)

    # Compute S and M
    for i in range(4):
        for j in range(i, 4):
            # Stiffness matrix
            d2bi_dx2 = sp.diff(b_functions[i], x, 2)  # Second derivative i
            d2bj_dx2 = sp.diff(b_functions[j], x, 2)  # Second derivative j

            Stiffness[i, j] = E * I * sp.integrate(d2bi_dx2 * d2bj_dx2, (x, 0, h))
            Stiffness[j, i] = Stiffness[i, j]  # Symmetry

            # Mass matrix
            Mass[i, j] = rho * sp.integrate(b_functions[i] * b_functions[j], (x, 0, h))
            Mass[j, i] = Mass[i, j]  # Symmetry

    # Substitute numerical values into the symbolic matrices
    local_s = Stiffness.subs({h: h_val, E: E_val, I: I_val})
    local_m = Mass.subs({h: h_val, rho: rho_val})

    # Convert the resulting symbolic matrices to NumPy arrays
    local_s = np.array(local_s).astype(np.float64)
    local_m = np.array(local_m).astype(np.float64)

    return local_s, local_m

# GLOBAL MATRICES
def global_matrices():
    local_s, local_m = local_matrices(E, I, rho, h)  # Compute local matrices

    # Initialize global stiffness and mass matrices
    global_s = np.zeros((2 * n_nodes, 2 * n_nodes))
    global_m = np.zeros((2 * n_nodes, 2 * n_nodes))

    for element in elements:
        start = element[0]  # Assemble with 2 degrees of freedom
        global_s[2 * start:2 * start + 4, 2 * start:2 * start + 4] += local_s
        global_m[2 * start:2 * start + 4, 2 * start:2 * start + 4] += local_m

    return global_s, global_m

# Print the global stiffness matrix
global_S, global_M = global_matrices()
#print("Global stiffness matrix S:")
#print(np.round(global_S, 2))

"""
FORCE APPLIED

"""
# FORCE vector. Uniform load
f = np.zeros((2 * n_nodes, 1))
q = 10  # Newtons ------EDIT

# Compute the force exerted by the force density on each element
force_per_element = q * h / 2

# Distribute the force equally to the adjacent nodes of each element
for i in range(n):
    ielem = 2 * i  # Starting position for the current element
    f[ielem] += force_per_element  # Add force at the first node (start) of the element
    f[ielem + 2] += force_per_element  # Add force at the second node (end) of the element

"""
BOUNDARY CONDITIONS DEPENDING ON TYPE OF BEAM
"""
beamtype = "cantilever"  # "cantilever" or "simply_supported" -----EDIT

# Apply boundary conditions and solve for displacements
if beamtype == "simply_supported":
    # Boundary conditions: Simply supported beam
    K = global_S.copy()
    f_copy = f.copy()
    Nnodes = n_nodes
    print("Simply")

    #fixed end (2 d.f. displacement and rotation)

    K=np.delete(K,[2*Nnodes-2],0) # delete last row
    K=np.delete(K,[2*Nnodes-2],1) # delete last column
    f=np.delete(f,[2*Nnodes-2]) #delete last force entry

    K=np.delete(K,[0],0) #delete first row
    K=np.delete(K,[0],1) #delete first column
    f=np.delete(f,[0]) #delete first force entry


    # Solve system
    dx_vec = np.linalg.solve(K, f)

    # Reconstruct full displacement including boundary conditions
    dx=np.hstack([0., dx_vec[1:2*Nnodes-4:2], 0.])

else:
    # Boundary conditions: Cantilever beam
    K = global_S.copy()
    f_copy = f.copy()
    Nnodes = n_nodes
    print("Cantilever")

    #fixed end (2 d.f. displacement and rotation)
    K = np.delete(K, [0, 1], axis=0) # delete fixed degrees of freedom (rows 1 and 2)
    K = np.delete(K, [0, 1], axis=1) # delete fixed degrees of freedom (columns 1 and 2
    f = np.delete(f, [0, 1]) # delete entries (rows 1 and 2) in force vector (lhs)

    # Solve system
    dx_vec = np.linalg.solve(K, f)

    # Reconstruct full displacement including boundary conditions
    dx = np.hstack([0., dx_vec[0:2*Nnodes-2:2]])

# Plot displacement
plt.figure(figsize=(8, 6))
plt.plot(x_nodes, dx, label='Displacement', marker='o')
plt.xlabel('Position along beam / Node')
plt.ylabel('Displacement')
plt.title(f'{beamtype} Beam, constant load = {q} N/m')
plt.legend()
plt.grid()
plt.show()
