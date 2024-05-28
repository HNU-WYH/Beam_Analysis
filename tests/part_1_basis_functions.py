import numpy as np
import matplotlib.pyplot as plt

#Question 1

def basis_func(x, h):
    bas_func = [
        lambda x, h: 1 - 3 * (x / h) ** 2 + 2 * (x / h) ** 3,
        lambda x, h: h * (x / h) * (x / h - 1) ** 2,
        lambda x, h: 3 * (x / h) ** 2 - 2 * (x / h) ** 3,
        lambda x, h: h * (x / h) ** 2 * (x / h - 1)
    ]
    return bas_func

def w_approx(x, u, L, n):

    """
    w is the approximation value.
    Calculated as: sum(x*basisfunct(x))
    including x start and end, both value and derivative

    """

    h = L / n  # length of each element
    x_nodes = np.linspace(0, L, n + 1)  # node positions
    bas_func = basis_func(x, h)  # basis functions for the x

    start_xi = int(x / h)  # start node index
    start = x_nodes[start_xi]  # start node position

    # w approximates by interpolation of basis functions
    w = 0  # initialize w

    # start node
    w += u[start_xi * 2] * bas_func[0](x - start, h)  # value
    w += u[start_xi * 2 + 1] * bas_func[1](x - start, h)  # derivative

    # end node
    if start_xi < n:
        w += u[start_xi * 2 + 2] * bas_func[2](x - start, h)  # value
        w += u[start_xi * 2 + 3] * bas_func[3](x - start, h)  # derivative

    return w

def u_sin(x_i):

    """
    Returns vector u of values and derivatives of sin(x),
    x are the node points

    """
    values = np.sin(x_i)
    derivatives = np.cos(x_i)
    u = np.empty(2 * len(x_i))  # Initialize u
    # Insert values and derivatives in u, intercalating them
    u[0::2] = values
    u[1::2] = derivatives
    return u

def test_plot(L,n,u,smoothness):

    """
    Plots the comparison of results.
    L, length
    n, number of elements
    u, vector with values and derivatives
    smoothness, number of x points

    """

    # original function
    x_points = np.linspace(0, L, smoothness)  # smooth enough sin function
    plt.plot(x_points, np.sin(x_points), label='Sin(x)')  # original

    # approximation using basis functions
    approx_values = np.array([w_approx(x, u, L, n) for x in x_points]) #get w for each x
    plt.plot(x_points, approx_values, label='Approximation', linestyle='-.')  # approx

    # Show points of the nodes
    x_nodes = np.linspace(0, L, n + 1)
    plt.scatter(x_nodes, np.sin(x_nodes), color='red', label='Nodes')

    #legend and show
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('w(x)')
    plt.title('Approximation of sin(x) using basis functions')
    plt.show()

if __name__ == "__main__":

  # Parameters (can edit)

    L = 10  # length
    n = 8  # number of elements
    smoothness = 100 # smoothness of the graph

    x_nodes = np.linspace(0, L, n + 1) # node positions in x
    u = u_sin(x_nodes)  # vector u, values and derivatives

    test_plot(L,n,u,smoothness) # plot results
