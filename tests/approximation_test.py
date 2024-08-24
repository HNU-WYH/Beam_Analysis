import numpy as np
import matplotlib.pyplot as plt
from src.utils.local_matrix import LocalElement


def approximation_test(num_elements = 5, plot_precision = 100):
    domain = [0, 2 * np.pi]
    x_values = np.linspace(domain[0], domain[1], plot_precision)
    node_list = np.linspace(domain[0], domain[1], num_elements + 1)

    # Create u as coefficients of the basis functions to approximate sin(x)
    u = np.zeros(2 * len(node_list))
    u[0::2] = np.sin(node_list)  # coefficients for phi_2i-1
    u[1::2] = np.cos(node_list)  # coefficients for phi_2i

    # Get the approximate function
    approx_values = np.array([LocalElement.app_func(x, u, domain, num_elements) for x in x_values])

    # Original function
    original_function = np.sin(x_values)
    x = np.linspace(domain[0], domain[1], 200)
    sin_x = np.sin(x)

    # Plot the original function and the approximation
    plt.plot(x, sin_x, label='Original sin(x)')
    plt.plot(x_values, approx_values, label='Approximation', linestyle='--')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Approximation of sin(x) using basis functions')
    plt.show()

if __name__ == "__main__":
    approximation_test(5,100)
