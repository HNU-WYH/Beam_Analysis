import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src.beam import Beam
from src.fem_beam import FEM, ConstraintType
from config import LoadType, uniform_load_function, SolvType
# matplotlib.use("WebAgg", force=True)

def cantilever_beam(load_type, solv_type = SolvType.STATIC):
    # Initialize a simple beam with 5 nodes and length 10
    E, I, rho = 210 * 10 ** 9, 36.92 * 10 ** (-6), 42.3
    A, length = 5383 * 10 ** (-6), 5

    # x- values
    x_values = np.linspace(0, length, 500)

    num_list = [1, 5, 50]

    if load_type == LoadType.F and solv_type == SolvType.STATIC:

        # calculate y_values
        y_list = []
        for num_elements in num_list:
            # Initialize the beam
            beam = Beam(length, E, rho, I, num_elements)

            # Initialize FEM model
            fem = FEM(beam)

            # Apply a force of 500N at the position of 5m
            fem.apply_force((5, 500), load_type)

            # Add constraints: displacement at node 0 is 0, rotation at node 4 is 0
            fem.add_constraint(0, 0, ConstraintType.DISPLACEMENT)
            fem.add_constraint(0, 0, ConstraintType.ROTATION)

            # Solve the static system
            fem.solv()

            y_list.append(fem.assess_dis(fem.stsol, x_values))

        # calculate theoretical displacement
        dis = lambda x: 500/(6*E*I) * (3 * length * x**2 - x**3)
        y_list.append([dis(x_value) for x_value in x_values])
        y_list = 1000 * np.array(y_list)

        # Plotting the results
        plt.figure(figsize=(10, 6))
        colors = ['r', 'g', 'b', 'y']
        labels = [f'Numerical Solution (num_elements={n})' for n in num_list] + ['Theoretical Solution']

        for y_values, color, label in zip(y_list, colors, labels):
            plt.plot(x_values, y_values, color=color, label=label)

        plt.xlabel('Position along the beam (x) [m]')
        plt.ylabel('Displacement [mm]')
        plt.title('Comparison of Theoretical and Numerical Solutions for Cantilever Beam Displacement')
        plt.legend()
        plt.grid(True)
        # Move x-axis to the top
        ax = plt.gca()
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        # Invert y-axis direction
        ax.invert_yaxis()
        plt.show()

        # # calculate the error
        # # Calculate error between numerical and theoretical solutions
        # errors = []
        # for y_numerical in y_list[:-1]:  # Exclude the theoretical solution itself
        #     error = 100 * np.abs(y_numerical - y_list[-1])/y_list[-1]
        #     errors.append(error)
        #
        # # Plotting the error
        # plt.figure(figsize=(10, 6))
        # for error, color, label in zip(errors, colors[:-1], labels[:-1]):
        #     plt.plot(x_values[1:], error[1:], color=color, label=f'Error (num_elements={label.split("=")[-1]})')
        #
        # plt.xlabel('Position along the beam (x) [m]')
        # plt.ylabel('Error(%)')
        # plt.title('Error between Numerical and Theoretical Solutions for Cantilever Beam Displacement')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

    elif load_type == LoadType.q and solv_type == SolvType.STATIC:
        dis_load = lambda x: 100 * uniform_load_function(x)

        # calculate y_values
        y_list = []
        for num_elements in num_list:
            # Initialize the beam
            beam = Beam(length, E, rho, I, num_elements)

            # Initialize FEM model
            fem = FEM(beam)

            # Apply a force of 500N at the position of 5m
            fem.apply_force(dis_load, load_type)

            # Add constraints: displacement at node 0 is 0, rotation at node 4 is 0
            fem.add_constraint(0, 0, ConstraintType.DISPLACEMENT)
            fem.add_constraint(0, 0, ConstraintType.ROTATION)

            # Solve the static system
            fem.solv()

            y_list.append(fem.assess_dis(fem.stsol, x_values))

        # calculate theoretical displacement
        dis = lambda x: 100 / (24 * E * I) * (x ** 4 - 4 * length * x ** 3 + 6 * length ** 2 * x ** 2)
        y_list.append([dis(x_value) for x_value in x_values])
        y_list = 1000 * np.array(y_list)

        # Plotting the results
        plt.figure(figsize=(10, 6))
        colors = ['r', 'g', 'b', 'y']
        labels = [f'Numerical Solution (num_elements={n})' for n in num_list] + ['Theoretical Solution']

        for y_values, color, label in zip(y_list, colors, labels):
            plt.plot(x_values, y_values, color=color, label=label)

        plt.xlabel('Position along the beam (x) [m]')
        plt.ylabel('Displacement [mm]')
        plt.title('Comparison of Theoretical and Numerical Solutions for Cantilever Beam Displacement')
        plt.legend()
        plt.grid(True)
        # Move x-axis to the top
        ax = plt.gca()
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        # Invert y-axis direction
        ax.invert_yaxis()
        plt.show()

        # calculate the error
        # Calculate error between numerical and theoretical solutions
        errors = []
        for y_numerical in y_list[:-1]:  # Exclude the theoretical solution itself
            error = np.abs(y_numerical - y_list[-1])
            errors.append(error)

        # Plotting the error
        plt.figure(figsize=(10, 6))
        for error, color, label in zip(errors, colors[:-1], labels[:-1]):
            plt.plot(x_values[1:], error[1:], color=color, label=f'Error (num_elements={label.split("=")[-1]})')

        plt.xlabel('Position along the beam (x) [mm]')
        plt.ylabel('Error(mm)')
        plt.title('Error between Numerical and Theoretical Solutions for Cantilever Beam Displacement')
        plt.legend()
        plt.grid(True)
        plt.show()

    else:
        for tau in [0.001]:
            for num_elements in [1, 5, 50]:
                # # compare with EIgen value method & FEM method
                matplotlib.use("WebAgg", force=True)

                # Initialize the beam
                beam = Beam(length, E, rho, I, num_elements)
                fem = FEM(beam)

                # Add constraints: displacement at node 0 is 0, rotation at node 0 is 0
                fem.add_constraint(0, 0, ConstraintType.DISPLACEMENT)
                fem.add_constraint(0, 0, ConstraintType.ROTATION)

                # Compute the displacement under F = 500N as the initial position
                temp = Beam(length, E, rho, I, num_elements)
                temp = FEM(temp)
                temp.apply_force((5, 500), LoadType.F)
                temp.add_constraint(0, 0, ConstraintType.DISPLACEMENT)
                temp.add_constraint(0, 0, ConstraintType.ROTATION)
                temp.solv()

                # Initial condition
                y0 = temp.stsol[:2 * temp.beam.num_nodes]

                # Solve the dynamic system using FEM method
                fem.solv(tau=tau, num_steps=1000, x0=y0, dx0=0, sol_type=SolvType.DYNAMIC)
                dysol1 = np.array([fem.assess_dis(fem.dysol[i], x_values) for i in range(fem.dysol.shape[0])])

                # Solve the dynamic system using Eigen method
                fem.solv(tau=tau, num_steps=1000, x0=y0, dx0=0, sol_type=SolvType.EIGEN)
                dysol2 = np.array([fem.assess_dis(fem.dysol[i], x_values) for i in range(fem.dysol.shape[0])])

                # Visualization
                # Create a figure and axis for the animation
                fig, ax = plt.subplots()
                line1, = ax.plot([], [], 'r-', label='Dynamic Solution (FEM)')
                line2, = ax.plot([], [], 'b-', label='Dynamic Solution (Eigen)')
                ax.set_xlim(0, x_values[-1])
                ax.set_ylim(min(np.min(dysol1), np.min(dysol2)), max(np.max(dysol1), np.max(dysol2)))
                ax.set_xlabel('Position along the beam (x) [m]')
                ax.set_ylabel('Displacement [mm]')
                ax.set_title(f'Comparison of FEM and Eigen Solutions Over Time\nTau={tau}, Num Elements={num_elements}')
                ax.legend(loc="upper left")

                # Update function for animation
                def update(frame):
                    line1.set_data(x_values, dysol1[frame, :])
                    line2.set_data(x_values, dysol2[frame, :])
                    return line1, line2

                # Create animation
                ani = animation.FuncAnimation(fig, update, frames=min(dysol1.shape[0], dysol2.shape[0]), blit=True,
                                              repeat=False)

                # save animation
                ani.save(f'..\output\eigen_vs_fem_Tau={tau}_NumEle={num_elements}.gif', writer='imagemagick', fps=30)

                # Display the animation
                plt.show()

if __name__ == "__main__":
    # Run the test
    cantilever_beam(LoadType.F)
    cantilever_beam(LoadType.q)
    cantilever_beam(None, SolvType.DYNAMIC)
