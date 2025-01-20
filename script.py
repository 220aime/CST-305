# CST-305 Project 1 - Visualize ODE with SciPy
#  Aime Serge Tuyishime
# Ricardo Citro
# 1/19/2024
# This program models and visualizes the decay rate of a computer system's performance over time.
# Using numpy and scipy, the program accounts for error estimates in solving the ODE and visualizes performance metrics.
# The 3D plot includes time (seconds) as the X-axis, decay rate k as the Y-axis, and system performance.
# The error tolerance parameters rtol and atol are chosen to balance accuracy and computational efficiency.

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting


def get_user_input():
    # Get decay rate from user, ensuring it is non-negative
    while True:
        try:
            k = float(input("Enter the decay rate (k): "))
            if k < 0:
                raise ValueError("Decay rate must be a non-negative number.")
            return k
        except ValueError as e:
            print(f"Invalid input. {e}")


def model(t, y, k):
    # Define the ODE model, dy/dt = -k * y, representing system performance degradation.
    return -k * y


def solve_ode(k):
    # Initial condition with clear units: operations per second at t=0
    y0 = [1000]  # Initial performance at t=0
    t_span = (0, 50)  # Time span for the solution (in seconds)
    t_eval = np.linspace(*t_span, 500)  # Generate 500 evenly spaced points for a smoother curve

    # Solve the ODE with specified error tolerances to ensure a balance between accuracy and efficiency
    # rtol=1e-6 ensures the relative error is below 0.0001%
    # atol=1e-9 helps in handling the precision of smaller values in the solution
    sol = solve_ivp(model, t_span, y0, args=(k,), t_eval=t_eval, rtol=1e-6, atol=1e-9)

    # The error tolerances are chosen based on the need for high accuracy in modeling system performance decay,
    # particularly important in systems where even small errors can lead to significant outcome deviations over time.
    return sol


def plot_solution(sol, k):
    # Initialize a 3D plot with appropriate labels and a title
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    k_values = np.full_like(sol.t, k)  # Create an array of k values same length as time array

    # Plot the 3D curve showing performance decay
    ax.plot(sol.t, k_values, sol.y[0], label=f'Performance decay over time with k={k}')
    ax.set_title('3D Visualization of System Performance Degradation')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Constant Decay Rate k')
    ax.set_zlabel('Performance (operations per second)')
    ax.legend()
    ax.grid(True)
    plt.show()


def main():
    k = get_user_input()
    sol = solve_ode(k)
    plot_solution(sol, k)


if __name__ == "__main__":
    main()
