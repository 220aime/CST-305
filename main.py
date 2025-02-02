"""
CST-305: Project 2 – Runge-Kutta-Fehlberg (RKF) for ODE
Author: Aime Serge Tuyishime
Date: 2025
Description:
    - This script implements the RKF45 method for solving the ODE: dy/dx = -y^2 / x
    - It calculates and compares numerical solutions with an analytical solution.
    - The program tests performance by running RKF on 2000 steps.
    - Outputs include numerical values, computational time, error analysis, and plots.
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Define the given ODE function
def f(x, y):
    """Computes dy/dx = -y^2 / x"""
    return -y ** 2 / x


# RKF45 Coefficients
c = [0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2]
a = [
    [],
    [1 / 4],
    [3 / 32, 9 / 32],
    [1932 / 2197, -7200 / 2197, 7296 / 2197],
    [439 / 216, -8, 3680 / 513, -845 / 4104],
    [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40]
]
b = [16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55]

# Initial Conditions
x0, y0 = 1, 3
h = 0.01
n_steps = 6  # For manual verification


def run_rkf45(x0, y0, h, steps):
    """Runge-Kutta-Fehlberg (RKF45) numerical solver for ODE."""
    x_values, y_values = [x0], [y0]

    for i in range(1, steps + 1):
        x_n, y_n = x_values[-1], y_values[-1]

        k1 = h * f(x_n, y_n)
        k2 = h * f(x_n + c[1] * h, y_n + a[1][0] * k1)
        k3 = h * f(x_n + c[2] * h, y_n + a[2][0] * k1 + a[2][1] * k2)
        k4 = h * f(x_n + c[3] * h, y_n + a[3][0] * k1 + a[3][1] * k2 + a[3][2] * k3)
        k5 = h * f(x_n + c[4] * h, y_n + a[4][0] * k1 + a[4][1] * k2 + a[4][2] * k3 + a[4][3] * k4)
        k6 = h * f(x_n + c[5] * h, y_n + a[5][0] * k1 + a[5][1] * k2 + a[5][2] * k3 + a[5][3] * k4 + a[5][4] * k5)

        y_next = y_n + b[0] * k1 + b[1] * k2 + b[2] * k3 + b[3] * k4 + b[4] * k5 + b[5] * k6
        x_next = x_n + h

        x_values.append(x_next)
        y_values.append(y_next)
        print(f"Step {i}: x = {x_next:.4f}, y = {y_next:.6f}")

    return x_values, y_values


# Run for first 6 steps to match manual calculations
x_values, y_values = run_rkf45(x0, y0, h, n_steps)


# Function for analytical solution
def exact_solution(x):
    """Exact solution y(x) = 3 / (x + 2)"""
    return 3 / (x + 2)


# Compute exact values for comparison
exact_values = [exact_solution(x) for x in x_values]
absolute_errors = np.abs(np.array(y_values) - np.array(exact_values))
relative_errors = np.abs(absolute_errors / np.array(exact_values))

# Large-scale computation for performance testing
n_steps_large = 2000
print(f"\nRunning RKF45 for {n_steps_large} steps...")

start_time = time.time()
x_large, y_large = run_rkf45(x0, y0, h, n_steps_large)
end_time = time.time()

computation_time = end_time - start_time
print(f"Completed {n_steps_large} steps in {computation_time:.6f} seconds.")

# Solve using scipy.integrate.solve_ivp for comparison
sol = solve_ivp(f, [x0, x_large[-1]], [y0], t_eval=x_large, method='RK45')

# Plot RKF45 vs. Scipy RK45
plt.figure(figsize=(12, 6))
plt.plot(x_large, y_large, label="RKF45 Solution", color='blue')
plt.plot(sol.t, sol.y[0], '--', label="Scipy RK45 Solution", color='orange')
plt.xlabel("x values")
plt.ylabel("y values")
plt.title("Comparison of RKF45 and Scipy RK45")
plt.legend()
plt.grid()
plt.xlim(min(x_large), max(x_large))  # Ensuring full range is shown
plt.ylim(min(min(y_large), min(sol.y[0])), max(max(y_large), max(sol.y[0])))
plt.show()

# Plot absolute error with full range
plt.figure(figsize=(12, 6))
plt.plot(x_large, np.abs(np.array(y_large) - np.array([exact_solution(x) for x in x_large])),
         label="Absolute Error", color='red')
plt.xlabel("x values")
plt.ylabel("Error")
plt.title("Absolute Error in RKF45 Solution")
plt.legend()
plt.grid()
plt.xlim(min(x_large), max(x_large))  # Full range
plt.ylim(min(np.abs(y_large - np.array([exact_solution(x) for x in x_large]))),
         max(np.abs(y_large - np.array([exact_solution(x) for x in x_large]))))  # Full range
plt.show()

# Plot relative error with full range
plt.figure(figsize=(12, 6))
plt.plot(x_large, np.abs(np.array(y_large) - np.array([exact_solution(x) for x in x_large])) /
         np.abs(np.array([exact_solution(x) for x in x_large])),
         label="Relative Error", color='green')
plt.xlabel("x values")
plt.ylabel("Error")
plt.title("Relative Error in RKF45 Solution")
plt.legend()
plt.grid()
plt.xlim(min(x_large), max(x_large))  # Full range
plt.ylim(min(np.abs(np.array(y_large) - np.array([exact_solution(x) for x in x_large])) /
             np.abs(np.array([exact_solution(x) for x in x_large]))),
         max(np.abs(np.array(y_large) - np.array([exact_solution(x) for x in x_large])) /
             np.abs(np.array([exact_solution(x) for x in x_large]))))  # Full range
plt.show()


# Save results for further analysis
large_scale_df = pd.DataFrame({'Step': range(n_steps_large + 1), 'x_n': x_large, 'y_n': y_large})
large_scale_df.to_csv("rkf45_results_large.csv", index=False)
print("\nResults saved to 'rkf45_results_large.csv'.")
