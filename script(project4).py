"""
Data Degradation Simulation in a Digital Storage System
CST-305: Project 4
Name: Aime Serge Tuyishime
Instructor: Professor Ricardo Citro
Date: 3/5/2025

Description:
- Reads and solves the ODE: dx/dt = Ax for a two-processor system.
- Uses SciPy's ODE solver to dynamically compute x1(t) and x2(t).
- Plots and visualizes data degradation over time.

Packages Used:
- NumPy (numerical calculations)
- SciPy (solving ODEs)
- Matplotlib (plotting results)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define system matrix A for the ODE dx/dt = Ax
A = np.array([[-1, 1], [1, -1]])  # Coefficients for data exchange between processors

# Define initial conditions
x0 = [1, -1]  # Initial values for x1 and x2 at t=0

# Define time range
t_span = (0, 2)  # Time from 0 to 2 seconds
t_eval = np.linspace(0, 2, 100)  # Time points for evaluation

# Define function to represent dx/dt = Ax
def system(t, x):
    return A @ x  # Matrix multiplication to get derivatives

# Solve the system of ODEs
solution = solve_ivp(system, t_span, x0, t_eval=t_eval)

# Extract results
t = solution.t
x1, x2 = solution.y  # x1(t) and x2(t) from the solution

# Plot the solutions
plt.figure(figsize=(8, 5))
plt.plot(t, x1, label=r'$x_1(t) = e^{-t/Z_0}$', color='b', linewidth=2)
plt.plot(t, x2, label=r'$x_2(t) = -e^{-t/Z_0}$', color='r', linestyle='dashed', linewidth=2)

# Labels and title
plt.xlabel('Time (t)', fontsize=12)
plt.ylabel('I/O Data', fontsize=12)
plt.title('Graph of $x_1(t)$ and $x_2(t)$ over Time (Data Degradation)', fontsize=14)
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
