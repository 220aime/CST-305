
# Project 3 – Green's Function and ODE with IVP
# Aime Serge Tuyishime
# Professor Ricardo Citro
# Feb 16. 2025
# Principles of Modeling and Simulation Lecture & Lab | CST-305
# Objective: Use RKF to assess the power of a computing system
# Description: 
#This project uses Green’s function to simulate propagation of data 
# throughout a network inspired by isotherms  in thermodynamics. 
# It uses ODE, Green, and SciPy to solve and visualize the propagation
# across network nodes, given initial values.

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the ODE
def ode(t, y, a, b):
    dydt = y[1]  # y[0] = y(t), y[1] = dy/dt
    d2ydt2 = -a * y[1] - b * y[0] + np.sin(t)  # Forcing function f(t) = sin(t)
    return [dydt, d2ydt2]

# Initial conditions
y0 = [0, 0]  # y(0) = 0, y'(0) = 0

# Parameters
a = 1.0  # Damping factor
b = 1.0  # Stiffness factor

# Solve the ODE
t_span = [0, 10]  # Time range
t_eval = np.linspace(0, 10, 1000)  # Points at which to store the solution
sol = solve_ivp(ode, t_span, y0, args=(a, b), t_eval=t_eval)

# Plot the solution
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label="y(t)", color="blue")
plt.xlabel("Time (t)", fontsize=12)
plt.ylabel("Data Propagation (y)", fontsize=12)
plt.title("Data Propagation in a Network", fontsize=14)
plt.legend()
plt.grid()
plt.show()