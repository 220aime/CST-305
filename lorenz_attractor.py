# lorenz_attractor.py
# Aime Serge Tuyishime
#4/16/2025
# Purpose: Solve and visualize the Lorenz attractor system

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the Lorenz system differential equations
def lorenz(t, state, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)                   # dx/dt
    dydt = x * (rho - z) - y                 # dy/dt
    dzdt = x * y - beta * z                  # dz/dt
    return [dxdt, dydt, dzdt]

# Initialize system parameters
sigma = 10.0                                 # Convection rate
rho = 28.0                                   # Temperature difference
beta = 8.0 / 3.0                             # Geometry constant
initial_state = [1.0, 1.0, 1.0]              # Initial x, y, z
t_span = (0, 40)                             # Time range
t_eval = np.linspace(t_span[0], t_span[1], 10000)  # Time steps

# Solve the Lorenz system using Runge-Kutta method
solution = solve_ivp(lorenz, t_span, initial_state, args=(sigma, rho, beta), t_eval=t_eval)

# Create a 3D plot of the Lorenz attractor
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(solution.y[0], solution.y[1], solution.y[2], lw=0.5)
ax.set_title("Lorenz Attractor")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
plt.show()