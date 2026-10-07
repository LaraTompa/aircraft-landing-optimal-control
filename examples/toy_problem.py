"""
Toy Optimal Control Problem
============================

Goal:
    Move a system from an initial state x(0) = 0 to a desired
    final state x(T) = 1 while minimizing the control effort.

Continuous-time formulation:

    minimize    J = ∫₀ᵀ u(t)² dt

    subject to  dx/dt = u(t)
                x(0) = 0
                x(T) = 1

where:
    x(t) : system state (here interpreted as position)
    u(t) : control input
    T    : final time

Discretization:

    The time interval [0, T] is divided into N intervals
    of length dt = T/N.

    The dynamics are discretized using forward Euler:

        x[k+1] = x[k] + dt * u[k]

    and the integral cost is approximated by:

        J ≈ Σₖ u[k]² * dt

The resulting finite-dimensional nonlinear optimization problem
is solved using CasADi and IPOPT.
"""

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

T = 1.0
N = 100
dt = T/N

# Variable definition
x = ca.MX.sym("x", N+1)
u = ca.MX.sym("u", N)

# Objective
J = ca.sumsqr(u)*dt

#Constraints
g =[]

g = ca.vertcat(
    x[0],
    *[x[k+1] - x[k] - dt*u[k] for k in range(N)],
    x[N] - 1
)

#Optimization variable
z = ca.vertcat(x,u)

#Dictionary
nlp = {"x": z, "f": J, "g": g}

#Solver
solver = ca.nlpsol("solver", "ipopt", nlp)

#Initial guess
z0 = [0] *(2*N+1)

#Solve
sol = solver(x0=z0, lbg=0, ubg=0)

print(sol["x"])

#Plot x(t) and u(t) 
# Extract optimal solution
z_opt = sol["x"].full().flatten()

x_opt = z_opt[:N+1]
u_opt = z_opt[N+1:]

# Time grids
t_x = np.linspace(0, T, N+1)
t_u = np.linspace(0, T-dt, N)

# Plot state x(t)
plt.figure()
plt.plot(t_x, x_opt)
plt.xlabel("Time [s]")
plt.ylabel("State x")
plt.title("Optimal State Trajectory")
plt.grid()
plt.show()

# Plot control u(t)
plt.figure()
plt.step(t_u, u_opt, where="post")
plt.xlabel("Time [s]")
plt.ylabel("Control u")
plt.title("Optimal Control")
plt.grid()
plt.show()