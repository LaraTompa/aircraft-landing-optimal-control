
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

from airplane_animator import animate_airplane


# ============================================================
# 1. Aircraft parameters (illustrative, not validated A320 data)
# ============================================================

m = 73500           # mass [kg]
g = 9.81              # gravitational acceleration [m/s^2]
rho = 1.225           # air density [kg/m^3]
S = 122.4             # wing reference area [m^2]

CD0 = 0.025           # zero-lift drag coefficient
k = 0.045             # induced drag factor

T_max = 90070      # maximum thrust [N]
CL = 1.6

T_scale = 100000.0


# ============================================================
# 2. Initial and final conditions
# ============================================================

tf = 70.0
N = 100
dt = tf / N

gamma0 = np.deg2rad(-3.0)
gamma_f = np.deg2rad(-3.0)

x0 = np.array([
    0.0,       # horizontal position [m]
    200.0,     # altitude [m]
    80.0,      # airspeed [m/s]
    gamma0     # flight-path angle [rad]
])

xf = np.array([
    5420.0,    # touchdown position [m]
    0.0,       # touchdown altitude [m]
    74.0,      # touchdown airspeed [m/s]
    gamma_f    # touchdown flight-path angle [rad]
])


# ============================================================
# 3. Define the point-mass equations of motion
# ============================================================

def aircraft_dynamics(x, u):

    s = x[0]
    h = x[1]
    V = x[2]
    gamma = x[3]

    T = u[0]

    qbar = 0.5 * rho * V**2

    CD = CD0 + k * CL**2

    L = qbar * S * CL
    D = qbar * S * CD

    s_dot = V * ca.cos(gamma)
    h_dot = V * ca.sin(gamma)

    V_dot = (T - D) / m - g * ca.sin(gamma)

    gamma_dot = (
        L - m * g * ca.cos(gamma)
    ) / (m * V)

    return ca.vertcat(
        s_dot,
        h_dot,
        V_dot,
        gamma_dot
    )


# ============================================================
# 4. RK4 integration
# ============================================================

def rk4_step(x, u):

    k1 = aircraft_dynamics(x, u)
    k2 = aircraft_dynamics(x + dt/2 * k1, u)
    k3 = aircraft_dynamics(x + dt/2 * k2, u)
    k4 = aircraft_dynamics(x + dt * k3, u)

    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


# ============================================================
# 5. Set up the optimal control problem
# ============================================================

opti = ca.Opti()

# State matrix: 4 x (N+1)
X = opti.variable(4, N + 1)

# Control matrix: thrust, 1 x N
U = opti.variable(1, N)

J = 0

for k_idx in range(N):

    x_k = X[:, k_idx]
    u_k = U[:, k_idx]

    # Dynamics constraint
    x_next = rk4_step(x_k, u_k)

    opti.subject_to(
        X[:, k_idx + 1] == x_next
    )

    T_k = u_k[0]

    # Integral of squared normalized thrust
    J += dt * (
        (T_k / T_scale)**2
    )

    # Control bounds
    opti.subject_to(opti.bounded(0.0, T_k, T_max))


# ============================================================
# 6. Boundary conditions
# ============================================================

opti.subject_to(X[:, 0] == x0)
opti.subject_to(X[:, N] == xf)


# ============================================================
# 7. State constraints
# ============================================================

for k_idx in range(N + 1):

    opti.subject_to(X[1, k_idx] >= 0.0)

    opti.subject_to(
        opti.bounded(55.0, X[2, k_idx], 85.0)
    )

    opti.subject_to(
        opti.bounded(
            np.deg2rad(-8.0),
            X[3, k_idx],
            np.deg2rad(2.0)
        )
    )


# ============================================================
# 8. Objective
# ============================================================

opti.minimize(J)


# ============================================================
# 9. Initial guess
# ============================================================

for i in range(4):
    guess = np.linspace(x0[i], xf[i], N + 1)
    opti.set_initial(X[i, :], guess)

opti.set_initial(U[0, :], 45000.0)


# ============================================================
# 10. Solve
# ============================================================

opti.solver(
    "ipopt",
    {"expand": True},
    {
        "max_iter": 3000,
        "tol": 1e-7,
        "print_level": 5
    }
)

sol = opti.solve()

X_opt = sol.value(X)
U_opt = np.asarray(sol.value(U)).reshape(-1)

time = np.linspace(0, tf, N + 1)
time_u = time[:-1]


# ============================================================
# 11. Plot results
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Flight trajectory
axes[0, 0].plot(X_opt[0, :], X_opt[1, :])
axes[0, 0].set_xlabel("Horizontal distance [m]")
axes[0, 0].set_ylabel("Altitude [m]")
axes[0, 0].set_title("Landing trajectory")
axes[0, 0].grid()

# Airspeed
axes[0, 1].plot(time, X_opt[2, :])
axes[0, 1].set_xlabel("Time [s]")
axes[0, 1].set_ylabel("Airspeed [m/s]")
axes[0, 1].set_title("Airspeed")
axes[0, 1].grid()

# Flight-path angle
axes[1, 0].plot(time, np.rad2deg(X_opt[3, :]))
axes[1, 0].set_xlabel("Time [s]")
axes[1, 0].set_ylabel("Flight-path angle [deg]")
axes[1, 0].set_title("Flight-path angle")
axes[1, 0].grid()

# Controls
axes[1, 1].plot(time_u, U_opt / 1000, label="Thrust")
axes[1, 1].set_xlabel("Time [s]")
axes[1, 1].set_ylabel("Thrust [kN]")
axes[1, 1].set_title("Control inputs")
axes[1, 1].grid()

fig.tight_layout()

# The point-mass model does not solve for pitch or the wheel position.
# Assume a constant angle of attack and use the model position as the wheel contact.
playback_speed = 5.0
animation_interval = 30  # ms
animation_frames = int(np.ceil(tf * 1000 / (playback_speed * animation_interval))) + 1
animation_time = np.linspace(time[0], time[-1], animation_frames)
animation_x = np.interp(animation_time, time, X_opt[0, :])
animation_y = np.interp(animation_time, time, X_opt[1, :])
animation_pitch = np.interp(animation_time, time, X_opt[3, :]) + np.deg2rad(5.0)
animation = animate_airplane(
    animation_x, animation_y, animation_pitch,
    target=xf[0], interval=animation_interval, follow=True,
    title="Optimized landing trajectory (5x speed)",
)
plt.show()


print("Optimal objective:", sol.value(J))
print("Final state:", X_opt[:, -1])
