import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

from scripts.airplane_animator import animate_airplane
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Aircraft parameters
#    Illustrative values, not validated A320 data
# ============================================================

m = 73500.0
g = 9.81
rho = 1.225
S = 122.4

CD0 = 0.025
k_drag = 0.045

T_max = 90070.0

CL_min = 0.8
CL_max = 1.8

T_scale = 100000.0


# ============================================================
# 2. Initial and final conditions
# ============================================================

tf = 65.0
N = 200
dt = tf / N

gamma0 = np.deg2rad(-3.0)
gamma_f = np.deg2rad(-3.0)

x0 = np.array([
    0.0,       # Horizontal position [m]
    300.0,     # Altitude [m]
    95.0,      # Airspeed [m/s]
    gamma0     # Flight-path angle [rad]
])

xf = np.array([
    5420.0,    # Touchdown position [m]
    0.0,       # Touchdown altitude [m]
    80.0,      # Touchdown airspeed [m/s]
    gamma_f    # Touchdown flight-path angle [rad]
])


# ============================================================
# 3. Wind field parameters
#    Stationary Gaussian vertical downdraft
# ============================================================

W_max = 15.0

s_center = 3500.0
h_center = 150.0

sigma_s = 800.0
sigma_h = 150.0


def wind_field(s, h, t):
    """
    Gaussian vertical wind field.

    Coordinates:
        s: horizontal position [m]
        h: altitude [m], positive upwards

    Negative w_h means downward wind.
    """

    exponent = (
        -((s - s_center) / sigma_s)**2
        -((h - h_center) / sigma_h)**2
    )

    w_s = 0.0
    w_h = -W_max * ca.exp(exponent)

    return ca.vertcat(w_s, w_h)


# ============================================================
# 4. Wind-aware aircraft dynamics
# ============================================================

def aircraft_dynamics(x, u, t):

    # States
    s = x[0]
    h = x[1]
    V = x[2]
    gamma = x[3]

    # Controls
    T = u[0]
    CL = u[1]

    # --------------------------------------------------------
    # Aerodynamic forces
    # --------------------------------------------------------

    qbar = 0.5 * rho * V**2

    CD = CD0 + k_drag * CL**2

    L = qbar * S * CL
    D = qbar * S * CD

    # --------------------------------------------------------
    # Wind velocity
    # --------------------------------------------------------

    w = wind_field(s, h, t)

    w_s = w[0]
    w_h = w[1]

    # --------------------------------------------------------
    # Ground-relative velocity
    # --------------------------------------------------------

    s_dot = V * ca.cos(gamma) + w_s
    h_dot = V * ca.sin(gamma) + w_h

    # --------------------------------------------------------
    # Material derivative of the wind field
    #
    # The wind field is stationary and purely vertical:
    #
    # a_w_h = dw_h/ds * s_dot
    #       + dw_h/dh * h_dot
    # --------------------------------------------------------

    dw_h_ds = (
        -2.0 * (s - s_center) / sigma_s**2
    ) * w_h

    dw_h_dh = (
        -2.0 * (h - h_center) / sigma_h**2
    ) * w_h

    a_w_h = (
        dw_h_ds * s_dot
        + dw_h_dh * h_dot
    )

    # Horizontal wind acceleration is zero
    a_w_s = 0.0

    # --------------------------------------------------------
    # Wind acceleration in tangential and normal directions
    # --------------------------------------------------------

    a_w_t = (
        ca.cos(gamma) * a_w_s
        + ca.sin(gamma) * a_w_h
    )

    a_w_n = (
        -ca.sin(gamma) * a_w_s
        + ca.cos(gamma) * a_w_h
    )

    # --------------------------------------------------------
    # Equations of motion
    # --------------------------------------------------------

    V_dot = (
        (T - D) / m
        - g * ca.sin(gamma)
        - a_w_t
    )

    gamma_dot = (
        (L - m * g * ca.cos(gamma)) / (m * V)
        - a_w_n / V
    )

    return ca.vertcat(
        s_dot,
        h_dot,
        V_dot,
        gamma_dot
    )


# ============================================================
# 5. RK4 integration
# ============================================================

def rk4_step(x, u, t):

    k1 = aircraft_dynamics(x, u, t)

    k2 = aircraft_dynamics(
        x + dt / 2.0 * k1,
        u,
        t + dt / 2.0
    )

    k3 = aircraft_dynamics(
        x + dt / 2.0 * k2,
        u,
        t + dt / 2.0
    )

    k4 = aircraft_dynamics(
        x + dt * k3,
        u,
        t + dt
    )

    return x + dt / 6.0 * (
        k1 + 2.0 * k2 + 2.0 * k3 + k4
    )


# ============================================================
# 6. Set up the optimal control problem
# ============================================================

opti = ca.Opti()

# State matrix: 4 x (N+1)
X = opti.variable(4, N + 1)

# Control matrix: 2 x N
# U[0,:] = thrust
# U[1,:] = lift coefficient
U = opti.variable(2, N)

# Objective functions
J1 = 0
J2 = 0


# ============================================================
# 7. Dynamics, objectives and control constraints
# ============================================================

for k_idx in range(N):

    x_k = X[:, k_idx]
    u_k = U[:, k_idx]

    t_k = k_idx * dt

    # RK4 dynamics constraint
    x_next = rk4_step(x_k, u_k, t_k)

    opti.subject_to(
        X[:, k_idx + 1] == x_next
    )

    T_k = u_k[0]

    # J1: Integral of squared normalized thrust
    J1 += dt * (T_k / T_scale)**2

    # J2: Integral of normalized thrust
    J2 += dt * (T_k / T_scale)

    # Control constraints
    opti.subject_to(
        opti.bounded(0.0, T_k, T_max)
    )

    opti.subject_to(
        opti.bounded(CL_min, u_k[1], CL_max)
    )


# ============================================================
# 8. Boundary conditions
# ============================================================

opti.subject_to(X[:, 0] == x0)
opti.subject_to(X[:, N] == xf)


# ============================================================
# 9. State constraints
# ============================================================

for k_idx in range(N + 1):

    # Altitude
    opti.subject_to(
        X[1, k_idx] >= 0.0
    )

    # Airspeed
    opti.subject_to(
        opti.bounded(
            55.0,
            X[2, k_idx],
            95.0
        )
    )

    # Flight-path angle
    opti.subject_to(
        opti.bounded(
            np.deg2rad(-8.0),
            X[3, k_idx],
            np.deg2rad(2.0)
        )
    )


# ============================================================
# 10. Objective selection
# ============================================================

# alpha = 1: minimize J1
# alpha = 0: minimize J2

alpha = opti.parameter()

opti.minimize(
    alpha * J1 + (1.0 - alpha) * J2
)


# ============================================================
# 11. Initial guess
# ============================================================

for i in range(4):

    guess = np.linspace(
        x0[i],
        xf[i],
        N + 1
    )

    opti.set_initial(X[i, :], guess)


opti.set_initial(U[0, :], 45000.0)
opti.set_initial(U[1, :], 1.4)


# ============================================================
# 12. IPOPT solver
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


# ============================================================
# 13. Solve with J1: squared thrust
# ============================================================

print("\n============================================")
print("Solving optimization problem with J1")
print("Minimize integral of squared normalized thrust")
print("============================================")

opti.set_value(alpha, 1.0)

sol_J1 = opti.solve()

X_J1 = sol_J1.value(X)
U_J1 = np.asarray(sol_J1.value(U))

J1_opt = sol_J1.value(J1)
J2_at_J1 = sol_J1.value(J2)


# ============================================================
# 14. Solve with J2: linear thrust
# ============================================================

print("\n============================================")
print("Solving optimization problem with J2")
print("Minimize integral of normalized thrust")
print("============================================")

# Warm-start with the J1 solution
opti.set_initial(X, X_J1)
opti.set_initial(U, U_J1)

opti.set_value(alpha, 0.0)

sol_J2 = opti.solve()

X_J2 = sol_J2.value(X)
U_J2 = np.asarray(sol_J2.value(U))

J1_at_J2 = sol_J2.value(J1)
J2_opt = sol_J2.value(J2)


# ============================================================
# 15. Time vectors
# ============================================================

time = np.linspace(0.0, tf, N + 1)
time_u = time[:-1]


# ============================================================
# 16. Evaluate vertical wind along both trajectories
# ============================================================

def evaluate_vertical_wind(X_solution):

    return (
        -W_max
        * np.exp(
            -((X_solution[0, :] - s_center) / sigma_s)**2
            -((X_solution[1, :] - h_center) / sigma_h)**2
        )
    )


wind_J1 = evaluate_vertical_wind(X_J1)
wind_J2 = evaluate_vertical_wind(X_J2)


# ============================================================
# 17. Plot comparison
# ============================================================

fig, axes = plt.subplots(3, 2, figsize=(14, 12))


# ------------------------------------------------------------
# Landing trajectories
# ------------------------------------------------------------

axes[0, 0].plot(
    X_J1[0, :],
    X_J1[1, :],
    label="J1: Squared thrust"
)

axes[0, 0].plot(
    X_J2[0, :],
    X_J2[1, :],
    "--",
    label="J2: Linear thrust"
)

axes[0, 0].scatter(
    [s_center],
    [h_center],
    marker="x",
    s=80,
    label="Downdraft centre"
)

axes[0, 0].set_xlabel("Horizontal distance [m]")
axes[0, 0].set_ylabel("Altitude [m]")
axes[0, 0].set_title("Landing trajectories")
axes[0, 0].grid()
axes[0, 0].legend()


# ------------------------------------------------------------
# Airspeed
# ------------------------------------------------------------

axes[0, 1].plot(
    time,
    X_J1[2, :],
    label="J1: Squared thrust"
)

axes[0, 1].plot(
    time,
    X_J2[2, :],
    "--",
    label="J2: Linear thrust"
)

axes[0, 1].set_xlabel("Time [s]")
axes[0, 1].set_ylabel("Airspeed [m/s]")
axes[0, 1].set_title("Airspeed comparison")
axes[0, 1].grid()
axes[0, 1].legend()


# ------------------------------------------------------------
# Flight-path angle
# ------------------------------------------------------------

axes[1, 0].plot(
    time,
    np.rad2deg(X_J1[3, :]),
    label="J1: Squared thrust"
)

axes[1, 0].plot(
    time,
    np.rad2deg(X_J2[3, :]),
    "--",
    label="J2: Linear thrust"
)

axes[1, 0].set_xlabel("Time [s]")
axes[1, 0].set_ylabel("Flight-path angle [deg]")
axes[1, 0].set_title("Flight-path angle comparison")
axes[1, 0].grid()
axes[1, 0].legend()


# ------------------------------------------------------------
# Thrust
# ------------------------------------------------------------

axes[1, 1].plot(
    time_u,
    U_J1[0, :] / 1000.0,
    label="J1: Squared thrust"
)

axes[1, 1].plot(
    time_u,
    U_J2[0, :] / 1000.0,
    "--",
    label="J2: Linear thrust"
)

axes[1, 1].set_xlabel("Time [s]")
axes[1, 1].set_ylabel("Thrust [kN]")
axes[1, 1].set_title("Thrust comparison")
axes[1, 1].grid()
axes[1, 1].legend()


# ------------------------------------------------------------
# Lift coefficient
# ------------------------------------------------------------

axes[2, 0].plot(
    time_u,
    U_J1[1, :],
    label="J1: Squared thrust"
)

axes[2, 0].plot(
    time_u,
    U_J2[1, :],
    "--",
    label="J2: Linear thrust"
)

axes[2, 0].set_xlabel("Time [s]")
axes[2, 0].set_ylabel("Lift coefficient")
axes[2, 0].set_title("Lift coefficient comparison")
axes[2, 0].grid()
axes[2, 0].legend()


# ------------------------------------------------------------
# Vertical wind experienced
# ------------------------------------------------------------

axes[2, 1].plot(
    time,
    wind_J1,
    label="J1: Squared thrust"
)

axes[2, 1].plot(
    time,
    wind_J2,
    "--",
    label="J2: Linear thrust"
)

axes[2, 1].set_xlabel("Time [s]")
axes[2, 1].set_ylabel("Vertical wind [m/s]")
axes[2, 1].set_title("Vertical wind experienced")
axes[2, 1].grid()
axes[2, 1].legend()

fig.tight_layout()


# ============================================================
# 18. Plot wind field and both trajectories
# ============================================================

s_grid = np.linspace(0.0, 5500.0, 150)
h_grid = np.linspace(0.0, 400.0, 100)

S_grid, H_grid = np.meshgrid(
    s_grid,
    h_grid
)

W_grid = (
    -W_max
    * np.exp(
        -((S_grid - s_center) / sigma_s)**2
        -((H_grid - h_center) / sigma_h)**2
    )
)

fig_wind, ax_wind = plt.subplots(figsize=(11, 6))

contour = ax_wind.contourf(
    S_grid,
    H_grid,
    W_grid,
    levels=25,
    cmap="Blues_r"
)

ax_wind.plot(
    X_J1[0, :],
    X_J1[1, :],
    label="J1: Squared thrust",
    linewidth=2
)

ax_wind.plot(
    X_J2[0, :],
    X_J2[1, :],
    "--",
    label="J2: Linear thrust",
    linewidth=2
)

ax_wind.scatter(
    [s_center],
    [h_center],
    marker="x",
    color="black",
    s=80,
    label="Downdraft centre"
)

ax_wind.set_xlabel("Horizontal distance [m]")
ax_wind.set_ylabel("Altitude [m]")
ax_wind.set_title("Wind field and optimized trajectories")
ax_wind.legend()
ax_wind.grid()

fig_wind.colorbar(
    contour,
    ax=ax_wind,
    label="Vertical wind [m/s]"
)

fig_wind.tight_layout()

# ============================================================
# 19. Numerical comparison
# ============================================================

print("\n============================================")
print("              RESULTS")
print("============================================")

print("\nSolution minimizing J1:")

print("J1 =", J1_opt)
print("J2 =", J2_at_J1)

print("\nSolution minimizing J2:")

print("J1 =", J1_at_J2)
print("J2 =", J2_opt)

print("\nFinal state for J1:")
print(X_J1[:, -1])

print("\nFinal state for J2:")
print(X_J2[:, -1])

print("\nMaximum downward wind experienced:")

print("J1:", np.min(wind_J1), "m/s")
print("J2:", np.min(wind_J2), "m/s")

print("\nMaximum thrust:")

print("J1:", np.max(U_J1[0, :]), "N")
print("J2:", np.max(U_J2[0, :]), "N")

print("\nMinimum thrust:")

print("J1:", np.min(U_J1[0, :]), "N")
print("J2:", np.min(U_J2[0, :]), "N")

print("============================================")


# ============================================================
# 20. Aircraft animation
# ============================================================

# Choose which solution to animate:
#
# X_animation = X_J1
# X_animation = X_J2

X_animation = X_J1

playback_speed = 5.0
animation_interval = 30  # milliseconds

animation_frames = (
    int(
        np.ceil(
            tf * 1000.0
            / (playback_speed * animation_interval)
        )
    )
    + 1
)

animation_time = np.linspace(
    time[0],
    time[-1],
    animation_frames
)

animation_x = np.interp(
    animation_time,
    time,
    X_animation[0, :]
)

animation_y = np.interp(
    animation_time,
    time,
    X_animation[1, :]
)

animation_pitch = (
    np.interp(
        animation_time,
        time,
        X_animation[3, :]
    )
    + np.deg2rad(5.0)
)

animation = animate_airplane(
    animation_x,
    animation_y,
    animation_pitch,
    target=xf[0],
    interval=animation_interval,
    follow=True,
    title="Optimized landing trajectory (5x speed)"
)


# ============================================================
# 21. Show all plots
# ============================================================

plt.show()
