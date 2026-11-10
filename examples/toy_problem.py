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
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def solve_basics(T=1.0, N=100):
    """Solve the toy problem; return t (N+1,), X (1, N+1), U (1, N)."""
    dt = T / N
    x = ca.MX.sym("x", N + 1)
    u = ca.MX.sym("u", N)
    J = ca.sumsqr(u) * dt
    g = ca.vertcat(
        x[0],
        *[x[k + 1] - x[k] - dt * u[k] for k in range(N)],
        x[N] - 1,
    )
    solver = ca.nlpsol("solver", "ipopt", {"x": ca.vertcat(x, u), "f": J, "g": g})
    sol = solver(x0=[0] * (2 * N + 1), lbg=0, ubg=0)
    stats = solver.stats()
    if not stats["success"]:
        raise RuntimeError(f"IPOPT failed: {stats.get('return_status')}")

    z = sol["x"].full().flatten()
    t = np.linspace(0, T, N + 1)
    return t, z[: N + 1].reshape(1, -1), z[N + 1 :].reshape(1, -1)


def plot_solution(t, X, U, state_names=None, control_names=None):
    """Plot every state and control over time (X: nx x N+1, U: nu x N)."""
    X, U = np.atleast_2d(X), np.atleast_2d(U)
    state_names = state_names or [f"x{i}" for i in range(X.shape[0])]
    control_names = control_names or [f"u{i}" for i in range(U.shape[0])]

    fig, (ax_x, ax_u) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    for row, name in zip(X, state_names):
        ax_x.plot(t, row, label=name)
    for row, name in zip(U, control_names):
        ax_u.step(t[:-1], row, where="post", label=name)

    ax_x.set_ylabel("State")
    ax_u.set_ylabel("Control")
    ax_u.set_xlabel("Time [s]")
    for ax in (ax_x, ax_u):
        ax.grid()
        ax.legend()
    fig.tight_layout()
    return fig


def animate_trajectory(X, plot_dims=(0, 1), y_fn=None, target=None, interval=30):
    """
    Animate a trajectory of any state dimension (X: nx x N+1).

    nx >= 2: plot_dims selects the (horizontal, vertical) states.
    nx == 1: vertical coordinate is y_fn(x), or 0 if y_fn is None.
    target: optional (x, y) landing point to mark.
    """
    X = np.atleast_2d(X)
    if X.shape[0] == 1:
        px = X[0]
        py = y_fn(px) if y_fn else np.zeros_like(px)
    else:
        px, py = X[plot_dims[0]], X[plot_dims[1]]

    def limits(a, include=()):
        lo = min([a.min(), *include])
        hi = max([a.max(), *include])
        pad = 0.1 * (hi - lo) or 0.1
        return lo - pad, hi + pad

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(*limits(px, [target[0]] if target else []))
    ax.set_ylim(*limits(py, [0] + ([target[1]] if target else [])))
    ax.set_xlabel("Position")
    ax.set_ylabel("Altitude")
    ax.set_title("Optimal Landing Trajectory")

    ax.axhline(0, linewidth=3)
    if target:
        ax.plot(*target, "x", markersize=10, markeredgewidth=2)

    plane, = ax.plot([], [], "o", markersize=12)
    trajectory, = ax.plot([], [], linewidth=2)

    def init():
        plane.set_data([], [])
        trajectory.set_data([], [])
        return plane, trajectory

    def update(frame):
        plane.set_data([px[frame]], [py[frame]])
        trajectory.set_data(px[: frame + 1], py[: frame + 1])
        return plane, trajectory

    return FuncAnimation(
        fig, update, frames=px.size, init_func=init, interval=interval, blit=True
    )


if __name__ == "__main__":
    t, X, U = solve_basics()
    print(ca.DM(np.concatenate((X[0], U[0]))))
    plot_solution(t, X, U, state_names=["x"], control_names=["u"])
    ani = animate_trajectory(X, y_fn=lambda x: 0.5 * (1 - x), target=(1, 0))
    plt.show()