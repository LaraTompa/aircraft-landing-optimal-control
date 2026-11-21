
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_solution(
    solution,
    objective_name,
    wind_model=None,
    output_path=None,
    show=True,
):
    """
    Plot the state and control trajectories of an OCP solution.

    Expected solution format:
        X: (4, N+1)
        U: (2, N)
        time: (N+1,)
        control_time: (N,)
    """

    X = np.asarray(solution["X"])
    U = np.asarray(solution["U"])
    time = np.asarray(solution["time"])
    control_time = np.asarray(solution["control_time"])

    s = X[0, :]
    h = X[1, :]
    V = X[2, :]
    gamma = np.rad2deg(X[3, :])

    thrust = U[0, :]
    CL = U[1, :]

    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    fig.suptitle(
        f"Aircraft landing - {objective_name}",
        fontsize=15,
    )

    # Trajectory
    ax = axes[0, 0]
    ax.plot(s, h, linewidth=2)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Horizontal position [m]")
    ax.set_ylabel("Altitude [m]")
    ax.set_title("Landing trajectory")
    ax.grid(True)

    # Airspeed
    ax = axes[0, 1]
    ax.plot(time, V, linewidth=2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Airspeed [m/s]")
    ax.set_title("Airspeed")
    ax.grid(True)

    # Flight-path angle
    ax = axes[1, 0]
    ax.plot(time, gamma, linewidth=2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Flight-path angle [deg]")
    ax.set_title("Flight-path angle")
    ax.grid(True)

    # Thrust
    ax = axes[1, 1]
    ax.step(control_time, thrust, where="post", linewidth=2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Thrust [N]")
    ax.set_title("Thrust")
    ax.grid(True)

    # Lift coefficient
    ax = axes[2, 0]
    ax.step(control_time, CL, where="post", linewidth=2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Lift coefficient [-]")
    ax.set_title("Lift coefficient")
    ax.grid(True)

    # Experienced wind
    ax = axes[2, 1]

    if wind_model is not None:
        wind_values = np.array([
            np.asarray(
                wind_model.velocity(float(si), float(hi), float(ti))
            ).astype(float).reshape(-1)
            for si, hi, ti in zip(s, h, time)
        ])

        ax.plot(time, wind_values[:, 0], label="Horizontal wind")
        ax.plot(time, wind_values[:, 1], label="Vertical wind")

        ax.legend()
    else:
        ax.text(
            0.5, 0.5,
            "No wind model provided",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Wind velocity [m/s]")
    ax.set_title("Experienced wind")
    ax.grid(True)

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return fig
