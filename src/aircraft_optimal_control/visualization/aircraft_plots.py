
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aircraft_optimal_control.models.point_mass import thrust_at, lift_coefficient


def plot_solution(
    solution,
    objective_name,
    wind_model=None,
    aircraft_model=None,
    output_path=None,
    show=True,
):
    """
    Plot the state and control trajectories of an OCP solution.

    Expected solution format:
        X: (n_states, N+1)
        U: (n_controls, N)
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

    paper_model = aircraft_model is not None and "angle_of_attack" in aircraft_model.STATE_INDICES

    fig, axes = plt.subplots(4 if paper_model else 3, 2,
                             figsize=(14, 14 if paper_model else 11))
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

    if paper_model:
        alpha = X[aircraft_model.STATE_INDICES["angle_of_attack"], :]
        alpha_dot = U[aircraft_model.CONTROL_INDICES["angle_of_attack_rate"], :]
        ax = axes[1, 1]
        ax.plot(time, np.rad2deg(alpha), linewidth=2)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Angle of attack [deg]")
        ax.set_title("Angle of attack")
        ax.grid(True)

        thrust = np.asarray([
            float(thrust_at(Vi, ti, aircraft_model.aircraft))
            for Vi, ti in zip(V, time)
        ])
        CL = np.asarray([
            float(lift_coefficient(alpha_i, aircraft_model.aircraft))
            for alpha_i in alpha
        ])
        plots = [
            (axes[2, 0], control_time, alpha_dot, "Angle-of-attack rate [rad/s]", "Control"),
            (axes[2, 1], time, thrust, "Scheduled thrust [N]", "Thrust"),
            (axes[3, 0], time, CL, "Lift coefficient [-]", "Lift coefficient"),
        ]
    else:
        thrust = U[0, :]
        CL = U[1, :]
        plots = [
            (axes[1, 1], control_time, thrust, "Thrust [N]", "Thrust"),
            (axes[2, 0], control_time, CL, "Lift coefficient [-]", "Lift coefficient"),
        ]

    for ax, tx, values, ylabel, title in plots:
        ax.step(tx, values, where="post", linewidth=2)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)

    # Experienced wind
    ax = axes[3, 1] if paper_model else axes[2, 1]

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
