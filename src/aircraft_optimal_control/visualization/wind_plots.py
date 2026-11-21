
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_wind_field(
    wind_model,
    s_bounds,
    h_bounds,
    trajectory=None,
    t=0.0,
    resolution=100,
    output_path=None,
    show=True,
):
    """
    Plot the wind field and optionally overlay an aircraft trajectory.

    The vertical wind component is shown as a contour field.
    """

    s_values = np.linspace(*s_bounds, resolution)
    h_values = np.linspace(*h_bounds, resolution)

    S, H = np.meshgrid(s_values, h_values)

    W_s = np.zeros_like(S)
    W_h = np.zeros_like(H)

    for i in range(resolution):
        for j in range(resolution):
            velocity = np.asarray(
                wind_model.velocity(
                    float(S[i, j]),
                    float(H[i, j]),
                    float(t),
                )
            ).astype(float).reshape(-1)

            W_s[i, j] = velocity[0]
            W_h[i, j] = velocity[1]

    fig, ax = plt.subplots(figsize=(12, 6))

    contour = ax.contourf(
        S,
        H,
        W_h,
        levels=30,
        cmap="coolwarm",
    )

    colorbar = fig.colorbar(contour, ax=ax)
    colorbar.set_label("Vertical wind velocity [m/s]")

    # Wind direction arrows
    step = max(1, resolution // 15)

    ax.quiver(
        S[::step, ::step],
        H[::step, ::step],
        W_s[::step, ::step],
        W_h[::step, ::step],
        color="black",
        alpha=0.65,
    )

    # Aircraft trajectory
    if trajectory is not None:
        X = np.asarray(trajectory["X"])

        ax.plot(
            X[0, :],
            X[1, :],
            color="yellow",
            linewidth=2.5,
            label="Aircraft trajectory",
        )

        ax.scatter(
            X[0, 0],
            X[1, 0],
            color="lime",
            edgecolor="black",
            s=70,
            label="Initial position",
            zorder=5,
        )

        ax.scatter(
            X[0, -1],
            X[1, -1],
            color="red",
            edgecolor="black",
            s=70,
            label="Landing position",
            zorder=5,
        )

        ax.legend()

    ax.set_xlabel("Horizontal position [m]")
    ax.set_ylabel("Altitude [m]")
    ax.set_title("Vertical wind field")
    ax.set_aspect("auto")

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return fig
