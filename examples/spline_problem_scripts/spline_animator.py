"""Animate the spline solutions against the analytical curve."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

if __package__:
    from ..solve_spline_problem import (
        METHOD_LABELS,
        METHOD_STYLES,
        METHODS,
        constrained_analytical_solution,
        reconstruct_trajectory,
        solve_spline,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from solve_spline_problem import (
        METHOD_LABELS,
        METHOD_STYLES,
        METHODS,
        constrained_analytical_solution,
        reconstruct_trajectory,
        solve_spline,
    )


def animate_solutions(solutions, interval=30):
    """Animate the curves y(t) as they are traced from t=0 to t=1."""
    first_solution = next(iter(solutions.values()))
    t = first_solution["t"]
    L = first_solution["L"]
    t_fine = np.linspace(t[0], t[-1], 10 * (len(t) - 1) + 1)
    curves = {
        method: reconstruct_trajectory(solution)
        for method, solution in solutions.items()
    }
    curves["analytical"] = (t_fine, constrained_analytical_solution(t_fine, L))
    styles = dict(METHOD_STYLES)
    styles["analytical"] = ("black", ":", "D")
    labels = dict(METHOD_LABELS)
    labels["analytical"] = "Analytical constrained"

    fig, ax = plt.subplots(figsize=(9, 5))
    all_y = np.concatenate([values[1] for values in curves.values()])
    y_min = min(float(np.min(all_y)), 0.0)
    y_max = max(float(np.max(all_y)), L)
    padding = 0.08 * (y_max - y_min) or 0.1
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.axhline(L, color="red", linestyle=":", label=f"Constraint L={L:g}")
    ax.set_xlabel("Time t")
    ax.set_ylabel("y(t)")
    ax.set_title("Spline trajectories")
    ax.grid(True)

    lines = {}
    markers = {}
    for method in curves:
        color, linestyle, marker = styles[method]
        (line,) = ax.plot([], [], color=color, linestyle=linestyle, label=labels[method])
        (marker_artist,) = ax.plot([], [], marker, color=color, markersize=9)
        lines[method] = line
        markers[method] = marker_artist
    ax.legend()

    def init():
        for line, marker in zip(lines.values(), markers.values()):
            line.set_data([], [])
            marker.set_data([], [])
        return tuple(lines.values()) + tuple(markers.values())

    def update(frame):
        current_t = t[frame]
        artists = []
        for method, (curve_t, curve_y) in curves.items():
            end = int(np.searchsorted(curve_t, current_t, side="right"))
            end = min(max(end, 1), len(curve_t))
            lines[method].set_data(curve_t[:end], curve_y[:end])
            current_y = np.interp(current_t, curve_t, curve_y)
            markers[method].set_data([current_t], [current_y])
            artists.extend((lines[method], markers[method]))
        return tuple(artists)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(t),
        init_func=init,
        interval=interval,
        blit=True,
    )
    return fig, animation


if __name__ == "__main__":
    solutions = {method: solve_spline(method=method) for method in METHODS}
    for method, solution in solutions.items():
        print(f"{METHOD_LABELS[method]} objective: {solution['objective']:.10f}")
    print("Analytical unconstrained objective: 2.0000000000")
    figure, animation = animate_solutions(solutions)
    plt.show()
