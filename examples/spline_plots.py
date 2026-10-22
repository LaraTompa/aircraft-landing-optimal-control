"""Generate report figures for the spline problem into the (git-ignored) data/ folder.

Run from the repository root:
    python examples\\spline_plots.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .spline_problem import (
        analytical_solution,
        constrained_analytical_solution,
        solve_spline,
    )
else:
    from spline_problem import (
        analytical_solution,
        constrained_analytical_solution,
        solve_spline,
    )

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
N = 100
L_VALUES = (0.25, 0.2, 1.0 / 6.0, 0.1, 0.05)
L_DETAIL = 0.1
N_VALUES = (25, 50, 100, 200)
DPI = 200


def label(method):
    return {"euler": "Euler", "rk4": "RK4"}[method]


def save(fig, name):
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def plot_trajectories_for_bounds():
    """RK4 solution y(t) for several L, with the analytical solution."""
    t_fine = np.linspace(0.0, 1.0, 1001)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, L in enumerate(L_VALUES):
        color = f"C{i}"
        sol = solve_spline("rk4", N=N, L=L)
        ax.plot(sol["t"], sol["X"][0], color=color, label=f"RK4, $L={L:.3g}$")
        ax.plot(t_fine, constrained_analytical_solution(t_fine, L), ":", color="k", lw=1)
        ax.axhline(L, color=color, ls="--", lw=0.6)
    ax.plot([], [], ":", color="k", label="analytical")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$y(t)$")
    ax.set_title(f"Constrained spline for different bounds $L$ ($N={N}$)")
    ax.grid(True)
    ax.legend()
    save(fig, "spline_trajectories_L.png")


def plot_methods_detail():
    """Euler vs RK4 vs analytical for one L, plus node-wise error."""
    L = L_DETAIL
    t_fine = np.linspace(0.0, 1.0, 1001)
    sols = {m: solve_spline(m, N=N, L=L) for m in ("euler", "rk4")}

    fig, (ax, ax_err) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax.plot(t_fine, analytical_solution(t_fine), "k--", lw=1, label="unconstrained $t(1-t)$")
    ax.plot(t_fine, constrained_analytical_solution(t_fine, L), "g-", lw=3, alpha=0.4,
            label="analytical constrained")
    for m, sol in sols.items():
        ax.plot(sol["t"], sol["X"][0], label=f"{label(m)} ($J={sol['objective']:.4f}$)")
    ax.axhline(L, color="red", ls=":", label=f"$L={L:g}$")
    ax.set_ylabel("$y(t)$")
    ax.set_title(f"Euler vs. RK4, $L={L:g}$, $N={N}$")
    ax.grid(True)
    ax.legend()

    for m, sol in sols.items():
        err = np.abs(sol["X"][0] - constrained_analytical_solution(sol["t"], L))
        ax_err.semilogy(sol["t"], np.maximum(err, 1e-12), label=label(m))
    ax_err.set_xlabel("$t$")
    ax_err.set_ylabel("$|y_k - y_{ana}(t_k)|$")
    ax_err.grid(True, which="both")
    ax_err.legend()
    save(fig, "spline_methods_L0p1.png")


def plot_control():
    """Optimal control u = y'' for L = L_DETAIL (RK4)."""
    L = L_DETAIL
    sol = solve_spline("rk4", N=N, L=L)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.step(sol["t"][:-1], sol["U"], where="post")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$u(t) = y''(t)$")
    ax.set_title(f"Optimal control, RK4, $L={L:g}$ (zero on the flat arc)")
    ax.grid(True)
    save(fig, "spline_control_L0p1.png")


def plot_convergence():
    """Max node error vs N for Euler and RK4 at L = L_DETAIL."""
    L = L_DETAIL
    errors = {"euler": [], "rk4": []}
    for n in N_VALUES:
        for m in errors:
            sol = solve_spline(m, N=n, L=L)
            errors[m].append(
                np.max(np.abs(sol["X"][0] - constrained_analytical_solution(sol["t"], L)))
            )
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for m, e in errors.items():
        ax.loglog(N_VALUES, e, "o-", label=label(m))
    ax.set_xlabel("$N$")
    ax.set_ylabel("max node error")
    ax.set_title(f"Convergence, $L={L:g}$")
    ax.grid(True, which="both")
    ax.legend()
    save(fig, "spline_convergence_L0p1.png")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_trajectories_for_bounds()
    plot_methods_detail()
    plot_control()
    plot_convergence()
