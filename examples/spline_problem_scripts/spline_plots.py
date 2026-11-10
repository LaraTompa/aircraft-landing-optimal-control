"""Generate spline figures in the (git-ignored) data/spline folder.

Run from the repository root:
    python -m examples.spline_problem_scripts.spline_plots
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from ..solve_spline_problem import (
        METHOD_LABELS,
        METHOD_STYLES,
        METHODS,
        analytical_solution,
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
        analytical_solution,
        constrained_analytical_solution,
        reconstruct_trajectory,
        solve_spline,
    )

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "spline"
N = 100
L_VALUES = (0.25, 0.2, 1.0 / 6.0, 0.1, 0.05)
L_DETAIL = 0.1
N_VALUES = (25, 50, 100, 200)
N_COARSE = (6, 10, 15, 20)
DPI = 200


def label(method):
    return METHOD_LABELS[method]


def style(method, n_points):
    color, linestyle, marker = METHOD_STYLES[method]
    return dict(color=color, linestyle=linestyle, marker=marker,
                markevery=max(n_points // 20, 1), markersize=5)


def save(fig, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def plot_trajectories_for_bounds():
    """Collocation solution y(t) for several L, with the analytical solution."""
    t_fine = np.linspace(0.0, 1.0, 1001)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, L in enumerate(L_VALUES):
        color = f"C{i}"
        sol = solve_spline("hermite_simpson", N=N, L=L)
        t_dense, y_dense = reconstruct_trajectory(sol)
        ax.plot(t_dense, y_dense, color=color, label=f"Collocation, $L={L:.3g}$")
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
    """All methods vs analytical for one L (continuous y(t)), plus node-wise error."""
    L = L_DETAIL
    t_fine = np.linspace(0.0, 1.0, 1001)
    sols = {m: solve_spline(m, N=N, L=L) for m in METHODS}

    fig, (ax, ax_err) = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True)
    ax.plot(t_fine, analytical_solution(t_fine), color="gray", ls="--", lw=1,
            label="unconstrained $t(1-t)$")
    ax.plot(t_fine, constrained_analytical_solution(t_fine, L), color="k", lw=4,
            alpha=0.25, label="analytical constrained")
    for m, sol in sols.items():
        t_dense, y_dense = reconstruct_trajectory(sol)
        ax.plot(t_dense, y_dense, label=f"{label(m)} ($J={sol['objective']:.4f}$)",
                **style(m, len(t_dense)))
    ax.axhline(L, color="red", ls=":", label=f"$L={L:g}$")
    ax.set_ylabel("$y(t)$")
    ax.set_title(f"Shooting, collocation and QP/SQP, $L={L:g}$, $N={N}$")
    ax.grid(True)
    ax.legend(fontsize=8)

    for m, sol in sols.items():
        err = np.abs(sol["X"][0] - constrained_analytical_solution(sol["t"], L))
        ax_err.semilogy(sol["t"], np.maximum(err, 1e-12), label=label(m),
                        **style(m, len(sol["t"])))
    ax_err.set_xlabel("$t$")
    ax_err.set_ylabel("$|y_k - y_{ana}(t_k)|$")
    ax_err.grid(True, which="both")
    ax_err.legend(fontsize=8, loc="upper center")
    save(fig, "spline_methods_L0p1.png")


def plot_control():
    """Optimal control u = y'' for L = L_DETAIL (all methods)."""
    L = L_DETAIL
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for m in METHODS:
        sol = solve_spline(m, N=N, L=L)
        color, linestyle, _ = METHOD_STYLES[m]
        if len(sol["U"]) == len(sol["t"]):
            ax.plot(sol["t"], sol["U"], color=color, linestyle=linestyle, label=label(m))
        else:
            ax.step(sol["t"][:-1], sol["U"], where="post", color=color,
                    linestyle=linestyle, label=label(m))
    ax.set_xlabel("$t$")
    ax.set_ylabel("$u(t) = y''(t)$")
    ax.set_title(f"Optimal control, $L={L:g}$ (zero on the flat arc)")
    ax.grid(True)
    ax.legend(fontsize=8)
    save(fig, "spline_control_L0p1.png")


def plot_analytical_vs_numerical():
    """Analytical solution and the RK4 shooting solution for several N (L = L_DETAIL)."""
    L = L_DETAIL
    method = "rk4"
    t_fine = np.linspace(0.0, 1.0, 1001)
    y_ana = constrained_analytical_solution(t_fine, L)
    fig, (ax, ax_zoom, ax_flat) = plt.subplots(
        3, 1, figsize=(8, 11), gridspec_kw={"height_ratios": [3, 2, 2]}
    )
    ax.plot(t_fine, y_ana, color="k", lw=4, alpha=0.3, label="analytical")
    ax_zoom.plot(t_fine, y_ana, color="k", lw=4, alpha=0.3)
    ax_flat.axhline(0.0, color="k", lw=4, alpha=0.3, label="analytical")
    for i, n in enumerate(N_VALUES):
        sol = solve_spline(method, N=n, L=L)
        t_dense, y_dense = reconstruct_trajectory(sol, points_per_interval=40)
        kwargs = dict(color=f"C{i}", lw=1.5, label=f"{label(method)}, $N={n}$")
        ax.plot(t_dense, y_dense, **kwargs)
        ax_zoom.plot(t_dense, y_dense, **kwargs)
        ax_flat.plot(t_dense, y_dense - L, **kwargs)
    ax.axhline(L, color="red", ls=":", label=f"$L={L:g}$")
    ax_zoom.axhline(L, color="red", ls=":")
    ax_zoom.set_xlim(0.15, 0.45)
    ax_zoom.set_ylim(0.093, 0.1003)
    ax_zoom.set_title("Zoom near the start of the flat arc ($t=3L=0.3$)", fontsize=9)
    ax_zoom.set_xlabel("$t$")
    ax_zoom.set_ylabel("$y(t)$")
    ax_zoom.grid(True)
    ax_flat.set_xlim(0.3, 0.7)
    ax_flat.set_ylim(-4e-4, 6e-5)
    ax_flat.set_title("Flat arc $[3L,\\,1-3L]$ magnified: $y(t)-L$ "
                      "(sag shrinks like $N^{-2}$)", fontsize=9)
    ax_flat.set_xlabel("$t$")
    ax_flat.set_ylabel("$y(t)-L$")
    ax_flat.grid(True)
    ax_flat.legend(fontsize=7, loc="lower center", ncol=3)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$y(t)$")
    ax.set_title(f"Analytical vs. numerical solution, $L={L:g}$")
    ax.grid(True)
    ax.legend(fontsize=8, loc="lower center")
    fig.tight_layout()
    save(fig, "spline_analytical_vs_numerical_L0p1.png")


def plot_coarse_grids():
    """All methods on coarse grids: full y(t) (top) and y - L on the flat arc (bottom)."""
    L = L_DETAIL
    n_values = N_COARSE
    t_fine = np.linspace(0.0, 1.0, 1001)
    fig, axes = plt.subplots(2, len(n_values), figsize=(4.2 * len(n_values), 8))
    for col, n in enumerate(n_values):
        ax, ax_flat = axes[0, col], axes[1, col]
        ax.plot(t_fine, constrained_analytical_solution(t_fine, L), color="k", lw=4,
                alpha=0.25, label="analytical")
        ax_flat.axhline(0.0, color="k", lw=4, alpha=0.25)
        failed = []
        flat_values = [np.zeros(1)]
        for m in METHODS:
            if m == "euler" and L < 1.0 / n:
                failed.append(label(m))
                continue
            sol = solve_spline(m, N=n, L=L)
            t_dense, y_dense = reconstruct_trajectory(sol, points_per_interval=40)
            in_arc = (t_dense >= 3 * L) & (t_dense <= 1 - 3 * L)
            flat_values.append(y_dense[in_arc] - L)
            color, linestyle, _ = METHOD_STYLES[m]
            ax.plot(t_dense, y_dense, color=color, linestyle=linestyle, lw=1.5, label=label(m))
            ax_flat.plot(t_dense, y_dense - L, color=color, linestyle=linestyle, lw=1.5)
        ax.axhline(L, color="red", ls=":", label=f"$L={L:g}$")
        ax.set_title(f"$N={n}$" + (f" ({', '.join(failed)}: infeasible)" if failed else ""),
                     fontsize=9)
        ax.set_xlabel("$t$")
        ax.grid(True)
        ax_flat.set_xlim(3 * L, 1 - 3 * L)
        in_arc = [v for v in flat_values if v.size]
        if in_arc:
            lo = min(v.min() for v in in_arc)
            hi = max(v.max() for v in in_arc)
            pad = 0.1 * (hi - lo) or 1e-6
            ax_flat.set_ylim(lo - pad, hi + pad)
        ax_flat.set_xlabel("$t$")
        ax_flat.set_title("flat arc magnified: $y(t)-L$", fontsize=9)
        ax_flat.grid(True)
    axes[0, 0].set_ylabel("$y(t)$")
    axes[1, 0].set_ylabel("$y(t)-L$")
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle(f"Coarse grids, all methods, $L={L:g}$")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    save(fig, "spline_coarse_grids_L0p1.png")


def plot_convergence():
    """Max node error vs N for all methods, for L=0.1 and the generic L=0.113.

    For L=0.1 the kinks of the optimal control (t=0.3, 0.7) are grid nodes for
    N>=50, so Hermite-Simpson is exact there up to the solver tolerance. For
    L=0.113 (kinks at 0.339, 0.661) they are never nodes for these N.
    """
    cases = (
        (L_DETAIL, "kinks on nodes for $N\\geq50$"),
        (0.113, "kinks never on nodes"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (L, note) in zip(axes, cases):
        errors = {m: [] for m in METHODS}
        for n in N_VALUES:
            for m in errors:
                sol = solve_spline(m, N=n, L=L)
                errors[m].append(
                    np.max(np.abs(sol["X"][0] - constrained_analytical_solution(sol["t"], L)))
                )
        for m, e in errors.items():
            color, linestyle, marker = METHOD_STYLES[m]
            ax.loglog(N_VALUES, e, color=color, linestyle=linestyle, marker=marker,
                      label=label(m))
        n_ref = np.array(N_VALUES, dtype=float)
        ref = errors["rk4"][-1]
        ax.loglog(n_ref, ref * (n_ref[-1] / n_ref) ** 1, color="gray", lw=0.8, ls=(0, (1, 3)),
                  label="$O(N^{-1})$")
        ax.loglog(n_ref, ref * (n_ref[-1] / n_ref) ** 2, color="k", lw=0.8, ls=(0, (1, 3)),
                  label="$O(N^{-2})$")
        ax.set_xlabel("$N$")
        ax.set_title(f"$L={L:g}$ ({note})")
        ax.grid(True, which="both")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].set_ylabel("max node error")
    axes[0].text(
        0.03, 0.04,
        "Collocation $\\lesssim10^{-7}$ is IPOPT's floor\n(exact QP solve: $\\approx10^{-12}$)",
        transform=axes[0].transAxes, fontsize=8, va="bottom",
    )
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "spline_convergence.png")


if __name__ == "__main__":
    plot_trajectories_for_bounds()
    plot_methods_detail()
    plot_control()
    plot_analytical_vs_numerical()
    plot_coarse_grids()
    plot_convergence()
