"""Sensitivity of the spline problem solvers to the initial guess.

Every method is started from several initial guesses (y, y', y'') (feasible,
infeasible, inconsistent, random) and the converged trajectories, objectives,
iteration counts and failures are compared. The problem is a strictly convex
QP, so every successful run should reach the same solution.

Run from the repository root:
    python -m examples.spline_problem_scripts.spline_initializations
Writes figures and spline_initializations.txt to data/spline/.
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
        constrained_analytical_solution,
        default_guess,
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
        default_guess,
        reconstruct_trajectory,
        solve_spline,
    )

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "spline"
N = 100
L_VALUES = (0.2, 0.1)
DPI = 200


def _sine_series(coefficients):
    """Guess y = sum_k a_k sin(k pi t) with its exact derivatives."""
    k = np.arange(1, len(coefficients) + 1) * np.pi

    def guess(t):
        t = np.asarray(t, dtype=float)[..., None]
        a = np.asarray(coefficients)
        return (
            (a * np.sin(k * t)).sum(-1),
            (a * k * np.cos(k * t)).sum(-1),
            -(a * k**2 * np.sin(k * t)).sum(-1),
        )

    return guess


def zero_guess(t):
    t = np.asarray(t, dtype=float)
    return np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)


def constant_control_guess(t):
    """u = 5 integrated from x(0) = (0, 1); satisfies the initial conditions only."""
    t = np.asarray(t, dtype=float)
    return t + 2.5 * t**2, 1.0 + 5.0 * t, np.full_like(t, 5.0)


def large_guess(t):
    t = np.asarray(t, dtype=float)
    return 10.0 * t * (1.0 - t), 10.0 * (1.0 - 2.0 * t), np.full_like(t, -20.0)


def _random_guess(seed):
    rng = np.random.default_rng(seed)
    return _sine_series(rng.normal(size=6) / np.arange(1, 7))


# (label, guess(t) -> (y, y', y''))
INITIALIZATIONS = {
    "analytical (default)": default_guess,
    "zeros": zero_guess,
    "constant control u=5": constant_control_guess,
    "above bound: 2 sin(pi t)": _sine_series([2.0]),
    "oscillating: 0.5 sin(3 pi t)": _sine_series([0.0, 0.0, 0.5]),
    "large: 10 t(1-t)": large_guess,
    "random smooth (seed 0)": _random_guess(0),
    "random smooth (seed 1)": _random_guess(1),
    "random smooth (seed 2)": _random_guess(2),
}
DEFAULT_NAME = "analytical (default)"


def run_all(L):
    """Return {(init, method): result dict} with failures stored as status."""
    results = {}
    for init_name, guess in INITIALIZATIONS.items():
        for method in METHODS:
            try:
                solution = solve_spline(method=method, N=N, L=L, guess=guess)
                results[init_name, method] = {"ok": True, "solution": solution}
            except RuntimeError as exc:
                results[init_name, method] = {"ok": False, "error": str(exc).splitlines()[0]}
    return results


def deviations(results):
    """Max node-wise |y - y_default| per (init, method); NaN for failed runs."""
    out = {}
    for (init_name, method), res in results.items():
        if not results[DEFAULT_NAME, method]["ok"]:
            raise RuntimeError(
                f"Default initialization failed for {method}: "
                f"{results[DEFAULT_NAME, method]['error']}"
            )
        reference = results[DEFAULT_NAME, method]["solution"]["X"][0]
        out[init_name, method] = (
            np.max(np.abs(res["solution"]["X"][0] - reference)) if res["ok"] else np.nan
        )
    return out


def plot_results(L, results, dev):
    names = list(INITIALIZATIONS)
    colors = plt.cm.tab10(np.arange(len(names)))
    t_fine = np.linspace(0.0, 1.0, 1001)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    for i, (name, color) in enumerate(zip(names, colors)):
        ax.plot(t_fine, INITIALIZATIONS[name](t_fine)[0], color=color,
                label=f"#{i + 1} {name}")
    ax.axhline(L, color="red", ls=":", label=f"L = {L:g}")
    ax.set_ylim(-1.5, 3)
    ax.set_title("(a) Initial guesses y(t)")

    ax = axes[0, 1]
    ax.plot(t_fine, constrained_analytical_solution(t_fine, L), color="k", lw=6,
            alpha=0.25, label="Analytical")
    for name, color in zip(names, colors):
        res = results[name, "rk4"]
        if res["ok"]:
            t_d, y_d = reconstruct_trajectory(res["solution"])
            ax.plot(t_d, y_d, color=color, lw=1)
    ax.axhline(L, color="red", ls=":")
    ax.set_title("(b) Converged y(t), Shooting + RK4, all guesses (grey: analytical)")

    width = 0.8 / len(METHODS)
    for ax, title, ylabel in (
        (axes[1, 0], "(c) Solver iterations (failed runs: x)", "iterations"),
        (axes[1, 1], r"(d) Deviation $\max_k |y_k - y_k^{default}|$ (failed runs: x)",
         "deviation"),
    ):
        for i, method in enumerate(METHODS):
            x = np.arange(len(names)) + (i - (len(METHODS) - 1) / 2) * width
            if ylabel == "iterations":
                values = [
                    results[n, method]["solution"]["iterations"] if results[n, method]["ok"] else np.nan
                    for n in names
                ]
            else:
                values = [max(dev[n, method], 1e-17) if results[n, method]["ok"] else np.nan
                          for n in names]
            ax.bar(x, values, width, color=METHOD_STYLES[method][0],
                   label=METHOD_LABELS[method])
            failed = [xi for xi, n in zip(x, names) if not results[n, method]["ok"]]
            ax.plot(failed, np.full(len(failed), 1.0 if ylabel == "iterations" else 1e-16),
                    "kx", ms=8)
        if ylabel == "deviation":
            ax.set_yscale("log")
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels([f"#{i + 1}" for i in range(len(names))])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    axes[1, 0].set_ylim(0, 36)
    axes[1, 0].legend(fontsize=7, ncol=2, loc="upper center")
    axes[1, 1].set_xlabel("initial guess # (see panel a, same order)")
    axes[1, 0].set_xlabel("initial guess # (see panel a, same order)")
    for ax in (axes[0, 0], axes[0, 1]):
        ax.set_xlabel("t")
        ax.set_ylabel("y(t)")
        ax.grid(True)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Spline problem: influence of the initialization (N={N}, L={L:g})")
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"spline_initializations_L{str(L).replace('.', 'p')}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def report(L, results, dev):
    lines = [f"\nL = {L:g}, N = {N}"]
    header = f"{'initial guess':30s}" + "".join(f"{m:>22s}" for m in METHODS)
    lines += ["iterations / objective / max|y - y_default|", header]
    for name in INITIALIZATIONS:
        cells = []
        for method in METHODS:
            res = results[name, method]
            if res["ok"]:
                s = res["solution"]
                cells.append(f"{s['iterations']:3d}/{s['objective']:.6f}/{dev[name, method]:.0e}")
            else:
                cells.append("FAILED")
        lines.append(f"{name:30s}" + "".join(f"{c:>22s}" for c in cells))
    failures = [(k, v["error"]) for k, v in results.items() if not v["ok"]]
    for (name, method), error in failures:
        lines.append(f"  failure: {name} / {method}: {error}")
    objectives = {
        m: [results[n, m]["solution"]["objective"] for n in INITIALIZATIONS if results[n, m]["ok"]]
        for m in METHODS
    }
    lines.append("objective spread (max - min) over successful initializations:")
    for m, values in objectives.items():
        lines.append(f"  {METHOD_LABELS[m]}: {max(values) - min(values):.2e} ({len(values)}/{len(INITIALIZATIONS)} converged)")
    return "\n".join(lines)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    texts = []
    for L in L_VALUES:
        results = run_all(L)
        dev = deviations(results)
        plot_results(L, results, dev)
        texts.append(report(L, results, dev))
        print(texts[-1])
    (OUTPUT_DIR / "spline_initializations.txt").write_text("\n".join(texts) + "\n")
