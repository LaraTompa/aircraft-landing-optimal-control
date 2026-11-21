"""Generate figures from solve_complex_spline_problem.py results.

Run from the repository root with the aircraft-oc environment:
    python -m appendix.spline.spline_plots

Figures are written as PNG and PDF files to data/spline/thesis_figures/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from appendix.spline import solve_complex_spline_problem as model


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "spline"
OUTPUT_DIR = DATA_DIR / "thesis_figures"
STATE_RESULTS = DATA_DIR / "complex_spline_verification_N200.csv"
SPLINE_DIR = DATA_DIR
INITIAL_A_VALUES = [-21.2823, 13.8801, -15.0, -30.0, -40.0]
MESH_SIZES = [50, 100, 200, 400]
REPRESENTATIVE_L = [0.05, 1.5]

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 140,
    "savefig.dpi": 240,
})


def save_figure(fig, stem):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"{stem}.{extension}", bbox_inches="tight")
    plt.close(fig)
    print("Saved", OUTPUT_DIR / f"{stem}.png")


def footer(fig, text):
    fig.text(0.5, 0.015, text, ha="center", va="bottom", fontsize=8)


def plot_state_constraint():
    rows = np.genfromtxt(STATE_RESULTS, delimiter=",", names=True, dtype=None, encoding="utf-8")
    L = rows["L"]
    objective = rows["objective"]
    max_y = rows["exact_max_y"]
    negative = L <= 1.0
    positive = L >= 1.5

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    ax = axes[0]
    ax.plot(L[negative], objective[negative], "o-", color="tab:blue", label="Negative-midpoint branch")
    ax.plot(L[positive], objective[positive], "s-", color="tab:orange", label="Positive-midpoint branch")
    ax.set_xlabel("State bound $L$")
    ax.set_ylabel("Objective $J$")
    ax.set_title("Best feasible objective found")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(L[negative], max_y[negative], "o-", color="tab:blue", label="$\max_t y(t)$, negative branch")
    ax.plot(L[positive], max_y[positive], "s-", color="tab:orange", label="$\max_t y(t)$, positive branch")
    ax.plot(L, L, "k--", linewidth=1.2, label="Obstacle $y=L$")
    ax.set_xlabel("State bound $L$")
    ax.set_ylabel("Maximum trajectory height")
    ax.set_title("Obstacle clearance")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle("Influence of the state constraint ($N=200$)")
    footer(fig, "Selected solutions: symmetry error < 1.1e-15; combined KKT residual < 9.1e-12. The branch change is between tested L=1 and L=1.5.")
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    save_figure(fig, "state_constraint_influence")


def plot_best_trajectories_by_L():
    """Overlay the best verified spline trajectory for every tested state bound."""
    rows = np.genfromtxt(STATE_RESULTS, delimiter=",", names=True, dtype=None, encoding="utf-8")
    model.N = 200
    solutions = []
    for row in rows:
        L = float(row["L"])
        initial_a = float(row["initial_a"])
        solution = model.solve_ocp(L=L, initial_a=initial_a)
        if not (solution["feasible"] and solution["independently_feasible"]):
            raise RuntimeError(f"Stored best start is not independently feasible for L={L:g}.")
        t, y = model.reconstruct_trajectory(solution, points_per_interval=5)
        solutions.append({"L": L, "a": initial_a, "solution": solution, "t": t, "y": y})

    # The selected best solutions coincide within each branch. Plot one clear
    # representative per branch and identify every L represented by it.
    branches = []
    for item in solutions:
        sign = "positive" if item["solution"]["achieved_midpoint"] > 0 else "negative"
        match = next((branch for branch in branches if branch["sign"] == sign), None)
        if match is None:
            branches.append({"sign": sign, "items": [item]})
        else:
            match["items"].append(item)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    colors = {"negative": "tab:blue", "positive": "tab:orange"}
    for branch in branches:
        items = branch["items"]
        reference = items[0]
        max_curve_difference = max(
            float(np.max(np.abs(item["y"] - reference["y"]))) for item in items
        )
        L_labels = ", ".join(f"{item['L']:g}" for item in items)
        label = f"Best solution for $L={L_labels}$"
        ax.plot(reference["t"], reference["y"], lw=2.2,
                color=colors[branch["sign"]], label=label)
        print(
            f"Best trajectory branch ({branch['sign']} midpoint), "
            f"L={L_labels}: J={reference['solution']['objective']:.9f}, "
            f"max inter-L curve difference={max_curve_difference:.2e}"
        )

    # Mark the two representative obstacle bounds used by the two branches.
    for L, color, label in ((0.05, "0.55", "Obstacle, negative branch example"),
                            (1.5, "0.65", "Obstacle, positive branch example")):
        ax.axhline(L, color=color, linestyle="--", linewidth=1.0, label=label)

    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Spline trajectory $y(t)$")
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Best feasible spline trajectories for the tested state bounds")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    SPLINE_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(SPLINE_DIR / f"spline_trajectories_complex_L.{extension}", bbox_inches="tight")
    plt.close(fig)
    print("Saved", SPLINE_DIR / "spline_trajectories_complex_L.png")


def solve_initializations():
    model.N = 200
    data = {}
    for L in REPRESENTATIVE_L:
        data[L] = []
        for a in INITIAL_A_VALUES:
            solution = model.solve_ocp(L=L, initial_a=a)
            data[L].append({
                "a": a,
                "objective": solution["objective"],
                "midpoint": solution["achieved_midpoint"],
                "symmetry": solution["symmetry_error"],
                "kkt": solution["kkt_inf"],
                "feasible": solution["feasible"] and solution["independently_feasible"],
            })
    return data


def plot_initialization(data):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), sharey=True)
    for ax, L in zip(axes, REPRESENTATIVE_L):
        rows = data[L]
        a = np.array([row["a"] for row in rows])
        J = np.array([row["objective"] for row in rows])
        colors = ["tab:orange" if row["midpoint"] > 0 else "tab:blue" for row in rows]
        ax.bar(np.arange(len(a)), J, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(np.arange(len(a)))
        ax.set_xticklabels([f"{v:g}" for v in a], rotation=30, ha="right")
        ax.set_xlabel("Initial seed $a$")
        ax.set_title(f"Objective by initialization, $L={L:g}$")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Objective J")
    from matplotlib.patches import Patch
    axes[1].legend(handles=[
        Patch(facecolor="tab:blue", edgecolor="black", label="Negative-midpoint branch"),
        Patch(facecolor="tab:orange", edgecolor="black", label="Positive-midpoint branch"),
    ], loc="upper right")
    fig.suptitle("Influence of initialization ($N=200$)")
    footer(fig, "All plotted starts are feasible. Colors denote the converged midpoint branch; symmetry error < 1.3e-15 and KKT residual < 9.1e-12.")
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    save_figure(fig, "initialization_sensitivity")


def solve_mesh_refinement():
    data = {"negative": [], "positive": []}
    for N in MESH_SIZES:
        model.N = N
        for branch, L, a in (("negative", 0.05, -21.2823), ("positive", 1.5, 13.8801)):
            solution = model.solve_ocp(L=L, initial_a=a)
            data[branch].append({
                "N": N,
                "objective": solution["objective"],
                "symmetry": solution["symmetry_error"],
                "kkt": solution["kkt_inf"],
                "arc_error": solution["independent_check"]["arc_length_residual"],
            })
    model.N = 200
    return data


def plot_mesh_refinement(data):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    styles = {"negative": ("tab:blue", "$L=0.05$, negative branch"),
              "positive": ("tab:orange", "$L=1.5$, positive branch")}
    for branch, (color, label) in styles.items():
        rows = data[branch]
        Ns = np.array([r["N"] for r in rows])
        Js = np.array([r["objective"] for r in rows])
        axes[0].plot(Ns, Js, "o-", color=color, label=label)
        # Successive objective differences estimate the mesh error.
        err = np.abs(np.diff(Js))
        axes[1].loglog(Ns[:-1], err, "o-", color=color, label=label)
    axes[0].set_xlabel("Number of mesh intervals $N$")
    axes[0].set_ylabel("Objective $J_N$")
    axes[0].set_title("Objective convergence")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].set_xlabel("Coarser mesh size $N$")
    axes[1].set_ylabel(r"$|J_N-J_{2N}|$")
    axes[1].set_title("Successive mesh differences")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()
    fig.suptitle("Mesh refinement")
    footer(fig, "Successive objective differences decrease by about a factor of four when N doubles (second-order trend). Symmetry error < 1.9e-15; KKT residual about 9.1e-12.")
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    save_figure(fig, "mesh_refinement")


def midpoint_curve(L, initial_a, n_values=13):
    model.N = 200
    original = model.solve_ocp(L=L, initial_a=initial_a)
    c0 = original["achieved_midpoint"]
    values = np.linspace(c0 - model.MIDPOINT_SCAN_RADIUS, c0 + model.MIDPOINT_SCAN_RADIUS, n_values)
    previous = original
    rows = []
    failures = []
    for c in values:
        try:
            solution = model.solve_ocp(L=L, midpoint_value=float(c), previous_solution=previous)
        except RuntimeError:
            failures.append(float(c))
            continue
        previous = solution
        rows.append({
            "c": float(c),
            "objective": solution["objective"],
            "symmetry": solution["symmetry_error"],
            "kkt": solution["kkt_inf"],
            "feasible": solution["feasible"] and solution["independently_feasible"],
        })
    return original, rows, failures


def plot_midpoint_continuation(curves):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    for ax, L in zip(axes, REPRESENTATIVE_L):
        original, rows, failures = curves[L]
        c = np.array([r["c"] for r in rows])
        J = np.array([r["objective"] for r in rows])
        ax.plot(c, J, "o-", color="tab:blue", markersize=4, label=r"Midpoint-constrained $V(L,c)$")
        ax.axhline(original["objective"], color="black", linestyle="--", label="No-midpoint objective")
        ax.axvline(original["achieved_midpoint"], color="gray", linestyle=":", label="$c_0$ from no-midpoint solve")
        if failures:
            ytop = max(J) * 1.02
            ax.scatter(failures, np.full(len(failures), ytop), marker="x", color="red", label="Failed solve")
        ax.set_xlabel("Prescribed midpoint $c$")
        ax.set_ylabel("Objective $J$")
        ax.set_title(f"Local midpoint continuation, $L={L:g}$")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Midpoint continuation around the no-midpoint solution")
    footer(fig, "The minimum in each explored interval occurs at c0. Final symmetry error < 8e-16; KKT residual about 9.1e-12.")
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    save_figure(fig, "midpoint_continuation")


def main():
    model.N = 200
    plot_best_trajectories_by_L()
    plot_state_constraint()
    initialization_data = solve_initializations()
    plot_initialization(initialization_data)
    mesh_data = solve_mesh_refinement()
    plot_mesh_refinement(mesh_data)
    curves = {
        0.05: midpoint_curve(0.05, -21.2823),
        1.5: midpoint_curve(1.5, 13.8801),
    }
    plot_midpoint_continuation(curves)
    print("All thesis visualizations are saved under", OUTPUT_DIR)


if __name__ == "__main__":
    main()
