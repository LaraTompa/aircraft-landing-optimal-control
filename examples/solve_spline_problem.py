import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Configuration
# ============================================================

T = 1.0
N = 200                 # Must be even: t = 0.5 must be a mesh node
L = 0.05                # State constraint y(t) <= L
N_C = 41                # Number of midpoint values in the coarse scan
REFINEMENT_LEVELS = 2    # Local refinement rounds around the best sampled c
REFINEMENT_POINTS = 9    # New midpoint values per refinement round
IPOPT_TOL = 1e-10

# Interior points at which the state constraint is enforced
PATH_SAMPLES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


# ============================================================
# 2. Analytical solution of the unconstrained problem
# ============================================================

def analytical_solution(t):
    """
    Unconstrained analytical minimizer:
        y(t) = t(1-t).

    This is only a reference trajectory. It violates y(t) <= L
    whenever L < 1/4, so it is not used as a constrained optimum
    or as an objective benchmark.
    """
    t = np.asarray(t, dtype=float)
    return t * (1.0 - t)


# ============================================================
# 3. Trajectory reconstruction
# ============================================================

def reconstruct_trajectory(solution, points_per_interval=100):
    """
    Reconstruct the trajectory using the exact state polynomial
    for piecewise-constant acceleration.
    """
    t_nodes = solution["t"]
    y_nodes, v_nodes = solution["X"]
    u = solution["U"]

    dense_t = []
    dense_y = []

    for k in range(len(t_nodes) - 1):
        h = t_nodes[k + 1] - t_nodes[k]

        tau = np.linspace(
            0.0, h,
            points_per_interval,
            endpoint=False
        )

        y = (
            y_nodes[k]
            + v_nodes[k] * tau
            + 0.5 * u[k] * tau**2
        )

        dense_t.extend(t_nodes[k] + tau)
        dense_y.extend(y)

    dense_t.append(t_nodes[-1])
    dense_y.append(y_nodes[-1])

    return np.asarray(dense_t), np.asarray(dense_y)


# ============================================================
# 4. Direct multiple-shooting solver
# ============================================================

def solve_ocp(L, midpoint_value=None, previous_solution=None):
    """
    Solve

        min 1/2 integral_0^1 u(t)^2 dt

    subject to

        y' = v
        v' = u
        y(t) <= L

        y(0) = 0
        v(0) = 1
        y(1) = 0
        v(1) = -1

    Optionally add the parametric equality

        y(0.5) = midpoint_value.

    The control u is piecewise constant on each interval.
    """

    if N % 2 != 0:
        raise ValueError("N must be even.")

    if L <= 0:
        raise ValueError("L must be strictly positive.")

    if midpoint_value is not None:
        if midpoint_value > L + 1e-12:
            raise ValueError("The midpoint cannot exceed L.")

    h = T / N
    t = np.linspace(0.0, T, N + 1)

    # --------------------------------------------------------
    # Decision variables
    # --------------------------------------------------------

    X = ca.MX.sym("X", 2, N + 1)
    U = ca.MX.sym("U", N)

    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------

    objective = 0.5 * h * ca.sumsqr(U)

    # --------------------------------------------------------
    # Exact zero-order-hold dynamics
    # --------------------------------------------------------

    defects = []

    for k in range(N):
        y_next = (
            X[0, k]
            + h * X[1, k]
            + 0.5 * h**2 * U[k]
        )

        v_next = X[1, k] + h * U[k]

        defects.append(
            X[:, k + 1] - ca.vertcat(y_next, v_next)
        )

    dynamics = ca.vertcat(*defects)

    # --------------------------------------------------------
    # Boundary conditions
    # --------------------------------------------------------

    boundary = ca.vertcat(
        X[0, 0],
        X[1, 0] - 1.0,
        X[0, N],
        X[1, N] + 1.0
    )

    equalities = [boundary, dynamics]

    # --------------------------------------------------------
    # Optional midpoint equality
    # --------------------------------------------------------

    if midpoint_value is not None:
        equalities.append(
            X[0, N // 2] - float(midpoint_value)
        )

    equality_constraints = ca.vertcat(*equalities)

    # --------------------------------------------------------
    # State constraint at nodes and interior points
    # --------------------------------------------------------

    path_constraints = []

    for k in range(N):
        for s in PATH_SAMPLES:
            tau = s * h

            y_s = (
                X[0, k]
                + tau * X[1, k]
                + 0.5 * tau**2 * U[k]
            )

            path_constraints.append(y_s)

    path = ca.vertcat(
        X[0, :].T,
        ca.vertcat(*path_constraints)
    )

    g = ca.vertcat(equality_constraints, path)

    n_equalities = int(equality_constraints.numel())
    n_path = int(path.numel())

    lbg = np.concatenate((
        np.zeros(n_equalities),
        np.full(n_path, -np.inf)
    ))

    ubg = np.concatenate((
        np.zeros(n_equalities),
        np.full(n_path, L)
    ))

    # --------------------------------------------------------
    # NLP
    # --------------------------------------------------------

    z = ca.vertcat(ca.vec(X), U)

    nlp = {
        "x": z,
        "f": objective,
        "g": g
    }

    options = {
        "ipopt.print_level": 0,
        "ipopt.tol": IPOPT_TOL,
        "ipopt.acceptable_tol": 1e-9,
        "ipopt.max_iter": 3000,
        "ipopt.bound_relax_factor": 0.0,
        "print_time": False
    }

    solver = ca.nlpsol(
        "solver",
        "ipopt",
        nlp,
        options
    )

    # --------------------------------------------------------
    # Initial guess / warm start
    # --------------------------------------------------------

    if previous_solution is None:

        y_guess = t * (1.0 - t)
        v_guess = 1.0 - 2.0 * t
        u_guess = -2.0 * np.ones(N)

    else:

        y_guess = previous_solution["X"][0].copy()
        v_guess = previous_solution["X"][1].copy()
        u_guess = previous_solution["U"].copy()

    X_guess = np.vstack((y_guess, v_guess))

    z0 = np.concatenate((
        X_guess.reshape(-1, order="F"),
        u_guess
    ))

    # --------------------------------------------------------
    # Solve
    # --------------------------------------------------------

    result = solver(
        x0=z0,
        lbg=lbg,
        ubg=ubg
    )

    stats = solver.stats()

    if not stats["success"]:
        raise RuntimeError(
            f"IPOPT failed: {stats.get('return_status')}"
        )

    # --------------------------------------------------------
    # Extract solution
    # --------------------------------------------------------

    z_opt = np.asarray(result["x"]).ravel()

    n_x = 2 * (N + 1)

    X_opt = z_opt[:n_x].reshape(
        (2, N + 1),
        order="F"
    )

    U_opt = z_opt[n_x:]

    solution = {
        "t": t,
        "X": X_opt,
        "U": U_opt,
        "objective": float(result["f"]),
        "L": float(L),
        "midpoint_value": midpoint_value,
        "iterations": stats.get("iter_count", -1),
        "status": stats.get("return_status", "unknown")
    }

    # --------------------------------------------------------
    # Dense feasibility check
    # --------------------------------------------------------

    td, yd = reconstruct_trajectory(solution)

    achieved_midpoint = float(X_opt[0, N // 2])
    exact_segment_max = -np.inf

    # On each interval, y(t) is quadratic because u is constant.
    # Check its endpoints and, when it lies inside the interval,
    # its stationary point tau = -v_k/u_k.
    for k in range(N):
        h_k = t[k + 1] - t[k]
        y0 = X_opt[0, k]
        v0 = X_opt[1, k]
        u0 = U_opt[k]

        candidates = [0.0, h_k]
        if abs(u0) > 1e-14:
            tau_star = -v0 / u0
            if 0.0 < tau_star < h_k:
                candidates.append(tau_star)

        values = [
            y0 + v0 * tau + 0.5 * u0 * tau**2
            for tau in candidates
        ]
        exact_segment_max = max(exact_segment_max, max(values))

    solution["achieved_midpoint"] = achieved_midpoint
    solution["max_reconstructed_y"] = float(exact_segment_max)
    solution["max_path_violation"] = max(
        0.0,
        float(exact_segment_max - L)
    )

    solution["midpoint_error"] = (
        None if midpoint_value is None
        else abs(achieved_midpoint - midpoint_value)
    )

    solution["feasible"] = (
        solution["max_path_violation"] < 1e-5
        and (
            solution["midpoint_error"] is None
            or solution["midpoint_error"] < 1e-5
        )
    )

    return solution


# ============================================================
# 5. Midpoint continuation and numerical outer search
# ============================================================

def run_midpoint_continuation(
    L,
    n_values=N_C,
    refinement_levels=2,
    refinement_points=9
):
    """
    Numerically explore V(L,c), where V(L,c) is the minimum objective
    with the additional equality y(0.5) = c.

    The scan covers the full admissible midpoint interval [0, L].
    It starts at c=L and moves downwards, warm-starting each solve
    from the preceding solution.

    After the coarse scan, the interval around the best sampled
    midpoint is refined. No analytical constrained optimum or
    constrained objective is used.

    The original problem (without a midpoint equality) is also solved
    independently. Its numerical objective is used only as a numerical
    reference when checking whether a midpoint-constrained candidate
    improves the original NLP result.
    """

    if n_values < 2:
        raise ValueError("n_values must be at least 2.")
    if L <= 0:
        raise ValueError("L must be strictly positive.")

    # Solve the original problem without fixing its midpoint.
    original = solve_ocp(L)

    history = []
    solved_values = {}

    def solve_and_store(c, warm_start=None):
        # Avoid solving the same midpoint twice.
        key = round(float(c), 12)
        if key in solved_values:
            return solved_values[key]

        solution = solve_ocp(
            L=L,
            midpoint_value=float(c),
            previous_solution=warm_start
        )

        entry = {
            "c": float(c),
            "objective": solution["objective"],
            "solution": solution,
            "feasible": solution["feasible"],
            "max_path_violation": solution["max_path_violation"]
        }
        history.append(entry)
        solved_values[key] = entry
        return entry

    print("\nOriginal problem (no midpoint equality)")
    print("----------------------------------------")
    print(f"L                         = {L:.6f}")
    print(f"Numerical objective       = {original['objective']:.10f}")
    print(f"Numerical y(0.5)          = {original['achieved_midpoint']:.10f}")
    print(f"Maximum reconstructed y  = {original['max_reconstructed_y']:.10f}")
    print(f"Maximum path violation   = {original['max_path_violation']:.3e}")

    # Coarse continuation: start at the obstacle and move down to zero.
    coarse_values = np.linspace(L, 0.0, n_values)
    previous = original

    print("\nCoarse midpoint continuation")
    print("-----------------------------")
    for c in coarse_values:
        entry = solve_and_store(float(c), warm_start=previous)
        previous = entry["solution"]
        print(
            f"c = {entry['c']: .7f} | "
            f"J = {entry['objective']: .10f} | "
            f"achieved c = {entry['solution']['achieved_midpoint']: .8f} | "
            f"max violation = {entry['max_path_violation']:.2e}"
        )

    # Refine the sampled interval around the best feasible point.
    for level in range(refinement_levels):
        valid = [item for item in history if item["feasible"]]
        if not valid:
            break

        best = min(valid, key=lambda item: item["objective"])
        sorted_values = sorted(item["c"] for item in history)
        idx = sorted_values.index(best["c"])

        left = sorted_values[max(0, idx - 1)]
        right = sorted_values[min(len(sorted_values) - 1, idx + 1)]

        if np.isclose(left, right):
            break

        new_values = np.linspace(left, right, refinement_points)
        print(f"\nRefinement level {level + 1}: c in [{left:.8f}, {right:.8f}]")
        print("-" * 55)

        for c in new_values:
            if round(float(c), 12) in solved_values:
                continue

            # Warm-start from the nearest already computed midpoint.
            nearest = min(
                history,
                key=lambda item: abs(item["c"] - float(c))
            )
            entry = solve_and_store(
                float(c),
                warm_start=nearest["solution"]
            )
            print(
                f"c = {entry['c']: .9f} | "
                f"J = {entry['objective']: .10f} | "
                f"max violation = {entry['max_path_violation']:.2e}"
            )

    valid = [item for item in history if item["feasible"]]
    best = min(valid, key=lambda item: item["objective"]) if valid else None

    print("\nBest sampled midpoint-constrained candidate")
    print("-------------------------------------------")
    if best is None:
        print("No feasible midpoint-constrained solution was found.")
    else:
        print(f"Best prescribed midpoint = {best['c']:.10f}")
        print(f"Best numerical objective = {best['objective']:.10f}")
        print(f"Original numerical objective = {original['objective']:.10f}")
        print(
            "Difference (candidate - original) = "
            f"{best['objective'] - original['objective']:+.10e}"
        )
        if best["objective"] < original["objective"] - 1e-8:
            print(
                "The scan found a lower numerical objective than the "
                "independent original-problem solve. Recheck solver "
                "convergence and feasibility; this may indicate that "
                "the original solve did not reach the same numerical optimum."
            )
        else:
            print(
                "The scan did not find a lower objective than the original "
                "problem solve. This is consistent with the original problem "
                "being the minimization over all admissible midpoint values."
            )

    return original, history, best


# ============================================================
# 6. Plots
# ============================================================

def plot_results(L, original, history, best):

    t_fine = np.linspace(0.0, 1.0, 1001)
    unconstrained_y = analytical_solution(t_fine)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --------------------------------------------------------
    # Plot 1: Numerical midpoint value function
    # --------------------------------------------------------

    ax = axes[0]
    valid = [item for item in history if item["feasible"]]

    c_plot = np.array([item["c"] for item in valid])
    J_plot = np.array([item["objective"] for item in valid])
    order = np.argsort(c_plot)

    ax.plot(
        c_plot[order],
        J_plot[order],
        "o-",
        label=r"Numerical $V(L,c)$"
    )

    ax.axhline(
        original["objective"],
        color="black",
        linestyle="--",
        label="Original problem: numerical objective"
    )
    ax.axvline(
        original["achieved_midpoint"],
        color="gray",
        linestyle=":",
        label="Original numerical midpoint"
    )

    if best is not None:
        ax.scatter(
            [best["c"]],
            [best["objective"]],
            color="red",
            zorder=5,
            label="Lowest sampled value"
        )

    ax.set_xlabel("Prescribed midpoint c")
    ax.set_ylabel(r"Numerical objective $V(L,c)$")
    ax.set_title(f"Midpoint continuation, L = {L:g}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --------------------------------------------------------
    # Plot 2: Numerical trajectories and unconstrained reference
    # --------------------------------------------------------

    ax = axes[1]
    ax.plot(
        t_fine,
        unconstrained_y,
        "k--",
        linewidth=2,
        label=r"Unconstrained analytical reference: $t(1-t)$"
    )

    td, yd = reconstruct_trajectory(original)
    ax.plot(
        td,
        yd,
        color="black",
        linewidth=1.8,
        label="Original constrained numerical solution"
    )

    # Show representative points from the sampled value function.
    if valid:
        sorted_valid = sorted(valid, key=lambda item: item["c"])
        indices = np.linspace(
            0, len(sorted_valid) - 1,
            min(5, len(sorted_valid)),
            dtype=int
        )
        selected = [sorted_valid[i] for i in indices]

        for item in selected:
            td, yd = reconstruct_trajectory(item["solution"])
            ax.plot(
                td,
                yd,
                linewidth=1.3,
                label=f"Midpoint c = {item['c']:.4f}"
            )

    if best is not None:
        td, yd = reconstruct_trajectory(best["solution"])
        ax.plot(
            td,
            yd,
            linewidth=2.2,
            label=f"Best sampled candidate, c = {best['c']:.5f}"
        )

    ax.axhline(
        L,
        color="red",
        linestyle=":",
        label=f"Obstacle L = {L:g}"
    )

    ax.set_xlabel("Time t")
    ax.set_ylabel("y(t)")
    ax.set_title("Trajectories during midpoint continuation")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# 7. Main
# ============================================================

def main():

    print("STATE-CONSTRAINED SPLINE: MIDPOINT CONTINUATION")
    print("=" * 58)
    print("The only analytical trajectory used is the unconstrained")
    print("reference y(t) = t(1-t). No constrained analytical objective")
    print("or constrained analytical midpoint is used.")

    original, history, best = run_midpoint_continuation(
        L,
        n_values=N_C,
        refinement_levels=REFINEMENT_LEVELS,
        refinement_points=REFINEMENT_POINTS
    )

    plot_results(L, original, history, best)


if __name__ == "__main__":
    main()
