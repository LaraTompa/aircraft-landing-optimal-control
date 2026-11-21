from pathlib import Path
import csv

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import quad

# ============================================================
# 1. Configuration
# ============================================================

T = 1.0
N = 200                 # Must be even: t = 0.5 must be a mesh node
L = 0.05                # State constraint y(t) <= L
L_VALUES = (0.05, 0.1, 0.2, 0.5, 0.75, 1.0, 1.5, 2.0)
C = 2.5
N_C = 41                # Number of midpoint values in the coarse scan
MIDPOINT_SCAN_RADIUS = 0.05  # Local continuation radius around the seed solution
initial_a_values = [-21.2823, 13.8801, -15.0, -30.0, -40.0]
REFINEMENT_LEVELS = 2    # Local refinement rounds around the best sampled c
REFINEMENT_POINTS = 9    # New midpoint values per refinement round
IPOPT_TOL = 1e-10

# Interior points at which the state constraint is enforced
PATH_SAMPLES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
PLOT_DIR = Path(__file__).resolve().parents[2] / "data" / "spline"
ARC_QUAD_NODES, ARC_QUAD_WEIGHTS = np.polynomial.legendre.leggauss(5)

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


def trajectory_symmetry_error(solution):
    """Return max_t |y(t) - y(T-t)| for the reconstructed ZOH trajectory.

    On each mesh interval both terms are quadratic, so check the interval
    endpoints and any interior stationary point of their difference.
    """
    t_nodes = np.asarray(solution["t"], dtype=float)
    y_nodes, v_nodes = np.asarray(solution["X"], dtype=float)
    u = np.asarray(solution["U"], dtype=float)

    if not np.allclose(t_nodes, T - t_nodes[::-1], atol=1e-12, rtol=0.0):
        raise ValueError("The mesh must be symmetric around T/2.")

    max_error = 0.0
    for k in range(len(t_nodes) - 1):
        j = len(u) - 1 - k
        h = t_nodes[k + 1] - t_nodes[k]

        # With reflected intervals, the mirrored local time is h - tau.
        # The difference is d(tau) = A*tau**2 + B*tau + C.
        A = 0.5 * (u[k] - u[j])
        B = v_nodes[k] + v_nodes[j] + u[j] * h
        C0 = y_nodes[k] - y_nodes[j] - v_nodes[j] * h - 0.5 * u[j] * h**2

        candidates = [0.0, h]
        if abs(A) > 1e-14:
            tau_stationary = -B / (2.0 * A)
            if 0.0 < tau_stationary < h:
                candidates.append(tau_stationary)

        for tau in candidates:
            difference = A * tau**2 + B * tau + C0
            max_error = max(max_error, abs(float(difference)))

    return max_error


def independent_feasibility_check(solution, L):
    """Recheck endpoint, dynamics, arc length, and path feasibility."""
    t = np.asarray(solution["t"], dtype=float)
    X = np.asarray(solution["X"], dtype=float)
    u = np.asarray(solution["U"], dtype=float)
    y, v = X

    endpoint_residuals = np.array([
        y[0], v[0] - 1.0, y[-1], v[-1] + 1.0
    ])
    h = np.diff(t)
    y_next = y[:-1] + h * v[:-1] + 0.5 * h**2 * u
    v_next = v[:-1] + h * u
    dynamics_residual = max(
        float(np.max(np.abs(y[1:] - y_next))),
        float(np.max(np.abs(v[1:] - v_next)))
    )

    # Integrate each linear-velocity segment adaptively, independently of
    # the trapezoidal arc-length equality used by the NLP.
    fine_arc_length = sum(
        quad(
            lambda tau, vk=v[k], uk=u[k]: np.sqrt(1.0 + (vk + uk * tau)**2),
            0.0,
            h[k],
            epsabs=1e-12,
            epsrel=1e-12
        )[0]
        for k in range(len(u))
    )

    # A fine dense sample is reported, and the exact segment maxima provide
    # the conservative continuous-time obstacle check.
    _, dense_y = reconstruct_trajectory(solution, points_per_interval=1000)
    exact_max_y = -np.inf
    for k, hk in enumerate(h):
        candidates = [0.0, float(hk)]
        if abs(u[k]) > 1e-14:
            tau_star = -v[k] / u[k]
            if 0.0 < tau_star < hk:
                candidates.append(float(tau_star))
        exact_max_y = max(
            exact_max_y,
            *(y[k] + v[k] * tau + 0.5 * u[k] * tau**2 for tau in candidates)
        )

    return {
        "endpoint_residual_inf": float(np.max(np.abs(endpoint_residuals))),
        "endpoint_residuals": endpoint_residuals,
        "dynamics_residual_inf": dynamics_residual,
        "fine_arc_length": float(fine_arc_length),
        "arc_length_residual": float(fine_arc_length - C),
        "dense_max_y": float(np.max(dense_y)),
        "exact_max_y": float(exact_max_y),
        "exact_obstacle_violation": max(0.0, float(exact_max_y - L))
    }

# ============================================================
# 4. Direct multiple-shooting solver
# ============================================================

def solve_ocp(
    L,
    midpoint_value=None,
    previous_solution=None,
    initial_a=None
):
    """
    Solve

        min 1/2 integral_0^1 u(t)^2 dt

    subject to

        y' = v
        v' = u
        y(t) <= L
        integral_0^1 sqrt(1+v^2) dt = C

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


    # --------------------------------------------------------
    # Fixed arc-length constraint (5-point Gauss-Legendre per segment).
    # This integrates the actual linear velocity profile induced by each
    # piecewise-constant control, unlike trapezoidal integration at nodes.
    # --------------------------------------------------------

    arc_length = 0
    for k in range(N):
        for node, weight in zip(ARC_QUAD_NODES, ARC_QUAD_WEIGHTS):
            tau = 0.5 * h * (float(node) + 1.0)
            v_tau = X[1, k] + tau * U[k]
            arc_length += 0.5 * h * float(weight) * ca.sqrt(1 + v_tau**2)

    length_constraint = arc_length - C


    # --------------------------------------------------------
    # Collect equality constraints
    # --------------------------------------------------------

    equalities = [
        boundary,
        dynamics,
        length_constraint
    ]

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
    kkt_function = ca.Function(
        f"kkt_eval_{id(X)}",
        [z],
        [ca.gradient(objective, z), ca.jacobian(g, z)]
    )

    # --------------------------------------------------------
    # Initial guess / warm start
    # --------------------------------------------------------

    if previous_solution is None:

    # --------------------------------------------------------
    # Initial guess satisfying the endpoint velocities and targeting
    # the prescribed arc length for the smooth seed trajectory.
    # --------------------------------------------------------

        def initial_guess_v(tau, a):
            return (
                1.0 - 2.0 * tau
                + 2.0 * a * tau * (1.0 - tau) * (1.0 - 2.0 * tau)
            )

        def arc_length_for_a(a):
            return quad(
                lambda tau: np.sqrt(1.0 + initial_guess_v(tau, a)**2),
                0.0, 1.0
            )[0]

        if initial_a is None:
            # Automatically find an arc-length-matched seed. There can be
            # multiple roots; the first bracketed one is the negative branch.
            residual = lambda a: arc_length_for_a(a) - C
            bracket = None
            scan = np.linspace(-100.0, 100.0, 401)
            previous_a = float(scan[0])
            previous_residual = residual(previous_a)
            for candidate in scan[1:]:
                candidate = float(candidate)
                candidate_residual = residual(candidate)
                if previous_residual == 0.0:
                    bracket = (previous_a, previous_a)
                    break
                if previous_residual * candidate_residual < 0.0:
                    bracket = (previous_a, candidate)
                    break
                previous_a, previous_residual = candidate, candidate_residual
            if bracket is None:
                raise ValueError(
                    f"Could not bracket an initial-guess arc-length root for C={C}."
                )
            a = (
                bracket[0] if bracket[0] == bracket[1]
                else brentq(residual, *bracket)
            )
        else:
            a = float(initial_a)

        # Make the velocity dynamics exactly consistent at mesh nodes and
        # roll y forward with the same zero-order-hold dynamics as the NLP.
        v_guess = initial_guess_v(t, a)
        u_guess = np.diff(v_guess) / h
        y_guess = np.empty_like(t)
        y_guess[0] = 0.0
        for k in range(N):
            y_guess[k + 1] = (
                y_guess[k] + h * v_guess[k] + 0.5 * h**2 * u_guess[k]
            )

    else:

        # --------------------------------------------------------
        # Warm start from previous solution
        # --------------------------------------------------------

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
    g_opt = np.asarray(result["g"]).ravel()
    lam_g = np.asarray(result["lam_g"]).ravel()
    lam_x = np.asarray(result["lam_x"]).ravel()
    grad_f_opt, jac_g_opt = kkt_function(z_opt)
    stationarity = (
        np.asarray(grad_f_opt).ravel()
        + np.asarray(jac_g_opt).T @ lam_g
        + lam_x
    )
    eq_residual = float(np.max(np.abs(g_opt[:n_equalities])))
    ineq_g = g_opt[n_equalities:]
    ineq_lb = lbg[n_equalities:]
    ineq_ub = ubg[n_equalities:]
    ineq_lam = lam_g[n_equalities:]
    primal_ineq = max(
        float(np.max(np.maximum(ineq_lb - ineq_g, 0.0))),
        float(np.max(np.maximum(ineq_g - ineq_ub, 0.0)))
    )
    dual_sign = float(np.max(np.maximum(-ineq_lam, 0.0)))
    complementarity = float(np.max(np.abs(ineq_lam * (ineq_ub - ineq_g))))
    kkt_stationarity = float(np.max(np.abs(stationarity)))
    kkt_inf = max(
        eq_residual,
        primal_ineq,
        dual_sign,
        complementarity,
        kkt_stationarity
    )

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
        "initial_a": initial_a,
        "iterations": stats.get("iter_count", -1),
        "status": stats.get("return_status", "unknown"),
        "kkt_stationarity_inf": kkt_stationarity,
        "kkt_primal_eq_inf": eq_residual,
        "kkt_primal_ineq_inf": primal_ineq,
        "kkt_dual_sign_inf": dual_sign,
        "kkt_complementarity_inf": complementarity,
        "kkt_inf": kkt_inf
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
    solution["symmetry_error"] = trajectory_symmetry_error(solution)
    solution["independent_check"] = independent_feasibility_check(solution, L)
    checks = solution["independent_check"]
    solution["independently_feasible"] = (
        checks["endpoint_residual_inf"] < 1e-7
        and checks["dynamics_residual_inf"] < 1e-7
        and abs(checks["arc_length_residual"]) < 1e-7
        and checks["exact_obstacle_violation"] < 1e-7
    )
    solution["kkt_pass"] = solution["kkt_inf"] < 1e-6

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
# 5. Midpoint continuation and final unconstrained solve
# ============================================================

def run_midpoint_continuation(
    L,
    n_values=N_C,
    midpoint_radius=MIDPOINT_SCAN_RADIUS,
    initial_a=None,
    original_solution=None,
    refinement_levels=2,
    refinement_points=9
):
    """
    Explore midpoint-constrained solutions of the fixed-length
    spline problem and use the best feasible candidate as a
    warm start for the final problem without a midpoint equality.

    Every solve retains the fixed arc-length constraint:
        integral sqrt(1 + y'(t)^2) dt = C.

    The midpoint equality y(0.5) = c is temporary.

    Since y(t) <= L provides no lower bound on y(0.5), continue locally
    around the best available no-midpoint solution's midpoint instead of
    assuming c is in [0, L]. This local search does not establish globality.
    """

    if n_values < 2:
        raise ValueError("n_values must be at least 2.")
    if L <= 0:
        raise ValueError("L must be strictly positive.")
    if midpoint_radius <= 0:
        raise ValueError("midpoint_radius must be strictly positive.")

    # --------------------------------------------------------
    # 1. Solve the actual problem independently
    # --------------------------------------------------------

    original = (
        original_solution
        if original_solution is not None
        else solve_ocp(L, initial_a=initial_a)
    )
    if not original.get("feasible", False):
        raise ValueError("The no-midpoint starting solution must be feasible.")

    history = []
    solved_values = {}

    def solve_and_store(c, warm_start=None):

        key = round(float(c), 12)

        if key in solved_values:
            return solved_values[key]

        try:
            solution = solve_ocp(
                L=L,
                midpoint_value=float(c),
                previous_solution=warm_start
            )
            entry = {
                "c": float(c),
                "success": True,
                "objective": solution["objective"],
                "solution": solution,
                "feasible": solution["feasible"],
                "max_path_violation": solution["max_path_violation"],
                "error": None
            }
        except RuntimeError as exc:
            entry = {
                "c": float(c),
                "success": False,
                "objective": np.inf,
                "solution": None,
                "feasible": False,
                "max_path_violation": np.inf,
                "error": str(exc)
            }

        history.append(entry)
        solved_values[key] = entry

        return entry

    print("\nOriginal problem (no midpoint equality)")
    print("----------------------------------------")
    print(f"L                         = {L:.6f}")
    print(f"Initial-guess a           = {original.get('initial_a')}")
    print(f"Numerical objective       = {original['objective']:.10f}")
    print(f"Numerical y(0.5)          = {original['achieved_midpoint']:.10f}")
    print(f"Maximum reconstructed y  = {original['max_reconstructed_y']:.10f}")
    print(f"Maximum path violation   = {original['max_path_violation']:.3e}")
    print(f"Symmetry error           = {original['symmetry_error']:.3e}")
    checks = original["independent_check"]
    print(f"Fine quadrature length   = {checks['fine_arc_length']:.12f}")
    print(f"Arc-length residual      = {checks['arc_length_residual']:+.3e}")
    print(f"Endpoint residual (inf)  = {checks['endpoint_residual_inf']:.3e}")
    print(f"Dynamics residual (inf)  = {checks['dynamics_residual_inf']:.3e}")
    print(f"Exact obstacle violation= {checks['exact_obstacle_violation']:.3e}")
    print(f"KKT residual (inf)       = {original['kkt_inf']:.3e}")

    # --------------------------------------------------------
    # 2. Coarse midpoint continuation
    # --------------------------------------------------------

    c0 = original["achieved_midpoint"]
    coarse_lower = c0 - midpoint_radius
    coarse_upper = min(L, c0 + midpoint_radius)
    coarse_values = np.linspace(coarse_lower, coarse_upper, n_values)

    previous = original

    print(
        "\nLocal midpoint continuation "
        f"[{coarse_lower:.6f}, {coarse_upper:.6f}] around c0={c0:.6f}"
    )
    print("-----------------------------")

    for c in coarse_values:

        entry = solve_and_store(
            float(c),
            warm_start=previous
        )

        if entry["solution"] is None:
            print(f"c = {entry['c']: .7f} | FAILED: {entry['error']}")
        else:
            previous = entry["solution"]
            print(
                f"c = {entry['c']: .7f} | "
                f"J = {entry['objective']: .10f} | "
                f"achieved c = "
                f"{entry['solution']['achieved_midpoint']: .8f} | "
                f"max violation = "
                f"{entry['max_path_violation']:.2e}"
            )

    # --------------------------------------------------------
    # 3. Refine around the best feasible sampled midpoint
    # --------------------------------------------------------

    for level in range(refinement_levels):

        valid = [
            item for item in history
            if item["feasible"]
        ]

        if not valid:
            break

        best = min(
            valid,
            key=lambda item: item["objective"]
        )

        successful_values = sorted(
            item["c"] for item in history
            if item["solution"] is not None
        )

        idx = successful_values.index(best["c"])

        left = successful_values[max(0, idx - 1)]
        right = successful_values[min(len(successful_values) - 1, idx + 1)]

        if np.isclose(left, right):
            break

        new_values = np.linspace(
            left,
            right,
            refinement_points
        )

        print(
            f"\nRefinement level {level + 1}: "
            f"c in [{left:.8f}, {right:.8f}]"
        )
        print("-" * 55)

        for c in new_values:

            if round(float(c), 12) in solved_values:
                continue

            nearest = min(
                [item for item in history if item["solution"] is not None],
                key=lambda item: abs(item["c"] - float(c))
            )

            entry = solve_and_store(
                float(c),
                warm_start=nearest["solution"]
            )

            if entry["solution"] is None:
                print(f"c = {entry['c']: .9f} | FAILED: {entry['error']}")
            else:
                print(
                    f"c = {entry['c']: .9f} | "
                    f"J = {entry['objective']: .10f} | "
                    f"max violation = "
                    f"{entry['max_path_violation']:.2e}"
                )

    # --------------------------------------------------------
    # 4. Select a feasible warm start
    # --------------------------------------------------------

    valid = [
        item for item in history
        if item["feasible"]
    ]

    best = (
        min(valid, key=lambda item: item["objective"])
        if valid else None
    )

    if best is not None:
        warm_start = best["solution"]
        print("\nBest midpoint-constrained warm start")
        print("------------------------------------")
        print(f"Prescribed midpoint = {best['c']:.10f}")
        print(f"Objective            = {best['objective']:.10f}")
    else:
        warm_start = original
        print("\nNo feasible midpoint-constrained solution found.")
        print("Using the independent original solution as warm start.")

    # --------------------------------------------------------
    # 5. IMPORTANT: Release the midpoint equality
    # --------------------------------------------------------

    print("\nFinal solve WITHOUT midpoint equality")
    print("--------------------------------------")

    final_solution = solve_ocp(
        L=L,
        midpoint_value=None,
        previous_solution=warm_start
    )

    print(f"L                         = {L:.6f}")
    print(f"Final objective           = {final_solution['objective']:.10f}")
    print(f"Final y(0.5)              = {final_solution['achieved_midpoint']:.10f}")
    print(f"Maximum reconstructed y   = {final_solution['max_reconstructed_y']:.10f}")
    print(f"Maximum path violation    = {final_solution['max_path_violation']:.3e}")
    print(f"Symmetry error            = {final_solution['symmetry_error']:.3e}")
    checks = final_solution["independent_check"]
    print(f"Fine quadrature length    = {checks['fine_arc_length']:.12f}")
    print(f"Arc-length residual       = {checks['arc_length_residual']:+.3e}")
    print(f"Endpoint residual (inf)   = {checks['endpoint_residual_inf']:.3e}")
    print(f"Dynamics residual (inf)   = {checks['dynamics_residual_inf']:.3e}")
    print(f"Exact obstacle violation = {checks['exact_obstacle_violation']:.3e}")
    print(f"KKT residual (inf)        = {final_solution['kkt_inf']:.3e}")
    print(f"Independently feasible    = {final_solution['independently_feasible']}")
    print(f"Feasible                  = {final_solution['feasible']}")

    return original, history, best, final_solution



# ============================================================
# 6. Plots
# ============================================================

def plot_results(L, original, history, best):

    t_fine = np.linspace(0.0, 1.0, 1001)

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
    # Plot 2: Full trajectories, with and without midpoint equality
    # --------------------------------------------------------

    ax = axes[1]

    td, yd = reconstruct_trajectory(original)
    ax.plot(
        td,
        yd,
        color="black",
        linewidth=1.8,
        label="Original numerical solution (no midpoint equality)"
    )

    if best is not None:
        td, yd = reconstruct_trajectory(best["solution"])
        ax.plot(
            td,
            yd,
            color="tab:blue",
            linestyle="--",
            linewidth=2.2,
            label=f"Best midpoint candidate, c = {best['c']:.5f}"
        )

    ax.axhline(
        L,
        color="red",
        linestyle=":",
        label=f"Obstacle L = {L:g}"
    )

    ax.set_xlabel("Time t")
    ax.set_ylabel("y(t)")
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Full trajectory comparison over t = 0 to 1")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    a = original.get("initial_a")
    a_label = "auto" if a is None else f"a{a:g}"
    plot_path = PLOT_DIR / (
        f"complex_spline_midpoint_trajectories_L{L:g}_{a_label}.png"
    )
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Saved full trajectory comparison to: {plot_path}")
    plt.show()


# ============================================================
# 7. Main
# ============================================================

def main():

    print("STATE-CONSTRAINED SPLINE: FIXED LENGTH + MIDPOINT CONTINUATION")
    print("=" * 70)

    summaries = []
    for L_case in L_VALUES:
        print(f"\n\n{'=' * 24} L = {L_case:g}, N = {N} {'=' * 24}")

        initial_solutions = []
        for a in initial_a_values:
            try:
                solution = solve_ocp(L=L_case, initial_a=a)
            except RuntimeError as exc:
                print(f"Initial guess a={a:g} failed: {exc}")
                continue

            initial_solutions.append(solution)
            checks = solution["independent_check"]
            print(
                f"Initial a={a: .6f} | J={solution['objective']:.9f} | "
                f"sym={solution['symmetry_error']:.2e} | "
                f"arc err={checks['arc_length_residual']:+.2e} | "
                f"max y={checks['exact_max_y']:.8f} | "
                f"KKT={solution['kkt_inf']:.2e} | "
                f"feasible={solution['feasible'] and solution['independently_feasible']}"
            )

        feasible_starts = [
            s for s in initial_solutions
            if s["feasible"] and s["independently_feasible"]
        ]
        if not feasible_starts:
            print(f"No independently feasible initial solve found for L={L_case:g}; skipping.")
            continue

        best_start = min(feasible_starts, key=lambda s: s["objective"])
        print(
            f"\nUsing best direct solve a={best_start['initial_a']:g} "
            "as the midpoint-continuation start."
        )
        original, history, best, final_solution = run_midpoint_continuation(
            L_case,
            midpoint_radius=MIDPOINT_SCAN_RADIUS,
            initial_a=best_start["initial_a"],
            original_solution=best_start
        )
        plot_results(L_case, original, history, best)

        checks = final_solution["independent_check"]
        summaries.append({
            "L": L_case,
            "N": N,
            "initial_a": best_start["initial_a"],
            "objective": final_solution["objective"],
            "midpoint": final_solution["achieved_midpoint"],
            "symmetry_error": final_solution["symmetry_error"],
            "fine_arc_length": checks["fine_arc_length"],
            "arc_length_residual": checks["arc_length_residual"],
            "endpoint_residual_inf": checks["endpoint_residual_inf"],
            "endpoint_y0": checks["endpoint_residuals"][0],
            "endpoint_v0": checks["endpoint_residuals"][1],
            "endpoint_yT": checks["endpoint_residuals"][2],
            "endpoint_vT": checks["endpoint_residuals"][3],
            "dynamics_residual_inf": checks["dynamics_residual_inf"],
            "dense_max_y": checks["dense_max_y"],
            "exact_max_y": checks["exact_max_y"],
            "obstacle_violation": checks["exact_obstacle_violation"],
            "kkt_stationarity_inf": final_solution["kkt_stationarity_inf"],
            "kkt_primal_eq_inf": final_solution["kkt_primal_eq_inf"],
            "kkt_primal_ineq_inf": final_solution["kkt_primal_ineq_inf"],
            "kkt_dual_sign_inf": final_solution["kkt_dual_sign_inf"],
            "kkt_complementarity_inf": final_solution["kkt_complementarity_inf"],
            "kkt_inf": final_solution["kkt_inf"],
            "kkt_pass_1e-6": final_solution["kkt_pass"],
            "independently_feasible": final_solution["independently_feasible"]
        })

    if summaries:
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = PLOT_DIR / "complex_spline_verification_N200.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        print(f"\nSaved verification summary to: {summary_path}")


if __name__ == "__main__":
    main()
