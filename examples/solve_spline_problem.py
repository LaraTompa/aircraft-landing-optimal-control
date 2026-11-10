"""
State-constrained spline problem
================================

Continuous-time problem:

    Minimize
        J = 1/2 * integral_0^1 (y''(t))^2 dt

    subject to
        y(t) <= L,                    t in [0, 1]
        y(0) = 0,  y(1) = 0
        y'(0) = 1, y'(1) = -1

Optimal-control formulation:

    x1(t) = y(t),  x2(t) = y'(t),  u(t) = y''(t)

    Minimize
        J = 1/2 * integral_0^1 u(t)^2 dt

    subject to
        x1'(t) = x2(t)
        x2'(t) = u(t)
        x1(t) <= L

    x1(0) = 0, x2(0) = 1
    x1(1) = 0, x2(1) = -1

Dynamical-systems analysis:

    The state equation is x' = A*x + B*u, with
        A = [[0, 1], [0, 0]],  B = [[0], [1]].
    For constant u, equilibria satisfy x2 = 0 and u = 0, so every
    (x1, x2) = (c, 0) with u = 0 is an equilibrium. If u is fixed
    and nonzero, there is no equilibrium.

    The characteristic polynomial of A is lambda^2, so both
    eigenvalues are zero. A is a nilpotent Jordan block, so the origin
    (and every equilibrium (c, 0)) is not asymptotically stable; it is in
    fact Lyapunov unstable, because a nonzero x2 makes x1 grow linearly.

    Since A^2 = 0,
        exp(A*h) = I + A*h = [[1, h], [0, 1]].
    For u constant on [t, t+h], the variation-of-constants solution is
        x(t+h) = exp(A*h)*x(t)
                  + integral_0^h exp(A*(h-s))*B*u ds.
    Here exp(A*(h-s))*B = [h-s, 1]^T, so
        integral_0^h exp(A*(h-s))*B*u ds = [h^2/2, h]^T*u.
    Thus the exact zero-order-hold update is
        x[k+1] = [[1, h], [0, 1]]*x[k] + [[h^2/2], [h]]*u[k].

    Forward Euler instead gives A_E = [[1, h], [0, 1]] and
    B_E = [[0], [h]], omitting the input contribution h^2*u[k]/2
    to x1[k+1]. Classical RK4 gives the exact zero-order-hold
    update here because the state dynamics are at most quadratic in
    time when u is constant. The program prints all three updates
    using the actual Euler and RK4 step functions used by the solver.

Direct multiple-shooting discretization:

    Divide [0, 1] into N intervals of length h = 1/N. The decision
    variables are the states X[k] = (x1[k], x2[k]) at all N+1 nodes
    and one control U[k] on each interval. U[k] is held constant on
    [t[k], t[k+1]). Enforce each discrete dynamics equation as a
    shooting equality, impose the boundary values at the first and
    last nodes, and impose x1[k] <= L at every shooting node.

    Forward Euler:
        x1[k+1] = x1[k] + h*x2[k]
        x2[k+1] = x2[k] + h*U[k]
    Caveat: forward Euler solves a perturbed problem. Its model is not
    the exact response to the piecewise constant control, so the optimal U
    drives the Euler model to x_f but the true system to a point that
    misses x_f by O(h) (about h for these data). Also x1[1] = h*x2[0] = h
    for every U, so the Euler problem is infeasible when L < h.

    Classical RK4:
        Write each stage as K_i = (K_{i,1}, K_{i,2}). With
        f(x, U) = (x2, U), the stages are
            K1 = (x2[k],                     U[k])
            K2 = (x2[k] + h*K_{1,2}/2,       U[k])
            K3 = (x2[k] + h*K_{2,2}/2,       U[k])
            K4 = (x2[k] + h*K_{3,2},         U[k])
        and
            X[k+1] = X[k] + h/6*(K1 + 2*K2 + 2*K3 + K4).
        Since each stage's second component is U[k], this simplifies to
            x1[k+1] = x1[k] + h*x2[k] + (h^2/2)*U[k]
            x2[k+1] = x2[k] + h*U[k]
        RK4 is exact for these dynamics when U is constant per interval.

    Runge-Kutta order 5 (Dormand-Prince weights, fixed step, 6 stages):
        The same shooting equalities X[k+1] = Phi_RK5(X[k], U[k]) with the
        Dormand-Prince tableau a_ij, b_i. For constant U every stage has
        K_{i,1} = x2[k] + c_i*h*U[k], K_{i,2} = U[k] (row sums of a equal c_i),
        so the update is again exactly
            x1[k+1] = x1[k] + h*x2[k] + (h^2/2)*U[k]  (since sum b_i c_i = 1/2)
            x2[k+1] = x2[k] + h*U[k].
        Any Runge-Kutta method of order >= 2 is exact here, so RK5 yields the
        same discrete problem and solution as RK4 (differences ~1e-16) and only
        costs more stages.

    For either method the piecewise-constant-control cost is
        J_N = 1/2 * h * sum(k=0..N-1) U[k]^2,
    which is the exact integral of u^2 for this control parameterization.
    The path constraint is enforced at nodes only, as a standard
    finite-dimensional approximation of the all-times constraint.

Direct collocation (Hermite-Simpson):

    Unlike shooting, collocation treats the state at every node and at
    every interval midpoint as a decision variable and enforces the
    dynamics through algebraic "defect" constraints. The control is
    continuous and piecewise linear with node values U[0..N]; this suits
    the problem, since the optimal u = y'' is piecewise linear (y''''=0).
    With f_k = (x2[k], U[k]), f_{k+1} = (x2[k+1], U[k+1]) and
    U_c = (U[k] + U[k+1])/2, the midpoint state Xc[k] satisfies
        Xc[k]  = (X[k] + X[k+1])/2 + h/8*(f_k - f_{k+1})
        X[k+1] = X[k] + h/6*(f_k + 4*f_c + f_{k+1}),  f_c = (Xc2[k], U_c)
    (Hermite interpolation of the state at the midpoint, then Simpson's
    rule for the dynamics). Simpson's rule is exact for the quadratic
    integrand of a linear u, so the cost is
        J_N = h/12 * sum_k (U[k]^2 + 4*U_c^2 + U[k+1]^2).
    The path constraint x1 <= L is again enforced at the nodes only.

    Note: with piecewise CONSTANT u, Hermite-Simpson and RK4 give the same
    discrete solution for these dynamics (difference ~1e-16), so a
    piecewise-linear control is used to obtain a distinct method.

Single shooting with an ODE solver (cost discretized, dynamics not):

    Only the objective and the control are discretized: U[k] is constant on
    [t[k], t[k+1]) and J_N = 1/2 * h * sum U[k]^2. The dynamics are NOT
    turned into algebraic constraints; instead the ODE x' = (x2, u) is
    integrated numerically with CasADi's integrator (SUNDIALS CVODES,
    abstol = reltol = 1e-12) over each interval, starting from the
    previous interval's end state. The decision variables are only the N
    controls; the states X[k] = F(X[k-1], U[k-1]) are implicit functions of U.
    The NLP has the terminal condition x[N] = (0, -1) as equality and
    x1[k] <= L at k = 1..N-1 as inequality, and is solved with IPOPT. CVODES
    provides only first-order sensitivities, so IPOPT uses a limited-memory
    BFGS Hessian approximation. Because the integrator is (numerically)
    exact for constant u, the discrete problem equals the RK4/QP problem,
    so the solution agrees with them up to the solver tolerances.

Condensed QP solved with SQP (single shooting in matrix form):

    With the exact discrete dynamics x[k+1] = A_d x[k] + B_d u[k],
        A_d = [[1, h], [0, 1]],  B_d = [h^2/2, h]^T,
    collect the controls in U = [u[0], ..., u[N-1]]^T. Then
        x[k] = A_d^k x0 + sum_{j<k} A_d^(k-1-j) B_d u[j],
    i.e. all states are affine in U:  X = Phi*x0 + Gamma*U. The problem
    becomes a QP in U only:
        min_U   1/2 * U^T H U,                H = h*I   (J_N = h/2 U^T U)
        s.t.    A U = b,                      A = Gamma_N,  b = x_f - Phi_N x0
                G U <= g,                     G = Gamma_x,  g = L*1 - Phi_x x0
    The equality is the terminal condition x[N] = (0, -1); the inequality
    is x1[k] <= L at the interior nodes k = 1..N-1 (x1[0] = x1[N] = 0 are
    fixed by the boundary conditions). Since H = h*I is positive definite
    and the constraints are linear, this is a strictly convex QP with a
    unique solution. It is solved with CasADi's sqpmethod (qpOASES for the
    QP subproblems); for a QP, SQP terminates after one iteration. It is
    the same discrete problem as RK4 shooting (RK4 equals the exact update
    here), but solved by an active-set method instead of an interior-point
    method, so the two solutions agree to the interior-point tolerance.

Unconstrained comparison (Euler-Lagrange equation):

    For F(y, y', y'') = 1/2*(y'')^2, the higher-order
    Euler-Lagrange equation is y''''(t) = 0. Thus y is a cubic
    polynomial. Applying the four boundary conditions gives
        y(t) = t*(1-t),  y''(t) = -2,  J = 2.
    Its maximum is 1/4 at t=1/2, so it satisfies the constraint when
    L >= 1/4. The default L_BOUND below is set to a smaller value so that
    the constraint is active.

Analytical constrained solution (for comparison):

    For L < 1/4 the bound is active. On the free arcs y'''' = 0, so y
    is a cubic; the solution is symmetric about t = 1/2.

    Case 1/6 <= L < 1/4 (single contact point at t = 1/2):
        On [0, 1/2]: y(t) = t + a*t^2 + b*t^3 with y(1/2) = L and
        y'(1/2) = 0, giving a = -4 + 12*L and b = 4 - 16*L.
        Mirror it for t > 1/2: y(t) = y(1 - t).

    Case L < 1/6 (boundary arc y = L):
        On [0, t1], y is a cubic with y(0) = 0, y'(0) = 1, y(t1) = L,
        y'(t1) = 0. Minimality (continuity of y'') requires y''(t1) = 0,
        which gives t1 = 3*L and
            y(t) = t - t^2/(3*L) + t^3/(27*L^2),   0 <= t <= 3*L,
            y(t) = L,                              3*L <= t <= 1 - 3*L,
            y(t) = y(1 - t),                       1 - 3*L <= t <= 1.
        The flat arc exists only if 3*L < 1/2, i.e. L < 1/6.

Run from the repository root:
    python examples\\solve_spline_problem.py
    python -m examples.spline_problem_scripts.spline_animator

Requires CasADi, NumPy, and Matplotlib.
"""

import time

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np


N_INTERVALS = 100
L_BOUND = 0.2
FINAL_TIME = 1.0


A_CONTINUOUS = np.array([[0.0, 1.0], [0.0, 0.0]])
B_CONTINUOUS = np.array([[0.0], [1.0]])


def analytical_solution(t):
    """Return the unconstrained Euler-Lagrange solution y(t) = t*(1-t)."""
    return t * (1.0 - t)


def constrained_analytical_solution(t, L):
    """Return the analytical solution of the problem with y(t) <= L."""
    t = np.asarray(t, dtype=float)
    if L >= 0.25:
        return t * (1.0 - t)
    s = np.minimum(t, 1.0 - t)
    if L >= 1.0 / 6.0:
        a, b = -4.0 + 12.0 * L, 4.0 - 16.0 * L
        return s + a * s**2 + b * s**3
    t1 = 3.0 * L
    cubic = s - s**2 / (3.0 * L) + s**3 / (27.0 * L**2)
    return np.where(s < t1, cubic, L)


def _euler_step(x1, x2, u, h):
    return x1 + h * x2, x2 + h * u


def _rk4_step(x1, x2, u, h):
    """One RK4 step for x' = (x2, u), holding u constant."""
    k1_1, k1_2 = x2, u
    k2_1, k2_2 = x2 + (h / 2) * k1_2, u
    k3_1, k3_2 = x2 + (h / 2) * k2_2, u
    k4_1, k4_2 = x2 + h * k3_2, u
    return (
        x1 + (h / 6) * (k1_1 + 2 * k2_1 + 2 * k3_1 + k4_1),
        x2 + (h / 6) * (k1_2 + 2 * k2_2 + 2 * k3_2 + k4_2),
    )


def _explicit_rk_step(a, b, x1, x2, u, h):
    """One explicit Runge-Kutta step (Butcher tableau a, b) for x' = (x2, u), u constant."""
    stages = []  # stage derivatives K_i = (x2 at the stage, u)
    for row in a:
        x2_stage = x2 + h * sum((w * K[1] for w, K in zip(row, stages)), 0.0)
        stages.append((x2_stage, u))
    return (
        x1 + h * sum(w * K[0] for w, K in zip(b, stages)),
        x2 + h * sum(w * K[1] for w, K in zip(b, stages)),
    )


# Dormand-Prince 5th-order weights (6 stages, fixed step); rows are the a_ij.
_DP5_A = (
    (),
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
)
_DP5_B = (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84)


def _rk5_step(x1, x2, u, h):
    """One fixed-step Dormand-Prince order-5 step for x' = (x2, u), u constant."""
    return _explicit_rk_step(_DP5_A, _DP5_B, x1, x2, u, h)


def _affine_step_matrices(step, h):
    """Extract A_d and B_d from an affine step function x_next = A_d*x+B_d*u."""
    origin = np.asarray(step(0.0, 0.0, 0.0, h), dtype=float)
    state_columns = [
        np.asarray(step(*state, 0.0, h), dtype=float) - origin
        for state in ((1.0, 0.0), (0.0, 1.0))
    ]
    control_column = (
        np.asarray(step(0.0, 0.0, 1.0, h), dtype=float) - origin
    )
    return np.column_stack(state_columns), control_column.reshape(2, 1)


def print_dynamics_analysis(h):
    """Print the continuous-time analysis and the one-step updates."""
    A_exact = np.eye(2) + h * A_CONTINUOUS
    B_exact = np.array([[h**2 / 2], [h]])
    A_euler, B_euler = _affine_step_matrices(_euler_step, h)
    A_rk4, B_rk4 = _affine_step_matrices(_rk4_step, h)
    A_rk5, B_rk5 = _affine_step_matrices(_rk5_step, h)

    print("\nDynamical systems analysis")
    print("--------------------------")
    print("xdot = A x + B u, with")
    print("A =\n", A_CONTINUOUS)
    print("B =\n", B_CONTINUOUS)
    print("Equilibria: x* = (c, 0), u* = 0 for arbitrary c.")
    print("Eigenvalues of A:", np.linalg.eigvals(A_CONTINUOUS))
    print("A has a repeated zero eigenvalue with a Jordan block: every equilibrium")
    print("is Lyapunov unstable (not asymptotically stable).")
    print(f"\nOne-step updates for h = {h:g} (x_next = A_d x + B_d u):")
    for name, A_d, B_d in (
        ("Exact ZOH", A_exact, B_exact),
        ("Forward Euler", A_euler, B_euler),
        ("RK4", A_rk4, B_rk4),
        ("RK5 (Dormand-Prince)", A_rk5, B_rk5),
    ):
        print(f"{name}:")
        print("A_d =\n", A_d)
        print("B_d =\n", B_d)


METHODS = ("euler", "rk4", "rk5", "hermite_simpson", "sqp_qp", "ode_shooting")
METHOD_LABELS = {
    "euler": "Shooting + Euler",
    "rk4": "Shooting + RK4",
    "rk5": "Shooting + RK5 (Dormand-Prince)",
    "hermite_simpson": "Collocation (Hermite-Simpson)",
    "sqp_qp": "Condensed QP (SQP)",
    "ode_shooting": "Single shooting + ODE solver (CVODES)",
}
# (color, linestyle, marker) so the methods stay distinguishable in print.
METHOD_STYLES = {
    "euler": ("tab:blue", "-", "s"),
    "rk4": ("tab:orange", "--", "o"),
    "rk5": ("tab:brown", (0, (5, 2)), "v"),
    "hermite_simpson": ("tab:green", "-.", "^"),
    "sqp_qp": ("tab:purple", ":", "x"),
    "ode_shooting": ("tab:red", (0, (3, 1, 1, 1)), "D"),
}


def condensed_qp_matrices(N, L):
    """Build the condensed QP  min 1/2 U'HU  s.t.  A U = b,  G U <= g.

    With the exact discrete dynamics x[k+1] = A_d x[k] + B_d u[k], all states
    are affine in U = (u[0], ..., u[N-1]):  X = Phi x0 + Gamma U, where
        x[k] = A_d^k x0 + sum_{j<k} A_d^(k-1-j) B_d u[j].
    Returns a dict with H, A, b, G, g and the state map (Phi, Gamma) for the
    stacked states (x[0], ..., x[N]).
    """
    h = FINAL_TIME / N
    A_d = np.array([[1.0, h], [0.0, 1.0]])
    B_d = np.array([[h**2 / 2.0], [h]])
    x0 = np.array([0.0, 1.0])
    x_final = np.array([0.0, -1.0])

    Phi = np.zeros((2 * (N + 1), 2))
    Gamma = np.zeros((2 * (N + 1), N))
    power = np.eye(2)
    powers = [power]
    for _ in range(N):
        power = A_d @ power
        powers.append(power)
    for k in range(N + 1):
        Phi[2 * k : 2 * k + 2] = powers[k]
        for j in range(k):
            Gamma[2 * k : 2 * k + 2, j] = (powers[k - 1 - j] @ B_d).ravel()

    Phi_x0 = Phi @ x0
    # Terminal condition: Phi_N x0 + Gamma_N U = x_f.
    A_eq = Gamma[2 * N : 2 * N + 2]
    b_eq = x_final - Phi_x0[2 * N : 2 * N + 2]
    # State constraint x1[k] <= L at the interior nodes (x1[0]=0, x1[N]=0 are fixed).
    rows = [2 * k for k in range(1, N)]
    G = Gamma[rows]
    g = L - Phi_x0[rows]
    return {
        "H": h * np.eye(N),
        "A": A_eq,
        "b": b_eq,
        "G": G,
        "g": g,
        "Phi": Phi,
        "Gamma": Gamma,
        "x0": x0,
    }


def default_guess(t):
    """Default initial guess (y, y', y'') = (t(1-t), 1-2t, -2), evaluated at t."""
    t = np.asarray(t, dtype=float)
    return t * (1.0 - t), 1.0 - 2.0 * t, np.full_like(t, -2.0)


def solve_condensed_qp(N, L, guess=default_guess):
    """Solve the condensed QP with CasADi's SQP method (qpOASES for the QP).

    The problem is a strictly convex QP, so SQP converges after one QP
    subproblem. `guess(t)` returns the initial (y, y', y'') on the time grid;
    only the control y'' at the interval starts is used. Returns
    (t, X, U, objective, iterations, return_status) and raises RuntimeError if
    the solver does not report success.
    """
    qp = condensed_qp_matrices(N, L)
    h = FINAL_TIME / N
    U = ca.MX.sym("U", N)
    objective = 0.5 * ca.mtimes(U.T, ca.mtimes(ca.DM(qp["H"]), U))
    g_expr = ca.vertcat(ca.mtimes(ca.DM(qp["A"]), U), ca.mtimes(ca.DM(qp["G"]), U))
    solver = ca.nlpsol(
        "spline_sqp",
        "sqpmethod",
        {"x": U, "f": objective, "g": g_expr},
        {
            "qpsol": "qpoases",
            "qpsol_options": {"printLevel": "none"},
            "print_header": False,
            "print_iteration": False,
            "print_status": False,
            "print_time": False,
            "tol_pr": 1e-10,
            "tol_du": 1e-10,
        },
    )
    lbg = np.concatenate((qp["b"], np.full(qp["g"].size, -np.inf)))
    ubg = np.concatenate((qp["b"], qp["g"]))
    try:
        t_nodes = np.linspace(0.0, FINAL_TIME, N + 1)
        solution = solver(x0=guess(t_nodes[:-1])[2], lbg=lbg, ubg=ubg)
    except RuntimeError as exc:
        raise RuntimeError(
            f"SQP failed (N={N}, L={L}); the discrete problem may be infeasible. "
            f"CasADi message: {str(exc).splitlines()[-1]}"
        ) from exc
    stats = solver.stats()
    if not stats["success"]:
        raise RuntimeError(
            f"SQP failed (N={N}, L={L}): {stats.get('return_status')}"
        )
    U_opt = solution["x"].full().flatten()

    X_flat = qp["Phi"] @ qp["x0"] + qp["Gamma"] @ U_opt
    X_opt = X_flat.reshape((N + 1, 2)).T
    t = np.linspace(0.0, FINAL_TIME, N + 1)
    return (
        t,
        X_opt,
        U_opt,
        0.5 * h * float(U_opt @ U_opt),
        stats["iter_count"],
        stats["return_status"],
    )


def _hermite_simpson_defects(X, Xc, U, h):
    """Hermite-Simpson collocation constraints for x' = (x2, u).

    The control is continuous and piecewise linear with node values U[k], so
    its midpoint value is U_c = (U[k] + U[k+1])/2. With f_k = (x2[k], U[k]),
    f_{k+1} = (x2[k+1], U[k+1]), f_c = (Xc2[k], U_c) and the midpoint state
    Xc[k] as extra decision variable:
        Xc[k]  = (X[k] + X[k+1])/2 + h/8*(f_k - f_{k+1})
        X[k+1] = X[k] + h/6*(f_k + 4*f_c + f_{k+1})
    """
    N = X.shape[1] - 1
    defects = []
    for k in range(N):
        u_c = (U[k] + U[k + 1]) / 2
        f_k = ca.vertcat(X[1, k], U[k])
        f_next = ca.vertcat(X[1, k + 1], U[k + 1])
        f_c = ca.vertcat(Xc[1, k], u_c)
        midpoint = (X[:, k] + X[:, k + 1]) / 2 + (h / 8) * (f_k - f_next)
        defects.append(Xc[:, k] - midpoint)
        defects.append(X[:, k + 1] - X[:, k] - (h / 6) * (f_k + 4 * f_c + f_next))
    return ca.vertcat(*defects)


def solve_ode_shooting(N, L, guess=default_guess):
    """Single shooting with the dynamics integrated by CVODES (cost discretized).

    `guess(t)` returns the initial (y, y', y''); only y'' at the interval
    starts is used. Returns (t, X, U, objective, iterations, return_status); raises
    RuntimeError if IPOPT does not report success.
    """
    h = FINAL_TIME / N
    x = ca.MX.sym("x", 2)
    u = ca.MX.sym("u")
    integrator = ca.integrator(
        "spline_ode",
        "cvodes",
        {"x": x, "p": u, "ode": ca.vertcat(x[1], u)},
        0.0,
        h,
        {"abstol": 1e-12, "reltol": 1e-12},
    )
    U = ca.MX.sym("U", N)
    x_k = ca.DM([0.0, 1.0])
    states = [x_k]
    for k in range(N):
        x_k = integrator(x0=x_k, p=U[k])["xf"]
        states.append(x_k)
    X = ca.horzcat(*states)
    g = ca.vertcat(X[:, N] - ca.DM([0.0, -1.0]), X[0, 1:N].T)
    solver = ca.nlpsol(
        "spline_ode_shooting",
        "ipopt",
        {"x": U, "f": 0.5 * h * ca.sumsqr(U), "g": g},
        {
            "ipopt.print_level": 0,
            "ipopt.tol": 1e-12,
            "ipopt.acceptable_tol": 1e-12,
            "ipopt.bound_relax_factor": 0.0,
            # CVODES offers no second-order sensitivities.
            "ipopt.hessian_approximation": "limited-memory",
            "print_time": False,
        },
    )
    lbg = np.concatenate((np.zeros(2), np.full(N - 1, -np.inf)))
    ubg = np.concatenate((np.zeros(2), np.full(N - 1, L)))
    t_nodes = np.linspace(0.0, FINAL_TIME, N + 1)
    solution = solver(x0=guess(t_nodes[:-1])[2], lbg=lbg, ubg=ubg)
    stats = solver.stats()
    if not stats["success"]:
        raise RuntimeError(
            f"IPOPT failed (ode_shooting, N={N}, L={L}): {stats.get('return_status')}"
        )
    U_opt = solution["x"].full().flatten()
    X_opt = np.asarray(ca.Function("X_fun", [U], [X])(U_opt)).reshape(2, N + 1)
    t = np.linspace(0.0, FINAL_TIME, N + 1)
    return t, X_opt, U_opt, float(solution["f"]), stats["iter_count"], stats["return_status"]


def solve_spline(method="rk4", N=N_INTERVALS, L=L_BOUND, guess=default_guess):
    """Solve the spline problem and add the wall-clock time ("time", seconds).

    `guess(t)` returns the initial (y, y', y'') evaluated on the array t; it
    initializes the states and controls of the solver (the control-only
    methods use y'' only). The time covers the whole call (problem and solver
    construction plus the solve), measured with time.perf_counter().
    """
    start = time.perf_counter()
    solution = _solve_spline(method, N, L, guess)
    solution["time"] = time.perf_counter() - start
    return solution


def _solve_spline(method, N, L, guess=default_guess):
    """Solve with shooting (Euler/RK4/RK5/ODE), Hermite-Simpson collocation or the QP.

    Shooting methods return U with N entries (piecewise constant control);
    Hermite-Simpson returns U with N+1 node values (piecewise linear control).
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if N < 2:
        raise ValueError("N must be at least 2")
    if L < 0:
        raise ValueError("L must be nonnegative because x1(0) = 0")

    if method in ("sqp_qp", "ode_shooting"):
        solve = solve_condensed_qp if method == "sqp_qp" else solve_ode_shooting
        t, X_opt, U_opt, objective_value, iterations, status = solve(N, L, guess)
        return {
            "t": t,
            "X": X_opt,
            "U": U_opt,
            "objective": objective_value,
            "method": method,
            "L": L,
            "iterations": iterations,
            "status": status,
        }

    h = FINAL_TIME / N
    collocation = method == "hermite_simpson"
    n_controls = N + 1 if collocation else N
    X = ca.MX.sym("X", 2, N + 1)
    U = ca.MX.sym("U", n_controls)
    Xc = ca.MX.sym("Xc", 2, N) if collocation else None

    if collocation:
        dynamics = _hermite_simpson_defects(X, Xc, U, h)
        # Simpson's rule is exact for the quadratic integrand of a linear u.
        u_mid = (U[:-1] + U[1:]) / 2
        objective = (h / 12) * (
            ca.sumsqr(U[:-1]) + 4 * ca.sumsqr(u_mid) + ca.sumsqr(U[1:])
        )
    else:
        steps = {"euler": _euler_step, "rk4": _rk4_step, "rk5": _rk5_step}
        step = steps[method]
        dynamics = ca.vertcat(
            *[
                X[:, k + 1] - ca.vertcat(*step(X[0, k], X[1, k], U[k], h))
                for k in range(N)
            ]
        )
        objective = 0.5 * h * ca.sumsqr(U)
    boundary_conditions = ca.vertcat(
        X[0, 0],
        X[1, 0] - 1,
        X[0, N],
        X[1, N] + 1,
    )
    path_constraints = X[0, :].T
    g = ca.vertcat(boundary_conditions, dynamics, path_constraints)
    equality_count = 4 + dynamics.shape[0]

    decision_variables = ca.vertcat(ca.vec(X), U)
    if collocation:
        decision_variables = ca.vertcat(decision_variables, ca.vec(Xc))
    nlp = {"x": decision_variables, "f": objective, "g": g}
    solver = ca.nlpsol(
        f"spline_{method}",
        "ipopt",
        nlp,
        {
            "ipopt.print_level": 0,
            "ipopt.tol": 1e-12,
            "ipopt.acceptable_tol": 1e-12,
            # IPOPT relaxes bounds by 1e-8 by default, which would let x1 exceed L.
            "ipopt.bound_relax_factor": 0.0,
            "print_time": False,
        },
    )

    t = np.linspace(0.0, FINAL_TIME, N + 1)
    y_guess, v_guess, u_guess = guess(t)
    X_guess = np.vstack((y_guess, v_guess))
    U_guess = u_guess if collocation else u_guess[:-1]
    z0 = np.concatenate((X_guess.reshape(-1, order="F"), U_guess))
    if collocation:
        y_mid, v_mid, _ = guess(0.5 * (t[:-1] + t[1:]))
        z0 = np.concatenate((z0, np.vstack((y_mid, v_mid)).reshape(-1, order="F")))
    lbg = np.concatenate((np.zeros(equality_count), np.full(N + 1, -np.inf)))
    ubg = np.concatenate((np.zeros(equality_count), np.full(N + 1, L)))

    solution = solver(x0=z0, lbg=lbg, ubg=ubg)
    stats = solver.stats()
    if not stats["success"]:
        raise RuntimeError(
            f"IPOPT failed ({method}, N={N}, L={L}): {stats.get('return_status')}"
        )
    z_opt = solution["x"].full().flatten()
    state_size = 2 * (N + 1)
    X_opt = z_opt[:state_size].reshape((2, N + 1), order="F")
    U_opt = z_opt[state_size : state_size + n_controls]

    return {
        "t": t,
        "X": X_opt,
        "U": U_opt,
        "objective": float(solution["f"]),
        "method": method,
        "L": L,
        "iterations": stats["iter_count"],
        "status": stats["return_status"],
    }


def reconstruct_trajectory(solution, points_per_interval=20):
    """Continuous y(t) between the nodes, consistent with each method's model.

    With tau = t - t[k], h = t[k+1] - t[k], y_k = x1[k], v_k = x2[k]:
      * RK4, RK5, QP, ODE shooting (piecewise constant u, exact response):
            y = y_k + v_k*tau + u_k*tau^2/2                    (quadratic)
      * Euler (its model has x1[k+1] = x1[k] + h*x2[k]):
            y = y_k + v_k*tau                                  (linear)
      * Hermite-Simpson: cubic Hermite interpolant of y through the node
        values y_k, y_{k+1} and slopes v_k, v_{k+1}.
    Every piece passes through both neighbouring nodes. Returns (t, y).
    """
    method = solution["method"]
    t_nodes = solution["t"]
    y_nodes, v_nodes = solution["X"]
    U = np.asarray(solution["U"]).ravel()
    t_dense, y_dense = [], []
    for k in range(len(t_nodes) - 1):
        h = t_nodes[k + 1] - t_nodes[k]
        tau = np.linspace(0.0, h, points_per_interval)
        if method == "euler":
            y = y_nodes[k] + v_nodes[k] * tau
        elif method == "hermite_simpson":
            s = tau / h
            y = (
                (2 * s**3 - 3 * s**2 + 1) * y_nodes[k]
                + (s**3 - 2 * s**2 + s) * h * v_nodes[k]
                + (-2 * s**3 + 3 * s**2) * y_nodes[k + 1]
                + (s**3 - s**2) * h * v_nodes[k + 1]
            )
        else:
            y = y_nodes[k] + v_nodes[k] * tau + 0.5 * U[k] * tau**2
        t_dense.extend(t_nodes[k] + tau)
        y_dense.extend(y)
    return np.array(t_dense), np.array(y_dense)


def plot_comparison(solutions):
    """Plot all solutions alongside the analytical solutions."""
    fig, ax = plt.subplots(figsize=(9, 5))
    t_fine = np.linspace(0.0, FINAL_TIME, 1001)
    L = next(iter(solutions.values()))["L"]
    ax.plot(t_fine, analytical_solution(t_fine), color="gray", ls="--", lw=1,
            label="Unconstrained: t(1-t)")
    ax.plot(t_fine, constrained_analytical_solution(t_fine, L), color="k", lw=4,
            alpha=0.25, label="Analytical constrained")
    for method, solution in solutions.items():
        color, linestyle, marker = METHOD_STYLES[method]
        t_dense, y_dense = reconstruct_trajectory(solution)
        ax.plot(
            t_dense,
            y_dense,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markevery=max(len(t_dense) // 20, 1),
            markersize=5,
            label=f"{METHOD_LABELS[method]} (J={solution['objective']:.6f})",
        )
    ax.axhline(L, color="red", linestyle=":", label=f"Constraint L={L:g}")
    ax.set_xlabel("Time t")
    ax.set_ylabel("y(t) = x1(t)")
    ax.set_title("State-constrained spline: shooting, collocation and QP/SQP")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    print_dynamics_analysis(FINAL_TIME / N_INTERVALS)
    solutions = {method: solve_spline(method=method) for method in METHODS}
    for method, solution in solutions.items():
        print(
            f"{METHOD_LABELS[method]} objective: {solution['objective']:.10f} "
            f"(time {solution['time']:.3f} s)"
        )
    print("Analytical unconstrained objective: 2.0000000000")
    plot_comparison(solutions)
    plt.show()
