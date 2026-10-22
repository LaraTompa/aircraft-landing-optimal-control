"""
State-constrained spline problem
================================

Continuous-time problem:

    Minimiere
        J = 1/2 * integral_0^1 (y''(t))^2 dt

    unter den Nebenbedingungen
        y(t) <= L,                    t in [0, 1]
        y(0) = 0,  y(1) = 0
        y'(0) = 1, y'(1) = -1

Optimal-control formulation:

    x1(t) = y(t),  x2(t) = y'(t),  u(t) = y''(t)

    Minimiere
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
    eigenvalues are zero. A is a nilpotent Jordan block; the unforced
    origin is not asymptotically stable (a nonzero initial x2 causes
    x1 to grow linearly).

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

    For either method the piecewise-constant-control cost is
        J_N = 1/2 * h * sum(k=0..N-1) U[k]^2,
    which is the exact integral of u^2 for this control parameterization.
    The path constraint is enforced at nodes only, as a standard
    finite-dimensional approximation of the all-times constraint.

Unconstrained comparison (Euler-Lagrange equation):

    For F(y, y', y'') = 1/2*(y'')^2, the higher-order
    Euler-Lagrange equation is y''''(t) = 0. Thus y is a cubic
    polynomial. Applying the four boundary conditions gives
        y(t) = t*(1-t),  y''(t) = -2,  J = 2.
    Its maximum is 1/4 at t=1/2, so it satisfies the constraint when
    L >= 1/4. The default below is L=0.25.

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
    python examples\\spline_problem.py
    python examples\\spline_animator.py

Requires CasADi, NumPy, and Matplotlib.
"""

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np


N_INTERVALS = 100
L_BOUND = 0.1
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
    """Print the continuous-time analysis and three one-step updates."""
    A_exact = np.eye(2) + h * A_CONTINUOUS
    B_exact = np.array([[h**2 / 2], [h]])
    A_euler, B_euler = _affine_step_matrices(_euler_step, h)
    A_rk4, B_rk4 = _affine_step_matrices(_rk4_step, h)

    print("\nDynamical systems analysis")
    print("--------------------------")
    print("xdot = A x + B u, with")
    print("A =\n", A_CONTINUOUS)
    print("B =\n", B_CONTINUOUS)
    print("Equilibria: x* = (c, 0), u* = 0 for arbitrary c.")
    print("Eigenvalues of A:", np.linalg.eigvals(A_CONTINUOUS))
    print("A has a repeated zero eigenvalue and a Jordan block;")
    print("the unforced origin is not asymptotically stable.")
    print(f"\nOne-step updates for h = {h:g} (x_next = A_d x + B_d u):")
    for name, A_d, B_d in (
        ("Exact ZOH", A_exact, B_exact),
        ("Forward Euler", A_euler, B_euler),
        ("RK4", A_rk4, B_rk4),
    ):
        print(f"{name}:")
        print("A_d =\n", A_d)
        print("B_d =\n", B_d)


def solve_spline(method="rk4", N=N_INTERVALS, L=L_BOUND):
    """Solve the multiple-shooting transcription using Euler or RK4."""
    if method not in ("euler", "rk4"):
        raise ValueError("method must be 'euler' or 'rk4'")
    if N < 1:
        raise ValueError("N must be at least 1")
    if L < 0:
        raise ValueError("L must be nonnegative because x1(0) = 0")

    h = FINAL_TIME / N
    X = ca.MX.sym("X", 2, N + 1)
    U = ca.MX.sym("U", N)

    next_states = []
    for k in range(N):
        x1, x2, u = X[0, k], X[1, k], U[k]
        if method == "euler":
            next_states.append(ca.vertcat(*_euler_step(x1, x2, u, h)))
        else:
            next_states.append(ca.vertcat(*_rk4_step(x1, x2, u, h)))

    dynamics = ca.vertcat(
        *[X[:, k + 1] - next_states[k] for k in range(N)]
    )
    boundary_conditions = ca.vertcat(
        X[0, 0],
        X[1, 0] - 1,
        X[0, N],
        X[1, N] + 1,
    )
    path_constraints = X[0, :].T
    g = ca.vertcat(boundary_conditions, dynamics, path_constraints)
    equality_count = 4 + 2 * N

    objective = 0.5 * h * ca.sumsqr(U)
    decision_variables = ca.vertcat(ca.vec(X), U)
    nlp = {"x": decision_variables, "f": objective, "g": g}
    solver = ca.nlpsol(
        f"spline_{method}",
        "ipopt",
        nlp,
        {"ipopt.print_level": 0, "print_time": False},
    )

    t = np.linspace(0.0, FINAL_TIME, N + 1)
    X_guess = np.vstack((analytical_solution(t), 1.0 - 2.0 * t))
    U_guess = np.full(N, -2.0)
    z0 = np.concatenate((X_guess.reshape(-1, order="F"), U_guess))
    lbg = np.concatenate((np.zeros(equality_count), np.full(N + 1, -np.inf)))
    ubg = np.concatenate((np.zeros(equality_count), np.full(N + 1, L)))

    solution = solver(x0=z0, lbg=lbg, ubg=ubg)
    z_opt = solution["x"].full().flatten()
    state_size = 2 * (N + 1)
    X_opt = z_opt[:state_size].reshape((2, N + 1), order="F")
    U_opt = z_opt[state_size:]

    return {
        "t": t,
        "X": X_opt,
        "U": U_opt,
        "objective": float(solution["f"]),
        "method": method,
        "L": L,
    }


def plot_comparison(solutions):
    """Plot Euler and RK4 trajectories alongside the analytical solution."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, solution in solutions.items():
        ax.plot(
            solution["t"],
            solution["X"][0],
            label=f"{method.title()} (J={solution['objective']:.6f})",
        )

    t_fine = np.linspace(0.0, FINAL_TIME, 1001)
    L = next(iter(solutions.values()))["L"]
    ax.plot(t_fine, analytical_solution(t_fine), "k--", label="Unconstrained: t(1-t)")
    ax.plot(
        t_fine,
        constrained_analytical_solution(t_fine, L),
        "g:",
        linewidth=2,
        label="Analytical constrained",
    )
    ax.axhline(L, color="red", linestyle=":", label=f"Constraint L={L:g}")
    ax.set_xlabel("Time t")
    ax.set_ylabel("y(t) = x1(t)")
    ax.set_title("State-constrained spline: direct multiple shooting")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    print_dynamics_analysis(FINAL_TIME / N_INTERVALS)
    solutions = {
        method: solve_spline(method=method)
        for method in ("euler", "rk4")
    }
    for method, solution in solutions.items():
        print(f"{method.title()} objective: {solution['objective']:.10f}")
    print("Analytical unconstrained objective: 2.0000000000")
    plot_comparison(solutions)
    plt.show()
