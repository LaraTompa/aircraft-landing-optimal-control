import casadi as ca


def add_hermite_simpson_constraints(
    opti,
    X,
    U,
    dynamics,
    dt,
    N,
    t0=0.0,
):
    """
    Add Hermite-Simpson collocation constraints.

    The midpoint state is approximated using the Hermite interpolation
    relation. The midpoint control is the average of adjacent controls.

    X:
        State trajectory, shape (n_states, N + 1).

    U:
        Control trajectory, shape (n_controls, N).

    dynamics:
        Function dynamics(x, u, t) returning the state derivative.

    dt:
        Time-step size.

    N:
        Number of intervals.
    """

    for k in range(N):
        t_k = t0 + k * dt

        x_k = X[:, k]
        x_next = X[:, k + 1]
        u_k = U[:, k]

        # The control is piecewise constant over each interval.
        # Therefore, use the same control at both ends and midpoint.
        f_k = dynamics(x_k, u_k, t_k)
        f_next = dynamics(x_next, u_k, t_k + dt)

        # Hermite-Simpson midpoint state
        x_mid = (
            0.5 * (x_k + x_next)
            + (dt / 8.0) * (f_k - f_next)
        )

        f_mid = dynamics(
            x_mid,
            u_k,
            t_k + 0.5 * dt,
        )

        # Simpson integration defect
        defect = (
            x_next
            - x_k
            - (dt / 6.0) * (
                f_k + 4.0 * f_mid + f_next
            )
        )

        opti.subject_to(defect == 0)