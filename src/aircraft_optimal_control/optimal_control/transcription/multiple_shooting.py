def add_multiple_shooting_constraints(
    opti,
    X,
    U,
    dynamics,
    integrator,
    dt,
    N,
    t0=0.0,
):
    """
    Add multiple-shooting dynamic equality constraints.

    X:
        State trajectory, shape (n_states, N + 1).

    U:
        Control trajectory, shape (n_controls, N).

    dynamics:
        Function dynamics(x, u, t) returning the state derivative.

    integrator:
        Numerical integration function, e.g. rk4_step.

    dt:
        Time-step size.

    N:
        Number of shooting intervals.
    """

    for k in range(N):
        t_k = t0 + k * dt

        x_next = integrator(
            dynamics,
            X[:, k],
            U[:, k],
            t_k,
            dt,
        )

        opti.subject_to(X[:, k + 1] == x_next)