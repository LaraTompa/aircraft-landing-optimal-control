
def add_boundary_constraints(
    opti,
    X,
    initial_state,
    terminal_state,
):
    """Apply specified initial and terminal state values.

    Mapping inputs allow free terminal components and free auxiliary states;
    vector inputs retain compatibility with the former all-fixed API.
    """
    if isinstance(initial_state, dict):
        for index, value in initial_state.items():
            if value is not None:
                opti.subject_to(X[index, 0] == value)
    else:
        opti.subject_to(X[:, 0] == initial_state)

    if isinstance(terminal_state, dict):
        for index, value in terminal_state.items():
            if value is not None:
                opti.subject_to(X[index, -1] == value)
    else:
        opti.subject_to(X[:, -1] == terminal_state)


def _apply_bounds(opti, variable, bounds):
    """Apply lower and upper bounds, allowing either bound to be None."""
    lower, upper = bounds

    if lower is not None:
        opti.subject_to(variable >= lower)

    if upper is not None:
        opti.subject_to(variable <= upper)


def add_state_constraints(
    opti,
    X,
    state_indices,
    state_bounds,
):
    """Apply configured state bounds over all discretization nodes."""
    for name, bounds in state_bounds.items():
        _apply_bounds(opti, X[state_indices[name], :], bounds)


def add_control_constraints(
    opti,
    U,
    control_indices,
    control_bounds,
):
    """Apply configured control bounds over all control intervals."""
    for name, bounds in control_bounds.items():
        _apply_bounds(opti, U[control_indices[name], :], bounds)


def add_minimax_clearance_constraint(
    opti,
    X,
    altitude_index,
    zeta_index,
    runway_height,
):
    """Enforce h_R - h(t) - zeta(t) <= 0 at transcription nodes."""
    opti.subject_to(
        runway_height - X[altitude_index, :] - X[zeta_index, :] <= 0
    )


def add_landing_constraints(
    opti,
    X,
    U,
    initial_state,
    terminal_state,
    state_indices,
    control_indices,
    state_bounds,
    control_bounds,
):
    """Apply all boundary, state, and control constraints."""

    add_boundary_constraints(
        opti,
        X,
        initial_state,
        terminal_state,
    )

    add_state_constraints(
        opti=opti,
        X=X,
        state_indices=state_indices,
        state_bounds=state_bounds,
    )

    add_control_constraints(
        opti=opti,
        U=U,
        control_indices=control_indices,
        control_bounds=control_bounds,
    )
