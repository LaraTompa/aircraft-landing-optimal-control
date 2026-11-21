
def add_boundary_constraints(
    opti,
    X,
    initial_state,
    terminal_state,
):
    """Apply fixed initial and terminal state constraints."""
    opti.subject_to(X[:, 0] == initial_state)
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
    altitude_index,
    airspeed_index,
    flight_path_angle_index,
    altitude_bounds,
    airspeed_bounds,
    flight_path_angle_bounds,
):
    """Apply state bounds over all discretization nodes."""

    _apply_bounds(
        opti,
        X[altitude_index, :],
        altitude_bounds,
    )

    _apply_bounds(
        opti,
        X[airspeed_index, :],
        airspeed_bounds,
    )

    _apply_bounds(
        opti,
        X[flight_path_angle_index, :],
        flight_path_angle_bounds,
    )


def add_control_constraints(
    opti,
    U,
    thrust_index,
    lift_coefficient_index,
    thrust_bounds,
    lift_coefficient_bounds,
):
    """Apply control bounds over all control intervals."""

    _apply_bounds(
        opti,
        U[thrust_index, :],
        thrust_bounds,
    )

    _apply_bounds(
        opti,
        U[lift_coefficient_index, :],
        lift_coefficient_bounds,
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
        altitude_index=state_indices["altitude"],
        airspeed_index=state_indices["airspeed"],
        flight_path_angle_index=state_indices["flight_path_angle"],
        altitude_bounds=state_bounds["altitude"],
        airspeed_bounds=state_bounds["airspeed"],
        flight_path_angle_bounds=state_bounds["flight_path_angle"],
    )

    add_control_constraints(
        opti=opti,
        U=U,
        thrust_index=control_indices["thrust"],
        lift_coefficient_index=control_indices["lift_coefficient"],
        thrust_bounds=control_bounds["thrust"],
        lift_coefficient_bounds=control_bounds["lift_coefficient"],
    )
