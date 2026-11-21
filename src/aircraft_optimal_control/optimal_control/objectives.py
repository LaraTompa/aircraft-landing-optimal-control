import casadi as ca


def evaluate_objective(
    U,
    dt,
    objective_definition,
    control_indices,
):
    """
    Construct an integral objective from a control trajectory.

    Supported objective types:
        - integral_squared: integral of (u / scale)^2
        - integral:         integral of (u / scale)

    Uses left-node rectangular integration, matching the original
    implementation.
    """

    objective_type = objective_definition.type
    control_name = objective_definition.control
    scale = objective_definition.scale

    if control_name not in control_indices:
        raise ValueError(
            f"Unknown control '{control_name}'. "
            f"Available controls: {list(control_indices)}"
        )

    control_index = control_indices[control_name]
    control = U[control_index, :]

    if objective_type == "integral_squared":
        integrand = (control / scale) ** 2

    elif objective_type == "integral":
        integrand = control / scale

    else:
        raise ValueError(
            f"Unsupported objective type: {objective_type}"
        )

    return dt * ca.sum1(integrand.T)


def build_objectives(
    U,
    dt,
    objective_definitions,
    control_indices,
):
    """
    Construct all configured objectives.

    Returns a dictionary mapping objective names to CasADi expressions.
    """

    objectives = {}

    for name, definition in objective_definitions.items():
        objectives[name] = evaluate_objective(
            U=U,
            dt=dt,
            objective_definition=definition,
            control_indices=control_indices,
        )

    return objectives