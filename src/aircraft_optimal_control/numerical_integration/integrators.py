
import casadi as ca


def euler_step(dynamics, x, u, t, dt):
    """
    One forward Euler integration step.

    Parameters
    ----------
    dynamics : callable
        Function dynamics(x, u, t) returning the state derivative.
    x : casadi expression
        Current state.
    u : casadi expression
        Control held constant over the integration interval.
    t : float or casadi expression
        Current time.
    dt : float
        Integration step size.

    Returns
    -------
    casadi expression
        State at the next integration point.
    """

    return x + dt * dynamics(x, u, t)


def rk4_step(dynamics, x, u, t, dt):
    """
    One classical fourth-order Runge-Kutta integration step.

    The control is assumed constant over the integration interval.
    """

    k1 = dynamics(x, u, t)

    k2 = dynamics(
        x + 0.5 * dt * k1,
        u,
        t + 0.5 * dt
    )

    k3 = dynamics(
        x + 0.5 * dt * k2,
        u,
        t + 0.5 * dt
    )

    k4 = dynamics(
        x + dt * k3,
        u,
        t + dt
    )

    return x + (dt / 6.0) * (
        k1 + 2.0 * k2 + 2.0 * k3 + k4
    )


def get_integrator(method):
    """
    Return an integration function based on its name.
    """

    integrators = {
        "euler": euler_step,
        "rk4": rk4_step,
    }

    method = method.lower()

    if method not in integrators:
        raise ValueError(
            f"Unknown integration method: {method}. "
            f"Available methods: {list(integrators)}"
        )

    return integrators[method]
