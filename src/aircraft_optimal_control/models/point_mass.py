"""Primary five-state aircraft point-mass model."""

import casadi as ca

from aircraft_optimal_control.config.loader import PointMassAircraftConfig
from aircraft_optimal_control.models.base import AircraftModel


def thrust_at(V, t, config: PointMassAircraftConfig):
    """Scheduled thrust, ramping until ``t0`` and holding full power after."""
    params = config.thrust
    beta = ca.if_else(
        t < params.t0,
        params.beta0 + params.beta_dot0 * t,
        1.0,
    )
    return beta * (params.A0 + params.A1 * V + params.A2 * V**2)


def drag_coefficient(alpha, config: PointMassAircraftConfig):
    p = config.aerodynamics
    return p.B0 + p.B1 * alpha + p.B2 * alpha**2


def lift_coefficient(alpha, config: PointMassAircraftConfig):
    p = config.aerodynamics
    linear = p.C0 + p.C1 * alpha
    stalled = linear + p.C2 * (alpha - p.alpha_star) ** 2
    return ca.if_else(alpha <= p.alpha_star, linear, stalled)


def compute_angle_of_attack_aerodynamics(V, alpha, config):
    """Return (qbar, CD, CL, L, D) for the configured angle polars."""
    physical = config.physical
    qbar = 0.5 * physical.air_density * V**2
    CD = drag_coefficient(alpha, config)
    CL = lift_coefficient(alpha, config)
    qS = qbar * physical.wing_area
    return qbar, CD, CL, qS * CL, qS * CD


class PointMassAircraft(AircraftModel):
    """Paper-form dynamics with state [x, h, V, gamma, alpha].

    The scalar control is alpha_dot. Thrust follows the configured power
    schedule and is not a control. Wind must implement the repository's
    velocity and material_acceleration interface.
    """

    STATE_DIM = 5
    CONTROL_DIM = 1
    STATE_INDICES = {
        "horizontal_position": 0,
        "altitude": 1,
        "airspeed": 2,
        "flight_path_angle": 3,
        "angle_of_attack": 4,
    }
    CONTROL_INDICES = {"angle_of_attack_rate": 0}

    def __init__(self, aircraft_config, wind_model=None, wind_aware=True):
        self.aircraft = aircraft_config
        self.wind_model = wind_model
        self.wind_aware = wind_aware
        if wind_aware and wind_model is None:
            raise ValueError("A wind model must be provided when wind_aware=True.")

    def dynamics(self, x, u, t):
        position, altitude, V, gamma, alpha = (x[i] for i in range(5))
        alpha_dot = u[0]
        physical = self.aircraft.physical
        _, _, _, L, D = compute_angle_of_attack_aerodynamics(
            V, alpha, self.aircraft
        )
        T = thrust_at(V, t, self.aircraft)
        delta = physical.thrust_axis_inclination

        if self.wind_aware:
            wind = self.wind_model.velocity(position, altitude, t)
            W_x, W_h = wind[0], wind[1]
        else:
            W_x = W_h = 0.0

        x_dot = V * ca.cos(gamma) + W_x
        h_dot = V * ca.sin(gamma) + W_h

        if self.wind_aware:
            wind_acc = self.wind_model.material_acceleration(
                position, altitude, ca.vertcat(x_dot, h_dot), t
            )
            Wdot_x, Wdot_h = wind_acc[0], wind_acc[1]
        else:
            Wdot_x = Wdot_h = 0.0

        V_dot = (
            T / physical.mass * ca.cos(alpha + delta)
            - D / physical.mass
            - physical.gravity * ca.sin(gamma)
            - Wdot_x * ca.cos(gamma)
            - Wdot_h * ca.sin(gamma)
        )
        gamma_dot = (
            T / (physical.mass * V) * ca.sin(alpha + delta)
            + L / (physical.mass * V)
            - physical.gravity / V * ca.cos(gamma)
            + (Wdot_x * ca.sin(gamma) - Wdot_h * ca.cos(gamma)) / V
        )
        return ca.vertcat(x_dot, h_dot, V_dot, gamma_dot, alpha_dot)
