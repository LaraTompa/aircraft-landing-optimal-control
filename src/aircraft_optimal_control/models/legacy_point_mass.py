import casadi as ca

from aircraft_optimal_control.models.base import AircraftModel
from aircraft_optimal_control.models.aerodynamics import compute_aerodynamic_forces


class LegacyPointMassAircraft(AircraftModel):
    """Original four-state model with thrust and CL as controls."""

    STATE_DIM = 4
    CONTROL_DIM = 2
    STATE_INDICES = {
        "horizontal_position": 0,
        "altitude": 1,
        "airspeed": 2,
        "flight_path_angle": 3,
    }
    CONTROL_INDICES = {"thrust": 0, "lift_coefficient": 1}

    def __init__(self, aircraft_config, wind_model=None, wind_aware=True):
        self.aircraft = aircraft_config
        self.wind_model = wind_model
        self.wind_aware = wind_aware
        if wind_aware and wind_model is None:
            raise ValueError("A wind model must be provided when wind_aware=True.")

    def dynamics(self, x, u, t):
        s, h, V, gamma = (x[i] for i in range(4))
        T, CL = u[0], u[1]
        m, g = self.aircraft.mass, self.aircraft.gravity
        _, _, L, D = compute_aerodynamic_forces(V, CL, self.aircraft)

        if self.wind_aware:
            wind = self.wind_model.velocity(s, h, t)
            w_s, w_h = wind[0], wind[1]
        else:
            w_s = w_h = 0.0

        s_dot = V * ca.cos(gamma) + w_s
        h_dot = V * ca.sin(gamma) + w_h

        if self.wind_aware:
            acceleration = self.wind_model.material_acceleration(
                s, h, ca.vertcat(s_dot, h_dot), t
            )
            a_w_s, a_w_h = acceleration[0], acceleration[1]
        else:
            a_w_s = a_w_h = 0.0

        a_w_t = ca.cos(gamma) * a_w_s + ca.sin(gamma) * a_w_h
        a_w_n = -ca.sin(gamma) * a_w_s + ca.cos(gamma) * a_w_h
        V_dot = (T - D) / m - g * ca.sin(gamma) - a_w_t
        gamma_dot = (L - m * g * ca.cos(gamma)) / (m * V) - a_w_n / V
        return ca.vertcat(s_dot, h_dot, V_dot, gamma_dot)
