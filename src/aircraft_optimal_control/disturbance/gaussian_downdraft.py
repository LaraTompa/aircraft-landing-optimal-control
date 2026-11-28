
import casadi as ca

from aircraft_optimal_control.config.loader import WindConfig
from aircraft_optimal_control.disturbance.base import WindField


class GaussianWindField(WindField):
    """
    Stationary Gaussian wind field with a vertical downdraft.
    """

    def __init__(self, config: WindConfig):
        self.config = config

    def velocity(self, s, h, t):
        cfg = self.config

        # Gaussian spatial profile
        exponent = (
            -((s - cfg.s_center) / cfg.sigma_s) ** 2
            -((h - cfg.h_center) / cfg.sigma_h) ** 2
        )

        w_h = (
            cfg.vertical_wind_sign
            * cfg.max_vertical_wind
            * ca.exp(exponent)
        )

        w_s = cfg.horizontal_wind

        return ca.vertcat(w_s, w_h)

    def material_acceleration(self, s, h, ground_velocity, t):
        cfg = self.config

        # Evaluate the vertical wind
        wind = self.velocity(s, h, t)
        w_h = wind[1]

        # Spatial derivatives of the vertical wind
        dw_h_ds = (
            -2.0 * (s - cfg.s_center) / cfg.sigma_s**2
            * w_h
        )

        dw_h_dh = (
            -2.0 * (h - cfg.h_center) / cfg.sigma_h**2
            * w_h
        )

        # Material derivative for a stationary wind field:
        # D(w_h)/Dt = dw_h/ds * s_dot + dw_h/dh * h_dot
        a_w_s = 0.0

        a_w_h = (
            dw_h_ds * ground_velocity[0]
            + dw_h_dh * ground_velocity[1]
        )

        return ca.vertcat(a_w_s, a_w_h)
