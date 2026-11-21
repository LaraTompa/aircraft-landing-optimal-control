
import casadi as ca

from aircraft_optimal_control.config.loader import WindConfig
from aircraft_optimal_control.disturbance.base import WindField


class NoWindField(WindField):
    """
    Wind-free environment.
    """

    def __init__(self, config: WindConfig):
        self.config = config

    def velocity(self, s, h, t):
        return ca.vertcat(0.0, 0.0)

    def material_acceleration(self, s, h, ground_velocity, t):
        return ca.vertcat(0.0, 0.0)
