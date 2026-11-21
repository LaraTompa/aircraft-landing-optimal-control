
from abc import ABC, abstractmethod

import casadi as ca


class AircraftModel(ABC):
    """
    Abstract base class for aircraft dynamics models.

    State:
        x = [s, h, V, gamma]

    Control:
        u = [T, CL]

    Every aircraft model must implement:
        dynamics(x, u, t)
    """

    STATE_DIM = 4
    CONTROL_DIM = 2

    STATE_INDICES = {
        "horizontal_position": 0,
        "altitude": 1,
        "airspeed": 2,
        "flight_path_angle": 3,
    }

    CONTROL_INDICES = {
        "thrust": 0,
        "lift_coefficient": 1,
    }

    @abstractmethod
    def dynamics(self, x, u, t):
        """
        Evaluate the continuous-time state derivatives.

        Parameters
        ----------
        x : casadi.MX or casadi.DM
            State vector [s, h, V, gamma].

        u : casadi.MX or casadi.DM
            Control vector [T, CL].

        t : float or casadi expression
            Current time.

        Returns
        -------
        casadi expression
            State derivative [s_dot, h_dot, V_dot, gamma_dot].
        """
        raise NotImplementedError
