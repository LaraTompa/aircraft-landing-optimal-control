
from abc import ABC, abstractmethod

import casadi as ca


class AircraftModel(ABC):
    """
    Abstract base class for aircraft dynamics models.

    Concrete models declare their own state and control dimensions and
    implement ``dynamics(x, u, t)``.
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
            Model-specific state vector.

        u : casadi.MX or casadi.DM
            Model-specific control vector.

        t : float or casadi expression
            Current time.

        Returns
        -------
        casadi expression
            Model-specific state derivative.
        """
        raise NotImplementedError
