
from abc import ABC, abstractmethod


class WindField(ABC):
    """
    Interface for wind-field models.

    Wind velocity is expressed in ground-fixed coordinates:
        velocity = [w_s, w_h]

    The material acceleration is the derivative of the wind velocity
    along the aircraft's ground-relative trajectory.
    """

    @abstractmethod
    def velocity(self, s, h, t):
        """
        Return wind velocity [m/s] at position (s, h) and time t.
        """
        pass

    @abstractmethod
    def material_acceleration(self, s, h, ground_velocity, t):
        """
        Return the material derivative of wind velocity [m/s^2].

        Parameters
        ----------
        s, h : float or CasADi expression
            Horizontal position and altitude.

        ground_velocity : vector
            Aircraft ground-relative velocity [s_dot, h_dot].

        t : float or CasADi expression
            Time.
        """
        pass
