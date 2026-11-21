
import casadi as ca

from aircraft_optimal_control.models.base import AircraftModel
from aircraft_optimal_control.models.aerodynamics import (
    compute_aerodynamic_forces,
)


class PointMassAircraft(AircraftModel):
    """
    Two-dimensional point-mass aircraft model.

    State:
        x = [s, h, V, gamma]

    Control:
        u = [T, CL]

    The model supports wind-aware dynamics, including the
    material acceleration experienced in a spatially varying
    wind field.
    """

    def __init__(
        self,
        aircraft_config,
        wind_model=None,
        wind_aware=True,
    ):
        self.aircraft = aircraft_config
        self.wind_model = wind_model
        self.wind_aware = wind_aware

        if self.wind_aware and self.wind_model is None:
            raise ValueError(
                "A wind model must be provided when wind_aware=True."
            )

    def dynamics(self, x, u, t):
        """
        Compute the continuous-time aircraft dynamics.

        Parameters
        ----------
        x : CasADi vector
            [s, h, V, gamma]

        u : CasADi vector
            [T, CL]

        t : float or CasADi expression
            Current time.

        Returns
        -------
        CasADi vector
            [s_dot, h_dot, V_dot, gamma_dot]
        """

        # ----------------------------------------------------
        # State and control variables
        # ----------------------------------------------------

        s = x[0]
        h = x[1]
        V = x[2]
        gamma = x[3]

        T = u[0]
        CL = u[1]

        # ----------------------------------------------------
        # Aircraft parameters
        # ----------------------------------------------------

        m = self.aircraft.mass
        g = self.aircraft.gravity

        # ----------------------------------------------------
        # Aerodynamic forces
        # ----------------------------------------------------

        _, _, L, D = compute_aerodynamic_forces(
            V,
            CL,
            self.aircraft,
        )

        # ----------------------------------------------------
        # Wind velocity
        # ----------------------------------------------------

        if self.wind_aware:

            wind_velocity = self.wind_model.velocity(
                s,
                h,
                t,
            )

            w_s = wind_velocity[0]
            w_h = wind_velocity[1]

        else:

            w_s = 0.0
            w_h = 0.0

        # ----------------------------------------------------
        # Ground-relative kinematics
        # ----------------------------------------------------

        s_dot = V * ca.cos(gamma) + w_s
        h_dot = V * ca.sin(gamma) + w_h

        # ----------------------------------------------------
        # Wind material acceleration
        # ----------------------------------------------------

        if self.wind_aware:

            ground_velocity = ca.vertcat(
                s_dot,
                h_dot,
            )

            wind_acceleration = (
                self.wind_model.material_acceleration(
                    s,
                    h,
                    ground_velocity,
                    t,
                )
            )

            a_w_s = wind_acceleration[0]
            a_w_h = wind_acceleration[1]

        else:

            a_w_s = 0.0
            a_w_h = 0.0

        # ----------------------------------------------------
        # Transform wind acceleration into flight coordinates
        # ----------------------------------------------------

        a_w_t = (
            ca.cos(gamma) * a_w_s
            + ca.sin(gamma) * a_w_h
        )

        a_w_n = (
            -ca.sin(gamma) * a_w_s
            + ca.cos(gamma) * a_w_h
        )

        # ----------------------------------------------------
        # Airspeed dynamics
        # ----------------------------------------------------

        V_dot = (
            (T - D) / m
            - g * ca.sin(gamma)
            - a_w_t
        )

        # ----------------------------------------------------
        # Flight-path-angle dynamics
        # ----------------------------------------------------

        gamma_dot = (
            (L - m * g * ca.cos(gamma)) / (m * V)
            - a_w_n / V
        )

        # ----------------------------------------------------
        # Return state derivatives
        # ----------------------------------------------------

        return ca.vertcat(
            s_dot,
            h_dot,
            V_dot,
            gamma_dot,
        )
