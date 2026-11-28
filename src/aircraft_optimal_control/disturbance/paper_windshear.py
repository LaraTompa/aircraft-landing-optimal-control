"""Steady Bulirsch–Montrone–Pesch equation (6) windshear field."""

import math

import casadi as ca

from aircraft_optimal_control.disturbance.base import WindField


_FT_TO_M = 0.3048
_A = 6.0e-8
_B = -4.0e-11
_C = -math.log(25.0 / 30.6) * 1.0e-12
_D = -8.02881e-8
_E = 6.28083e-11
_H_STAR_FT = 1000.0


class PaperWindshear(WindField):
    """Paper windshear profile, steady in ground-fixed coordinates.

    The paper defines position in feet and wind in ft/s. ``units="ft"``
    exposes those units directly. The default ``units="si"`` adapts the
    field to the repository's aircraft model interface (metres, m/s, m/s²)
    while evaluating the profile internally in the paper's units.
    """

    def __init__(self, nu: float = 1.0, units: str = "si"):
        if units not in {"si", "ft"}:
            raise ValueError("units must be either 'si' or 'ft'.")
        self.nu = float(nu)
        self.units = units

    @staticmethod
    def _profiles(x):
        """Return A, B, dA/dx, dB/dx for x in feet."""
        q = 4600.0 - x
        left_A = -50.0 + _A * x**3 + _B * x**4
        middle_A = (x - 2300.0) / 40.0
        right_A = 50.0 - _A * q**3 - _B * q**4
        A = ca.if_else(
            x < 500.0,
            left_A,
            ca.if_else(x <= 4100.0, middle_A,
                       ca.if_else(x <= 4600.0, right_A, 50.0)),
        )

        left_B = _D * x**3 + _E * x**4
        middle_B = -51.0 * ca.exp(-_C * (x - 2300.0) ** 4)
        right_B = _D * q**3 + _E * q**4
        B = ca.if_else(
            x < 500.0,
            left_B,
            ca.if_else(x <= 4100.0, middle_B,
                       ca.if_else(x <= 4600.0, right_B, 0.0)),
        )

        left_dA = 3.0 * _A * x**2 + 4.0 * _B * x**3
        right_dA = 3.0 * _A * q**2 + 4.0 * _B * q**3
        dA = ca.if_else(
            x < 500.0,
            left_dA,
            ca.if_else(x <= 4100.0, 1.0 / 40.0,
                       ca.if_else(x <= 4600.0, right_dA, 0.0)),
        )

        left_dB = 3.0 * _D * x**2 + 4.0 * _E * x**3
        middle_dB = 204.0 * _C * (x - 2300.0) ** 3 * ca.exp(
            -_C * (x - 2300.0) ** 4
        )
        right_dB = -3.0 * _D * q**2 - 4.0 * _E * q**3
        dB = ca.if_else(
            x < 500.0,
            left_dB,
            ca.if_else(x <= 4100.0, middle_dB,
                       ca.if_else(x <= 4600.0, right_dB, 0.0)),
        )
        return A, B, dA, dB

    def _to_paper_units(self, s, h):
        if self.units == "si":
            return s / _FT_TO_M, h / _FT_TO_M
        return s, h

    def velocity(self, s, h, t):
        x_ft, h_ft = self._to_paper_units(s, h)
        A, B, _, _ = self._profiles(x_ft)
        wx_ft_s = self.nu * A
        wh_ft_s = self.nu * (h_ft / _H_STAR_FT) * B
        if self.units == "si":
            return ca.vertcat(_FT_TO_M * wx_ft_s, _FT_TO_M * wh_ft_s)
        return ca.vertcat(wx_ft_s, wh_ft_s)

    def material_acceleration(self, s, h, ground_velocity, t):
        x_ft, h_ft = self._to_paper_units(s, h)
        A, B, dA, dB = self._profiles(x_ft)
        if self.units == "si":
            s_dot_ft_s = ground_velocity[0] / _FT_TO_M
            h_dot_ft_s = ground_velocity[1] / _FT_TO_M
        else:
            s_dot_ft_s = ground_velocity[0]
            h_dot_ft_s = ground_velocity[1]

        wx_dot_ft_s2 = self.nu * dA * s_dot_ft_s
        wh_dot_ft_s2 = self.nu * (
            (h_ft / _H_STAR_FT) * dB * s_dot_ft_s
            + (B / _H_STAR_FT) * h_dot_ft_s
        )
        if self.units == "si":
            return ca.vertcat(_FT_TO_M * wx_dot_ft_s2,
                              _FT_TO_M * wh_dot_ft_s2)
        return ca.vertcat(wx_dot_ft_s2, wh_dot_ft_s2)
