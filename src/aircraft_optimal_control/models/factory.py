"""Explicit model selection helpers for current and legacy dynamics."""

from aircraft_optimal_control.models.legacy_point_mass import LegacyPointMassAircraft
from aircraft_optimal_control.models.point_mass import PointMassAircraft


def create_aircraft_model(model, aircraft_config, wind_model=None,
                          wind_aware=True):
    """Construct a named model while preserving both model APIs.

    ``point_mass`` selects the paper's angle-of-attack model.
    ``legacy_point_mass`` preserves the original four-state/two-control model.
    """
    if model == "point_mass":
        return PointMassAircraft(aircraft_config, wind_model, wind_aware)
    if model == "legacy_point_mass":
        return LegacyPointMassAircraft(aircraft_config, wind_model, wind_aware)
    raise ValueError(f"Unsupported aircraft dynamics model: {model}")
