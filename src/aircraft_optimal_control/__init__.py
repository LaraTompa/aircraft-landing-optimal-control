
"""
Aircraft landing optimal control package.

Provides aircraft models, disturbance models, optimal control
formulations, numerical integration, configuration loading,
and visualization utilities.
"""

from aircraft_optimal_control.models import (
    AircraftModel,
    LegacyPointMassAircraft,
    PointMassAircraft,
    compute_aerodynamic_forces,
    create_aircraft_model,
)

from aircraft_optimal_control.optimal_control import (
    LandingOptimalControlProblem,
)

from aircraft_optimal_control.config.loader import (
    load_experiment,
)


__all__ = [
    "AircraftModel",
    "LegacyPointMassAircraft",
    "PointMassAircraft",
    "compute_aerodynamic_forces",
    "create_aircraft_model",
    "LandingOptimalControlProblem",
    "load_experiment",
]
