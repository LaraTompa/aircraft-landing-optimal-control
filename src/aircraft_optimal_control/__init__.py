
"""
Aircraft landing optimal control package.

Provides aircraft models, disturbance models, optimal control
formulations, numerical integration, configuration loading,
and visualization utilities.
"""

from aircraft_optimal_control.models import (
    AircraftModel,
    PointMassAircraft,
    compute_aerodynamic_forces,
)

from aircraft_optimal_control.optimal_control import (
    LandingOptimalControlProblem,
)

from aircraft_optimal_control.config.loader import (
    load_experiment,
)


__all__ = [
    "AircraftModel",
    "PointMassAircraft",
    "compute_aerodynamic_forces",
    "LandingOptimalControlProblem",
    "load_experiment",
]
