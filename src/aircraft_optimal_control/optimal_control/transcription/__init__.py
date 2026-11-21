from aircraft_optimal_control.optimal_control.transcription.multiple_shooting import (
    add_multiple_shooting_constraints,
)

from aircraft_optimal_control.optimal_control.transcription.collocation import (
    add_hermite_simpson_constraints,
)


__all__ = [
    "add_multiple_shooting_constraints",
    "add_hermite_simpson_constraints",
]