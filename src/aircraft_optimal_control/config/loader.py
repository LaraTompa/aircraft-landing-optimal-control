
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import math
import yaml


# ============================================================
# 1. Configuration dataclasses
# ============================================================

@dataclass(frozen=True)
class AircraftConfig:
    name: str
    mass: float
    gravity: float
    air_density: float
    wing_area: float

    CD0: float
    k_drag: float

    max_thrust: float
    thrust_scale: float

    CL_min: float
    CL_max: float


@dataclass(frozen=True)
class WindConfig:
    name: str
    type: str

    max_vertical_wind: float
    horizontal_wind: float
    vertical_wind_sign: float

    s_center: float
    h_center: float

    sigma_s: float
    sigma_h: float


@dataclass(frozen=True)
class BoundaryState:
    horizontal_position: float
    altitude: float
    airspeed: float
    flight_path_angle: float  # radians


@dataclass(frozen=True)
class TimeConfig:
    initial: float
    final: float
    intervals: int

    @property
    def dt(self) -> float:
        return (self.final - self.initial) / self.intervals


@dataclass(frozen=True)
class StateConstraints:
    altitude_min: float
    altitude_max: float | None

    airspeed_min: float
    airspeed_max: float

    flight_path_angle_min: float  # radians
    flight_path_angle_max: float  # radians


@dataclass(frozen=True)
class ControlConstraints:
    thrust_min: float
    thrust_max: float

    lift_coefficient_min: float
    lift_coefficient_max: float


@dataclass(frozen=True)
class ObjectiveDefinition:
    type: str
    control: str
    scale: float


@dataclass(frozen=True)
class InitialGuess:
    thrust: float
    lift_coefficient: float


@dataclass(frozen=True)
class ProblemConfig:
    name: str

    time: TimeConfig

    initial_state: BoundaryState
    terminal_state: BoundaryState

    state_constraints: StateConstraints
    control_constraints: ControlConstraints

    objectives: dict[str, ObjectiveDefinition]
    initial_guess: InitialGuess


@dataclass(frozen=True)
class DynamicsConfig:
    model: str
    wind_aware: bool


@dataclass(frozen=True)
class TranscriptionConfig:
    method: str
    integrator: str


@dataclass(frozen=True)
class SolverConfig:
    name: str
    options: dict[str, Any]


@dataclass(frozen=True)
class ExperimentSettings:
    type: str
    objectives: list[str]
    warm_start: bool


@dataclass(frozen=True)
class AnimationSettings:
    solution: str
    playback_speed: float
    interval: int


@dataclass(frozen=True)
class VisualizationConfig:
    plots: bool
    wind_field: bool
    animation: bool
    animation_settings: AnimationSettings


@dataclass(frozen=True)
class ExperimentConfig:
    name: str

    aircraft: AircraftConfig
    wind: WindConfig
    problem: ProblemConfig

    dynamics: DynamicsConfig
    transcription: TranscriptionConfig
    solver: SolverConfig

    experiment: ExperimentSettings
    visualization: VisualizationConfig

    config_path: Path


# ============================================================
# 2. YAML loading utilities
# ============================================================

def load_yaml(path: Path) -> dict[str, Any]:
    """
    Load a YAML file and return its contents as a dictionary.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a YAML mapping in {path}"
        )

    return data


def resolve_config_path(
    base_path: Path,
    relative_path: str,
) -> Path:
    """
    Resolve a configuration path relative to another YAML file.
    """

    return (base_path.parent / relative_path).resolve()


def require(mapping: dict, key: str, context: str):
    """
    Retrieve a required configuration field.
    """

    if key not in mapping:
        raise KeyError(
            f"Missing required field '{key}' in {context}"
        )

    return mapping[key]


# ============================================================
# 3. Aircraft configuration
# ============================================================

def load_aircraft_config(data: dict) -> AircraftConfig:

    aero = require(data, "aerodynamics", "aircraft")
    drag = require(aero, "drag", "aircraft.aerodynamics")
    thrust = require(aero, "thrust", "aircraft.aerodynamics")
    lift = require(
        aero,
        "lift_coefficient",
        "aircraft.aerodynamics",
    )

    config = AircraftConfig(
        name=require(data, "name", "aircraft"),

        mass=float(require(data, "mass", "aircraft")),
        gravity=float(require(data, "gravity", "aircraft")),

        air_density=float(
            require(aero, "air_density", "aircraft.aerodynamics")
        ),

        wing_area=float(
            require(aero, "wing_area", "aircraft.aerodynamics")
        ),

        CD0=float(require(drag, "CD0", "aircraft.aerodynamics.drag")),
        k_drag=float(require(drag, "k", "aircraft.aerodynamics.drag")),

        max_thrust=float(
            require(thrust, "max", "aircraft.aerodynamics.thrust")
        ),

        thrust_scale=float(
            require(thrust, "scale", "aircraft.aerodynamics.thrust")
        ),

        CL_min=float(
            require(lift, "min", "aircraft.aerodynamics.lift_coefficient")
        ),

        CL_max=float(
            require(lift, "max", "aircraft.aerodynamics.lift_coefficient")
        ),
    )

    if config.mass <= 0:
        raise ValueError("Aircraft mass must be positive.")

    if config.gravity <= 0:
        raise ValueError("Gravity must be positive.")

    if config.air_density <= 0 or config.wing_area <= 0:
        raise ValueError(
            "Air density and wing area must be positive."
        )

    if config.thrust_scale <= 0:
        raise ValueError("Thrust scale must be positive.")

    if config.max_thrust <= 0:
        raise ValueError("Maximum thrust must be positive.")

    if config.CL_min > config.CL_max:
        raise ValueError("Invalid lift coefficient bounds.")

    return config


# ============================================================
# 4. Wind configuration
# ============================================================

def load_wind_config(data: dict) -> WindConfig:

    wind_type = require(data, "type", "wind")

    if wind_type == "gaussian":

        center = require(data, "center", "wind")
        std = require(data, "standard_deviation", "wind")

        config = WindConfig(
            name=require(data, "name", "wind"),
            type=wind_type,

            max_vertical_wind=float(
                require(data, "max_vertical_wind", "wind")
            ),

            horizontal_wind=float(data.get("horizontal_wind", 0.0)),

            vertical_wind_sign=float(
                data.get("vertical_wind_sign", -1.0)
            ),

            s_center=float(
                require(center, "horizontal_position", "wind.center")
            ),

            h_center=float(
                require(center, "altitude", "wind.center")
            ),

            sigma_s=float(
                require(
                    std,
                    "horizontal_position",
                    "wind.standard_deviation",
                )
            ),

            sigma_h=float(
                require(
                    std,
                    "altitude",
                    "wind.standard_deviation",
                )
            ),
        )

        if config.sigma_s <= 0 or config.sigma_h <= 0:
            raise ValueError(
                "Gaussian wind standard deviations must be positive."
            )

        if config.max_vertical_wind < 0:
            raise ValueError(
                "Maximum vertical wind magnitude must be non-negative."
            )

        if abs(config.vertical_wind_sign) != 1:
            raise ValueError(
                "vertical_wind_sign must be either -1 or +1."
            )

        return config

    elif wind_type == "none":

        return WindConfig(
            name=data.get("name", "no_wind"),
            type=wind_type,

            max_vertical_wind=0.0,
            horizontal_wind=0.0,
            vertical_wind_sign=-1.0,

            s_center=0.0,
            h_center=0.0,

            sigma_s=1.0,
            sigma_h=1.0,
        )

    else:
        raise ValueError(
            f"Unsupported wind type: '{wind_type}'"
        )


# ============================================================
# 5. Problem configuration
# ============================================================

def load_problem_config(data: dict) -> ProblemConfig:

    time_data = require(data, "time", "problem")

    initial_data = require(data, "initial_state", "problem")
    terminal_data = require(data, "terminal_state", "problem")

    state_data = require(data, "state_constraints", "problem")
    control_data = require(data, "control_constraints", "problem")

    thrust_data = require(control_data, "thrust", "problem.control_constraints")
    lift_data = require(
        control_data,
        "lift_coefficient",
        "problem.control_constraints",
    )

    altitude_data = require(state_data, "altitude", "problem.state_constraints")
    speed_data = require(state_data, "airspeed", "problem.state_constraints")
    gamma_data = require(
        state_data,
        "flight_path_angle",
        "problem.state_constraints",
    )

    def load_boundary_state(state: dict, context: str) -> BoundaryState:

        return BoundaryState(
            horizontal_position=float(
                require(state, "horizontal_position", context)
            ),

            altitude=float(
                require(state, "altitude", context)
            ),

            airspeed=float(
                require(state, "airspeed", context)
            ),

            flight_path_angle=math.radians(
                float(require(state, "flight_path_angle", context))
            ),
        )

    time_config = TimeConfig(
        initial=float(require(time_data, "initial", "problem.time")),
        final=float(require(time_data, "final", "problem.time")),
        intervals=int(require(time_data, "intervals", "problem.time")),
    )

    if time_config.final <= time_config.initial:
        raise ValueError("Final time must be greater than initial time.")

    if time_config.intervals <= 0:
        raise ValueError("Number of intervals must be positive.")

    state_constraints = StateConstraints(
        altitude_min=float(require(altitude_data, "min", "altitude constraints")),

        altitude_max=(
            None
            if altitude_data.get("max") is None
            else float(altitude_data["max"])
        ),

        airspeed_min=float(require(speed_data, "min", "airspeed constraints")),
        airspeed_max=float(require(speed_data, "max", "airspeed constraints")),

        flight_path_angle_min=math.radians(
            float(require(gamma_data, "min", "flight-path-angle constraints"))
        ),

        flight_path_angle_max=math.radians(
            float(require(gamma_data, "max", "flight-path-angle constraints"))
        ),
    )

    control_constraints = ControlConstraints(
        thrust_min=float(require(thrust_data, "min", "thrust constraints")),
        thrust_max=float(require(thrust_data, "max", "thrust constraints")),

        lift_coefficient_min=float(
            require(lift_data, "min", "lift coefficient constraints")
        ),

        lift_coefficient_max=float(
            require(lift_data, "max", "lift coefficient constraints")
        ),
    )

    objective_data = require(data, "objectives", "problem")

    objectives = {}

    for name, definition in objective_data.items():

        objectives[name] = ObjectiveDefinition(
            type=str(require(definition, "type", f"objective '{name}'")),
            control=str(require(definition, "control", f"objective '{name}'")),
            scale=float(require(definition, "scale", f"objective '{name}'")),
        )

        if objectives[name].scale <= 0:
            raise ValueError(
                f"Objective scale must be positive: '{name}'"
            )

    guess_data = require(data, "initial_guess", "problem")

    initial_guess = InitialGuess(
        thrust=float(require(guess_data, "thrust", "problem.initial_guess")),

        lift_coefficient=float(
            require(
                guess_data,
                "lift_coefficient",
                "problem.initial_guess",
            )
        ),
    )

    config = ProblemConfig(
        name=require(data, "name", "problem"),

        time=time_config,

        initial_state=load_boundary_state(
            initial_data,
            "problem.initial_state",
        ),

        terminal_state=load_boundary_state(
            terminal_data,
            "problem.terminal_state",
        ),

        state_constraints=state_constraints,
        control_constraints=control_constraints,

        objectives=objectives,
        initial_guess=initial_guess,
    )

    if state_constraints.airspeed_min > state_constraints.airspeed_max:
        raise ValueError("Invalid airspeed bounds.")

    if state_constraints.flight_path_angle_min > state_constraints.flight_path_angle_max:
        raise ValueError("Invalid flight-path-angle bounds.")

    if control_constraints.thrust_min > control_constraints.thrust_max:
        raise ValueError("Invalid thrust bounds.")

    if control_constraints.lift_coefficient_min > control_constraints.lift_coefficient_max:
        raise ValueError("Invalid lift coefficient bounds.")

    if config.initial_state.airspeed <= 0 or config.terminal_state.airspeed <= 0:
        raise ValueError("Boundary airspeeds must be positive.")

    if config.initial_guess.thrust < control_constraints.thrust_min:
        raise ValueError("Initial thrust guess is below its lower bound.")

    if config.initial_guess.thrust > control_constraints.thrust_max:
        raise ValueError("Initial thrust guess is above its upper bound.")

    if config.initial_guess.lift_coefficient < control_constraints.lift_coefficient_min:
        raise ValueError("Initial lift coefficient guess is below its lower bound.")

    if config.initial_guess.lift_coefficient > control_constraints.lift_coefficient_max:
        raise ValueError("Initial lift coefficient guess is above its upper bound.")

    return config


# ============================================================
# 6. Experiment configuration
# ============================================================

def load_experiment(config_path: str | Path) -> ExperimentConfig:
    """
    Load a complete experiment configuration.

    The experiment YAML references separate aircraft, wind
    and problem YAML files. All paths are resolved relative
    to the experiment YAML file.
    """

    config_path = Path(config_path).resolve()

    experiment_data = load_yaml(config_path)

    # --------------------------------------------------------
    # Resolve referenced configuration files
    # --------------------------------------------------------

    aircraft_path = resolve_config_path(
        config_path,
        require(experiment_data, "aircraft", "experiment"),
    )

    wind_path = resolve_config_path(
        config_path,
        require(experiment_data, "wind", "experiment"),
    )

    problem_path = resolve_config_path(
        config_path,
        require(experiment_data, "problem", "experiment"),
    )

    # --------------------------------------------------------
    # Load individual configurations
    # --------------------------------------------------------

    aircraft = load_aircraft_config(
        load_yaml(aircraft_path)
    )

    wind = load_wind_config(
        load_yaml(wind_path)
    )

    problem = load_problem_config(
        load_yaml(problem_path)
    )

    # --------------------------------------------------------
    # Dynamics
    # --------------------------------------------------------

    dynamics_data = require(
        experiment_data,
        "dynamics",
        "experiment",
    )

    dynamics = DynamicsConfig(
        model=str(require(dynamics_data, "model", "dynamics")),
        wind_aware=bool(
            require(dynamics_data, "wind_aware", "dynamics")
        ),
    )

    # --------------------------------------------------------
    # Transcription
    # --------------------------------------------------------

    transcription_data = require(
        experiment_data,
        "transcription",
        "experiment",
    )

    transcription = TranscriptionConfig(
        method=str(
            require(transcription_data, "method", "transcription")
        ),

        integrator=str(
            require(transcription_data, "integrator", "transcription")
        ),
    )

    # --------------------------------------------------------
    # Solver
    # --------------------------------------------------------

    solver_data = require(
        experiment_data,
        "solver",
        "experiment",
    )

    solver = SolverConfig(
        name=str(require(solver_data, "name", "solver")),
        options=dict(solver_data.get("options", {})),
    )

    # --------------------------------------------------------
    # Experiment settings
    # --------------------------------------------------------

    experiment_settings_data = require(
        experiment_data,
        "experiment",
        "experiment configuration",
    )

    experiment_settings = ExperimentSettings(
        type=str(
            require(
                experiment_settings_data,
                "type",
                "experiment",
            )
        ),

        objectives=list(
            require(
                experiment_settings_data,
                "objectives",
                "experiment",
            )
        ),

        warm_start=bool(
            experiment_settings_data.get("warm_start", False)
        ),
    )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    visualization_data = experiment_data.get("visualization", {})

    animation_data = visualization_data.get(
        "animation_settings",
        {},
    )

    animation_settings = AnimationSettings(
        solution=str(animation_data.get("solution", "squared_thrust")),

        playback_speed=float(
            animation_data.get("playback_speed", 1.0)
        ),

        interval=int(animation_data.get("interval", 30)),
    )

    visualization = VisualizationConfig(
        plots=bool(visualization_data.get("plots", True)),

        wind_field=bool(
            visualization_data.get("wind_field", True)
        ),

        animation=bool(
            visualization_data.get("animation", False)
        ),

        animation_settings=animation_settings,
    )

    # --------------------------------------------------------
    # Construct complete configuration
    # --------------------------------------------------------

    config = ExperimentConfig(
        name=str(require(experiment_data, "name", "experiment")),

        aircraft=aircraft,
        wind=wind,
        problem=problem,

        dynamics=dynamics,
        transcription=transcription,
        solver=solver,

        experiment=experiment_settings,
        visualization=visualization,

        config_path=config_path,
    )

    # --------------------------------------------------------
    # Cross-configuration validation
    # --------------------------------------------------------

    if dynamics.model != "point_mass":
        raise ValueError(
            f"Unsupported aircraft dynamics model: {dynamics.model}"
        )

    if transcription.method not in {"multiple_shooting", "collocation"}:
        raise ValueError(
            f"Unsupported transcription method: {transcription.method}"
        )

    if transcription.integrator not in {"RK4", "Euler"}:
        raise ValueError(
            f"Unsupported integrator: {transcription.integrator}"
        )

    if solver.name.lower() != "ipopt":
        raise ValueError(
            f"Unsupported solver: {solver.name}"
        )

    for objective_name in experiment_settings.objectives:
        if objective_name not in problem.objectives:
            raise ValueError(
                f"Objective '{objective_name}' is not defined "
                f"in the problem configuration."
            )

    if (
        problem.control_constraints.thrust_max
        > aircraft.max_thrust
    ):
        raise ValueError(
            "Problem thrust upper bound exceeds the aircraft's "
            "maximum available thrust."
        )

    if (
        problem.control_constraints.lift_coefficient_min
        < aircraft.CL_min
        or problem.control_constraints.lift_coefficient_max
        > aircraft.CL_max
    ):
        raise ValueError(
            "Problem lift coefficient bounds exceed the "
            "aircraft's physical limits."
        )

    if (
        visualization.animation_settings.playback_speed <= 0
        or visualization.animation_settings.interval <= 0
    ):
        raise ValueError("Animation speed and interval must be positive.")

    if (
        visualization.animation_settings.solution
        not in experiment_settings.objectives
    ):
        raise ValueError(
            "The animation solution must be one of the "
            "objectives selected for this experiment."
        )

    return config
