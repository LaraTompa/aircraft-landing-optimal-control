
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
class PhysicalAircraftParameters:
    """Physical constants for the angle-of-attack aircraft model."""

    mass: float
    wing_area: float
    thrust_axis_inclination: float
    gravity: float
    air_density: float


@dataclass(frozen=True)
class ThrustModelParameters:
    A0: float
    A1: float
    A2: float
    beta0: float
    beta_dot0: float
    t0: float


@dataclass(frozen=True)
class AerodynamicParameters:
    B0: float
    B1: float
    B2: float
    C0: float
    C1: float
    C2: float
    alpha_star: float
    alpha_max: float


@dataclass(frozen=True)
class PointMassAircraftConfig:
    """Physical, thrust, and aerodynamic parameters for PointMassAircraft.

    Coefficients must be identified for the chosen Airbus configuration;
    the repository's placeholder YAML intentionally does not supply values.
    Angles are in radians and SI units are used elsewhere.
    """

    name: str
    physical: PhysicalAircraftParameters
    thrust: ThrustModelParameters
    aerodynamics: AerodynamicParameters


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
    intensity: float = 1.0


@dataclass(frozen=True)
class BoundaryState:
    horizontal_position: float | None
    altitude: float | None
    airspeed: float | None
    flight_path_angle: float | None  # radians
    angle_of_attack: float | None = None  # radians; used by the paper model


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
    altitude_min: float | None
    altitude_max: float | None

    airspeed_min: float | None
    airspeed_max: float | None

    flight_path_angle_min: float | None  # radians
    flight_path_angle_max: float | None  # radians
    angle_of_attack_min: float | None = None
    angle_of_attack_max: float | None = None
    horizontal_position_min: float | None = None
    horizontal_position_max: float | None = None


@dataclass(frozen=True)
class ControlConstraints:
    thrust_min: float | None
    thrust_max: float | None

    lift_coefficient_min: float | None
    lift_coefficient_max: float | None
    angle_of_attack_rate_min: float | None = None
    angle_of_attack_rate_max: float | None = None


@dataclass(frozen=True)
class ObjectiveDefinition:
    type: str
    control: str | None = None
    scale: float | None = None
    theta: float | None = None
    runway_height: float | None = None


@dataclass(frozen=True)
class InitialGuess:
    thrust: float | None
    lift_coefficient: float | None
    angle_of_attack_rate: float | None = None


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

    aircraft: AircraftConfig | PointMassAircraftConfig
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


def load_point_mass_aircraft_config(
    path: str | Path,
) -> PointMassAircraftConfig:
    """Load parameters used by the paper-form PointMassAircraft.

    This schema is separate from the legacy point-mass configuration so
    its constant-CL coefficients and control limits cannot be mistaken for
    an identified angle-of-attack polar.
    """
    data = load_yaml(Path(path).resolve())
    physical = require(data, "physical", "aircraft")
    thrust = require(data, "thrust_model", "aircraft")
    aero = require(data, "aerodynamics", "aircraft")
    drag = require(aero, "drag", "aircraft.aerodynamics")
    lift = require(aero, "lift", "aircraft.aerodynamics")
    schedule = require(thrust, "power_schedule", "aircraft.thrust_model")

    fields = {
        "physical.mass": physical.get("mass"),
        "physical.wing_area": physical.get("wing_area"),
        "physical.thrust_axis_inclination": physical.get("thrust_axis_inclination"),
        "physical.gravity": physical.get("gravity"),
        "physical.air_density": physical.get("air_density"),
        **{f"thrust_model.{key}": thrust.get(key) for key in ("A0", "A1", "A2")},
        **{f"thrust_model.power_schedule.{key}": schedule.get(key) for key in ("beta0", "beta_dot0", "t0")},
        **{f"aerodynamics.drag.{key}": drag.get(key) for key in ("B0", "B1", "B2")},
        **{f"aerodynamics.lift.{key}": lift.get(key) for key in ("C0", "C1", "C2", "alpha_star", "alpha_max")},
    }
    missing = [key for key, value in fields.items() if value is None]
    if missing:
        raise ValueError(
            "Aircraft model parameters are unfilled placeholders: "
            + ", ".join(missing)
        )

    config = PointMassAircraftConfig(
        name=str(require(data, "name", "aircraft")),
        physical=PhysicalAircraftParameters(
            mass=float(require(physical, "mass", "aircraft.physical")),
            wing_area=float(require(physical, "wing_area", "aircraft.physical")),
            thrust_axis_inclination=float(require(physical, "thrust_axis_inclination", "aircraft.physical")),
            gravity=float(require(physical, "gravity", "aircraft.physical")),
            air_density=float(require(physical, "air_density", "aircraft.physical")),
        ),
        thrust=ThrustModelParameters(
            A0=float(require(thrust, "A0", "aircraft.thrust_model")),
            A1=float(require(thrust, "A1", "aircraft.thrust_model")),
            A2=float(require(thrust, "A2", "aircraft.thrust_model")),
            beta0=float(require(schedule, "beta0", "aircraft.thrust_model.power_schedule")),
            beta_dot0=float(require(schedule, "beta_dot0", "aircraft.thrust_model.power_schedule")),
            t0=float(require(schedule, "t0", "aircraft.thrust_model.power_schedule")),
        ),
        aerodynamics=AerodynamicParameters(
            B0=float(require(drag, "B0", "aircraft.aerodynamics.drag")),
            B1=float(require(drag, "B1", "aircraft.aerodynamics.drag")),
            B2=float(require(drag, "B2", "aircraft.aerodynamics.drag")),
            C0=float(require(lift, "C0", "aircraft.aerodynamics.lift")),
            C1=float(require(lift, "C1", "aircraft.aerodynamics.lift")),
            C2=float(require(lift, "C2", "aircraft.aerodynamics.lift")),
            alpha_star=float(require(lift, "alpha_star", "aircraft.aerodynamics.lift")),
            alpha_max=float(require(lift, "alpha_max", "aircraft.aerodynamics.lift")),
        ),
    )
    numeric_values = (
        *vars(config.physical).values(),
        *vars(config.thrust).values(),
        *vars(config.aerodynamics).values(),
    )
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError("Aircraft physical, thrust, and aerodynamic parameters must be finite.")
    if min(config.physical.mass, config.physical.wing_area,
           config.physical.gravity, config.physical.air_density) <= 0:
        raise ValueError("Mass, wing area, gravity and air density must be positive.")
    if config.thrust.t0 < 0:
        raise ValueError("Power schedule t0 must be non-negative.")
    if not 0.0 <= config.thrust.beta0 <= 1.0 or config.thrust.beta_dot0 < 0.0:
        raise ValueError("Power schedule must ramp from beta0 in [0, 1] with non-negative beta_dot0.")
    beta_at_ramp_end = config.thrust.beta0 + config.thrust.beta_dot0 * config.thrust.t0
    if not math.isclose(beta_at_ramp_end, 1.0, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError(
            "Power schedule must reach full power at t0: "
            "beta0 + beta_dot0 * t0 must equal 1."
        )
    if config.aerodynamics.alpha_max <= config.aerodynamics.alpha_star:
        raise ValueError("alpha_max must be greater than alpha_star.")
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

    elif wind_type == "paper_windshear":
        return WindConfig(
            name=data.get("name", "paper_windshear"),
            type=wind_type,
            max_vertical_wind=0.0,
            horizontal_wind=0.0,
            vertical_wind_sign=-1.0,
            s_center=0.0,
            h_center=0.0,
            sigma_s=1.0,
            sigma_h=1.0,
            intensity=float(data.get("nu", 1.0)),
        )

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

def load_problem_config(data: dict, model: str = "legacy_point_mass") -> ProblemConfig:

    time_data = require(data, "time", "problem")

    initial_data = require(data, "initial_state", "problem")
    terminal_data = require(data, "terminal_state", "problem")

    state_data = require(data, "state_constraints", "problem")
    control_data = require(data, "control_constraints", "problem")

    if model == "point_mass":
        alpha_data = require(state_data, "angle_of_attack", "problem.state_constraints")
        alpha_rate_data = require(control_data, "angle_of_attack_rate", "problem.control_constraints")
        thrust_data = lift_data = None
    else:
        thrust_data = require(control_data, "thrust", "problem.control_constraints")
        lift_data = require(control_data, "lift_coefficient", "problem.control_constraints")
        alpha_data = alpha_rate_data = None

    altitude_data = require(state_data, "altitude", "problem.state_constraints")
    speed_data = require(state_data, "airspeed", "problem.state_constraints")
    gamma_data = require(
        state_data,
        "flight_path_angle",
        "problem.state_constraints",
    )
    position_data = state_data.get("horizontal_position")

    def load_boundary_state(state: dict, context: str) -> BoundaryState:
        partial = model == "point_mass" and context.endswith("terminal_state")

        def number(key, converter=float):
            value = state.get(key)
            if value is None:
                required_value = (
                    not partial
                    and (key != "angle_of_attack" or model == "point_mass")
                )
                if required_value:
                    raise KeyError(f"Missing required field '{key}' in {context}")
                return None
            return converter(value)

        return BoundaryState(
            horizontal_position=number("horizontal_position"),
            altitude=number("altitude"),
            airspeed=number("airspeed"),
            flight_path_angle=number("flight_path_angle", math.radians),
            angle_of_attack=number("angle_of_attack", math.radians),
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
        altitude_min=(None if altitude_data.get("min") is None else float(altitude_data["min"])),

        altitude_max=(
            None
            if altitude_data.get("max") is None
            else float(altitude_data["max"])
        ),

        airspeed_min=(None if speed_data.get("min") is None else float(speed_data["min"])),
        airspeed_max=(None if speed_data.get("max") is None else float(speed_data["max"])),

        flight_path_angle_min=(None if gamma_data.get("min") is None else math.radians(float(gamma_data["min"]))),
        flight_path_angle_max=(None if gamma_data.get("max") is None else math.radians(float(gamma_data["max"]))),
        angle_of_attack_min=(
            None
            if alpha_data is None or alpha_data.get("min") is None
            else math.radians(float(alpha_data["min"]))
        ),
        angle_of_attack_max=(
            None
            if alpha_data is None or alpha_data.get("max") is None
            else math.radians(float(alpha_data["max"]))
        ),
        horizontal_position_min=(
            None if position_data is None or position_data.get("min") is None
            else float(position_data["min"])
        ),
        horizontal_position_max=(
            None if position_data is None or position_data.get("max") is None
            else float(position_data["max"])
        ),
    )

    if model == "point_mass":
        control_constraints = ControlConstraints(
            thrust_min=None,
            thrust_max=None,
            lift_coefficient_min=None,
            lift_coefficient_max=None,
            angle_of_attack_rate_min=math.radians(float(require(alpha_rate_data, "min", "angle-of-attack-rate constraints"))),
            angle_of_attack_rate_max=math.radians(float(require(alpha_rate_data, "max", "angle-of-attack-rate constraints"))),
        )
    else:
        control_constraints = ControlConstraints(
            thrust_min=float(require(thrust_data, "min", "thrust constraints")),
            thrust_max=float(require(thrust_data, "max", "thrust constraints")),
            lift_coefficient_min=float(require(lift_data, "min", "lift coefficient constraints")),
            lift_coefficient_max=float(require(lift_data, "max", "lift coefficient constraints")),
        )

    objective_data = require(data, "objectives", "problem")

    objectives = {}
    available_controls = (
        {"angle_of_attack_rate"}
        if model == "point_mass"
        else {"thrust", "lift_coefficient"}
    )

    for name, definition in objective_data.items():
        objective_type = str(require(definition, "type", f"objective '{name}'"))
        if objective_type == "minimax_clearance":
            if model != "point_mass":
                raise ValueError("The minimax clearance objective requires model: point_mass.")
            objectives[name] = ObjectiveDefinition(
                type=objective_type,
                theta=float(definition.get("theta", 1.0)),
                runway_height=float(require(definition, "runway_height", f"objective '{name}'")),
            )
            if objectives[name].theta <= 0:
                raise ValueError("Minimax objective theta must be positive.")
            continue

        objectives[name] = ObjectiveDefinition(
            type=objective_type,
            control=str(require(definition, "control", f"objective '{name}'")),
            scale=float(require(definition, "scale", f"objective '{name}'")),
        )

        if objectives[name].control not in available_controls:
            raise ValueError(
                f"Objective '{name}' uses control '{objectives[name].control}', "
                f"which is not available for model '{model}'."
            )

        if objectives[name].scale <= 0:
            raise ValueError(
                f"Objective scale must be positive: '{name}'"
            )

    guess_data = require(data, "initial_guess", "problem")

    if model == "point_mass":
        initial_guess = InitialGuess(
            thrust=None,
            lift_coefficient=None,
            angle_of_attack_rate=math.radians(float(require(guess_data, "angle_of_attack_rate", "problem.initial_guess"))),
        )
    else:
        initial_guess = InitialGuess(
            thrust=float(require(guess_data, "thrust", "problem.initial_guess")),
            lift_coefficient=float(require(guess_data, "lift_coefficient", "problem.initial_guess")),
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

    if (state_constraints.airspeed_min is not None
            and state_constraints.airspeed_max is not None
            and state_constraints.airspeed_min > state_constraints.airspeed_max):
        raise ValueError("Invalid airspeed bounds.")

    if (state_constraints.flight_path_angle_min is not None
            and state_constraints.flight_path_angle_max is not None
            and state_constraints.flight_path_angle_min > state_constraints.flight_path_angle_max):
        raise ValueError("Invalid flight-path-angle bounds.")

    if model == "point_mass":
        if control_constraints.angle_of_attack_rate_min > control_constraints.angle_of_attack_rate_max:
            raise ValueError("Invalid angle-of-attack-rate bounds.")
    else:
        if control_constraints.thrust_min > control_constraints.thrust_max:
            raise ValueError("Invalid thrust bounds.")
        if control_constraints.lift_coefficient_min > control_constraints.lift_coefficient_max:
            raise ValueError("Invalid lift coefficient bounds.")

    if (config.initial_state.airspeed is not None and config.initial_state.airspeed <= 0) or (
        config.terminal_state.airspeed is not None and config.terminal_state.airspeed <= 0
    ):
        raise ValueError("Boundary airspeeds must be positive.")

    if model == "point_mass":
        if config.initial_state.angle_of_attack is None:
            raise ValueError("Paper-model initial state requires angle_of_attack.")
        alpha_min = state_constraints.angle_of_attack_min
        alpha_max = state_constraints.angle_of_attack_max
        if alpha_min is not None and alpha_max is not None and alpha_min > alpha_max:
            raise ValueError("Invalid angle-of-attack bounds.")
        if alpha_min is not None and config.initial_state.angle_of_attack < alpha_min:
            raise ValueError("Initial angle of attack is below its lower bound.")
        if alpha_max is not None and config.initial_state.angle_of_attack > alpha_max:
            raise ValueError("Initial angle of attack is outside its bounds.")
        if config.terminal_state.angle_of_attack is not None and (
            (alpha_min is not None and config.terminal_state.angle_of_attack < alpha_min)
            or (alpha_max is not None and config.terminal_state.angle_of_attack > alpha_max)
        ):
            raise ValueError("Terminal angle of attack is outside its bounds.")
        if not control_constraints.angle_of_attack_rate_min <= initial_guess.angle_of_attack_rate <= control_constraints.angle_of_attack_rate_max:
            raise ValueError("Initial angle-of-attack-rate guess is outside its bounds.")
    else:
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

    dynamics_data = require(experiment_data, "dynamics", "experiment")
    model_name = str(require(dynamics_data, "model", "dynamics"))

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

    if model_name == "point_mass":
        aircraft = load_point_mass_aircraft_config(aircraft_path)
    elif model_name == "legacy_point_mass":
        aircraft = load_aircraft_config(load_yaml(aircraft_path))
    else:
        raise ValueError(f"Unsupported aircraft dynamics model: {model_name}")

    wind = load_wind_config(
        load_yaml(wind_path)
    )

    problem = load_problem_config(load_yaml(problem_path), model=model_name)

    # --------------------------------------------------------
    # Dynamics
    # --------------------------------------------------------

    dynamics = DynamicsConfig(
        model=model_name,
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

    if dynamics.model == "legacy_point_mass":
        if problem.control_constraints.thrust_max > aircraft.max_thrust:
            raise ValueError("Problem thrust upper bound exceeds the aircraft's maximum available thrust.")
        if (
            problem.control_constraints.lift_coefficient_min < aircraft.CL_min
            or problem.control_constraints.lift_coefficient_max > aircraft.CL_max
        ):
            raise ValueError("Problem lift coefficient bounds exceed the aircraft's physical limits.")
    else:
        alpha_min = problem.state_constraints.angle_of_attack_min
        alpha_max = problem.state_constraints.angle_of_attack_max
        if ((alpha_min is not None and alpha_min < -aircraft.aerodynamics.alpha_max)
                or (alpha_max is not None and alpha_max > aircraft.aerodynamics.alpha_max)):
            raise ValueError("Problem angle-of-attack bounds exceed the configured alpha_max.")
        for endpoint_name, endpoint in (
            ("initial", problem.initial_state),
            ("terminal", problem.terminal_state),
        ):
            alpha = endpoint.angle_of_attack
            if alpha is not None and alpha > aircraft.aerodynamics.alpha_max:
                raise ValueError(
                    f"The {endpoint_name} angle of attack exceeds the aircraft alpha_max."
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
