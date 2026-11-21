from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.aircraft_optimal_control.config.loader import load_experiment
from src.aircraft_optimal_control.models.point_mass import PointMassAircraft
from src.aircraft_optimal_control.disturbance.gaussian import GaussianWindField
from src.aircraft_optimal_control.disturbance.no_wind import NoWindField

from src.experiments.runner import run_comparison


def create_aircraft_model(config):

    if config.wind.type == "gaussian":
        wind_model = GaussianWindField(config.wind)

    elif config.wind.type == "none":
        wind_model = NoWindField(config.wind)

    else:
        raise ValueError(
            f"Unsupported wind type: {config.wind.type}"
        )

    return PointMassAircraft(
        aircraft_config=config.aircraft,
        wind_model=wind_model,
        wind_aware=config.dynamics.wind_aware,
    )


def main():

    config_path = (
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "baseline_pointmass_simple.yaml"
    )

    config = load_experiment(config_path)

    aircraft_model = create_aircraft_model(config)

    results_dir = PROJECT_ROOT / "results" / config.name

    run_comparison(
        config=config,
        aircraft_model=aircraft_model,
        results_dir=results_dir,
    )


if __name__ == "__main__":
    main()
