from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.aircraft_optimal_control.config.loader import load_experiment
from src.aircraft_optimal_control.models.factory import create_aircraft_model as make_aircraft_model
from src.aircraft_optimal_control.disturbance.gaussian_downdraft import GaussianWindField
from src.aircraft_optimal_control.disturbance.no_wind import NoWindField
from src.aircraft_optimal_control.disturbance.paper_windshear import PaperWindshear

from src.experiments.runner import run_comparison


def create_aircraft_model(config):

    if config.wind.type == "gaussian":
        wind_model = GaussianWindField(config.wind)

    elif config.wind.type == "none":
        wind_model = NoWindField(config.wind)

    elif config.wind.type == "paper_windshear":
        wind_model = PaperWindshear(
            nu=config.wind.intensity,
            units="si",
        )

    else:
        raise ValueError(
            f"Unsupported wind type: {config.wind.type}"
        )

    return make_aircraft_model(
        model=config.dynamics.model,
        aircraft_config=config.aircraft,
        wind_model=wind_model,
        wind_aware=config.dynamics.wind_aware,
    )


def main():

    experiment_file = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "baseline_pointmass_simple.yaml"
    )
    config_path = (
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / experiment_file
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
