
from pathlib import Path
import json

import numpy as np

from aircraft_optimal_control.optimal_control.problem import (
    LandingOptimalControlProblem,
)
from aircraft_optimal_control.numerical_integration.integrators import get_integrator


def build_ocp(config, aircraft_model):
    """
    Build the landing optimal control problem from the experiment config.
    """

    if config.transcription.method == "multiple_shooting":
        integrator = get_integrator(config.transcription.integrator)

    elif config.transcription.method == "collocation":
        integrator = None

    else:
        raise ValueError(
            f"Unsupported transcription method: "
            f"{config.transcription.method}"
        )

    objective_names = config.experiment.objectives

    problem = LandingOptimalControlProblem(
        aircraft_model=aircraft_model,
        problem_config=config.problem,
        integrator=integrator,
        objective_names=objective_names,
        solver_name=config.solver.name,
        solver_options=config.solver.options,
        transcription_method=config.transcription.method,
    )

    return problem


def run_comparison(config, aircraft_model, results_dir=None):
    """
    Solve one configured objective or compare two configured objectives.

    The first objective is solved from the problem's initial guess.
    If warm_start is enabled in a comparison, the first solution initializes
    the second solve.
    """

    objective_names = config.experiment.objectives

    if len(objective_names) not in {1, 2}:
        raise ValueError(
            "Configure one objective or a two-objective comparison."
        )

    problem = build_ocp(config, aircraft_model)

    results = {}

    for index, objective_name in enumerate(objective_names):
        print("\n" + "=" * 60)
        print(f"Solving objective: {objective_name}")
        print("=" * 60)

        solution = problem.solve(objective_name)
        extracted = problem.extract_solution(solution)

        results[objective_name] = extracted

        print(f"Objective value: {extracted['objectives'][objective_name]:.8f}")

        if (
            len(objective_names) == 2
            and index == 0
            and config.experiment.warm_start
        ):
            print("Using the first solution as a warm start.")
            problem.set_warm_start(
                extracted["X"],
                extracted["U"],
            )

    if results_dir is not None:
        results_dir = Path(results_dir)
        save_results(results, config, results_dir)

    if (
        config.visualization.plots
        or config.visualization.wind_field
        or config.visualization.animation
    ):
        if results_dir is None:
            results_dir = Path("results") / config.name

        run_visualizations(
            results=results,
            config=config,
            aircraft_model=aircraft_model,
            results_dir=Path(results_dir),
        )

    return results


def save_results(results, config, results_dir):
    """
    Save numerical solutions and experiment metadata.
    """

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    arrays = {}

    metadata = {
        "experiment": config.name,
        "objectives": {},
    }

    for objective_name, solution in results.items():
        arrays[f"{objective_name}_X"] = solution["X"]
        arrays[f"{objective_name}_U"] = solution["U"]
        arrays[f"{objective_name}_time"] = solution["time"]
        arrays[f"{objective_name}_control_time"] = solution["control_time"]

        metadata["objectives"][objective_name] = {
            name: float(value)
            for name, value in solution["objectives"].items()
        }

    np.savez_compressed(
        results_dir / "solutions.npz",
        **arrays,
    )

    with open(results_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print(f"\nResults saved to: {results_dir}")


def run_visualizations(results, config, aircraft_model, results_dir):
    """
    Generate the plots, wind-field visualization, and aircraft animation
    enabled in the experiment configuration.
    """

    import matplotlib.pyplot as plt

    from aircraft_optimal_control.visualization.aircraft_plots import (
        plot_solution,
    )
    from aircraft_optimal_control.visualization.wind_plots import (
        plot_wind_field,
    )
    from aircraft_optimal_control.visualization.airplane_animator import (
        animate_solution,
    )

    results_dir = Path(results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    wind_model = getattr(aircraft_model, "wind_model", None)

    # ---------------------------------------------------------
    # Solution plots
    # ---------------------------------------------------------

    if config.visualization.plots:
        for objective_name, solution in results.items():
            plot_solution(
                solution=solution,
                objective_name=objective_name,
                wind_model=wind_model,
                aircraft_model=aircraft_model,
                output_path=plots_dir / f"{objective_name}_solution.png",
                show=False,
            )

    # ---------------------------------------------------------
    # Wind-field plot
    # ---------------------------------------------------------

    if config.visualization.wind_field and wind_model is not None:
        first_objective = config.experiment.objectives[0]
        solution = results[first_objective]

        X = np.asarray(solution["X"])

        s_min, s_max = np.min(X[0, :]), np.max(X[0, :])
        h_min, h_max = np.min(X[1, :]), np.max(X[1, :])

        s_padding = max(100.0, 0.05 * (s_max - s_min))
        h_padding = max(50.0, 0.15 * (h_max - h_min))

        plot_wind_field(
            wind_model=wind_model,
            s_bounds=(s_min - s_padding, s_max + s_padding),
            h_bounds=(max(0.0, h_min - h_padding), h_max + h_padding),
            trajectory=solution,
            t=config.problem.time.initial,
            output_path=plots_dir / "wind_field.png",
            show=False,
        )

    # ---------------------------------------------------------
    # Aircraft animation
    # ---------------------------------------------------------

    animation = None

    if config.visualization.animation:
        settings = config.visualization.animation_settings
        objective_name = settings.solution

        if objective_name not in results:
            raise ValueError(
                f"Animation solution '{objective_name}' was not computed. "
                f"Available solutions: {list(results.keys())}"
            )

        playback_speed = max(float(settings.playback_speed), 1e-6)
        interval = max(1, round(settings.interval / playback_speed))

        animation = animate_solution(
            solution=results[objective_name],
            target=config.problem.terminal_state.horizontal_position,
            scale=2.5,
            interval=interval,
            follow=True,
            title=f"Aircraft landing: {objective_name}",
        )

    # Display generated figures and keep the animation alive while shown.
    if (
        config.visualization.plots
        or (
            config.visualization.wind_field
            and wind_model is not None
        )
        or config.visualization.animation
    ):
        plt.show()

    return animation
