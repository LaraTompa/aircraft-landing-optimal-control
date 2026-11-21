# aircraft-landing-optimal-control
Optimal control and uncertainty quantification for aircraft landing under uncertain wind disturbances.

## Setup (for conda)
conda create --name aircraft-oc python=3.11
conda activate aircraft-oc 
pip install -r requirements.txt

## Repository structure

```text
.
├── configs/
│   ├── aircraft/        # Aircraft parameter sets
│   ├── experiments/     # Top-level experiment configurations
│   ├── problems/        # States, constraints, and objectives
│   └── wind/            # Wind-field configurations
├── examples/
│   └── toy_problem.py   # Small standalone optimal-control example
├── appendix/
│   ├── README.md        # Reproduction steps and expected outputs
│   └── spline/
│       ├── solve_spline_problem.py
│       ├── solve_complex_spline_problem.py
│       └── spline_plots.py     # Thesis figures and comparison plots
├── src/
│   ├── aircraft_optimal_control/
│   │   ├── config/      # YAML loading and validation
│   │   ├── disturbance/ # Wind-field models
│   │   ├── models/      # Aircraft dynamics and aerodynamics
│   │   ├── numerical_integration/
│   │   ├── optimal_control/
│   │   └── visualization/
│   └── experiments/     # OCP assembly, solving, and result writing
├── data/                # Generated spline-example outputs
├── results/             # Generated experiment outputs (git-ignored)
├── land_airplane.py     # Aircraft-landing entry point
└── requirements.txt
```

Run commands from the repository root. The aircraft experiment combines the YAML files referenced by `configs/experiments/baseline_pointmass_simple.yaml` and writes outputs to `results/<experiment-name>/`. Thesis appendix commands and their generated files are documented in [appendix/README.md](appendix/README.md); spline outputs go under `data/spline/`.

## Toy problem

```bash
python examples/toy_problem.py      # Solve, plot, and animate the toy problem
```

## State-constrained spline problem

Minimize $\frac12\int_0^1 \ddot y(t)^2\,dt$ subject to $y(t)\le L$, $y(0)=y(1)=0$, $\dot y(0)=1$, $\dot y(1)=-1$.
With $x_1=y$, $x_2=\dot y$, $u=\ddot y$ this is a 2-state, 1-control problem: $\dot x_1=x_2$, $\dot x_2=u$, $J=\frac12\int_0^1 u^2\,dt$.

```bash
python -m appendix.spline.solve_spline_problem  # Solve and show direct and homotopy comparisons
```

Run commands from the repository root. See [appendix/README.md](appendix/README.md) for the complex solver and figure-generation workflow. Generated spline outputs are written under `data/spline/`.


## Airplane model

Side-view Airbus A320 (true proportions in metres, gear down, flaps extended) for animating the landing trajectory.

```bash
python land_airplane.py       # Solve both objectives, save results, and show plots/animation
```

Use `animate_airplane(x, y, theta, target=None, scale=1.0)` with the main wheel contact point `(x, y)` in metres and pitch angle `theta` [rad, nose up > 0]; use `scale` > 1 to enlarge the plane for visibility; `make_airplane` / `set_pose` give lower-level access to the patches.

The landing problem uses a fixed 65-second duration, an initial airspeed of 95 m/s and a touchdown airspeed of 80 m/s. Thrust and lift coefficient are piecewise-constant controls; the lift coefficient is bounded between 0.8 and 1.8 and the airspeed between 55 and 95 m/s. These parameters and bounds are illustrative, not validated aircraft limits. State constraints are enforced at the discretization nodes, not continuously between them.

The landing script reads `configs/experiments/baseline_pointmass_simple.yaml`, solves the squared-thrust and linear-thrust objectives on the same 100-interval problem, and warm-starts the second solve from the first. It writes `solutions.npz`, `metadata.json`, and three PNGs under `results/baseline_point_mass_gaussian/`; the PNGs are named `squared_thrust_solution.png`, `linear_thrust_solution.png`, and `wind_field.png`. It also opens the solution plots and a following-view animation at approximately 5x real time. The animation uses the optimized flight-path angle as its pitch approximation. The point-mass model does not calculate aircraft pitch or wheel position, so these are visualization assumptions, not optimized aircraft attitudes or validated gear geometry.

### Dependencies
| Package | Purpose |
|---|---|
| [CasADi](https://web.casadi.org/) | Symbolic expressions, automatic differentiation, and nonlinear optimization / optimal control |
| [NumPy](https://numpy.org/) | Numerical arrays and general numerical computations |
| [SciPy](https://scipy.org/) | Scientific computing and supporting numerical methods |
| [Matplotlib](https://matplotlib.org/) | Plotting trajectories, controls, convergence, and other results |
| [PyYAML](https://pyyaml.org/) | Reading aircraft, wind, problem, and experiment configuration files |
