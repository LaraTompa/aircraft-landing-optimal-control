# aircraft-landing-optimal-control
Optimal control and uncertainty quantification for aircraft landing under uncertain wind disturbances.

## Setup (for conda)
```bash
conda create --name aircraft-oc python=3.11
conda activate aircraft-oc 
pip install -r requirements.txt
```

## Toy problem

```bash
python examples/toy_problem.py      # Solution
python animator.py                  # Animation
```

### Dependencies
| Package | Purpose |
|---|---|
| [CasADi](https://web.casadi.org/) | Symbolic expressions, automatic differentiation, and nonlinear optimization / optimal control |
| [NumPy](https://numpy.org/) | Numerical arrays and general numerical computations |
| [SciPy](https://scipy.org/) | Scientific computing and supporting numerical methods |
| [Matplotlib](https://matplotlib.org/) | Plotting trajectories, controls, convergence, and other results |
