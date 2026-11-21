# Thesis appendix: spline studies

This folder contains the independent spline optimal-control pipeline used for the thesis appendix. Run commands from the repository root after installing the packages in `requirements.txt`.

## Reproduce the studies

```bash
python -m appendix.spline.solve_spline_problem
python -m appendix.spline.solve_complex_spline_problem
python -m appendix.spline.spline_plots
```

The first script runs the state-constrained spline problem and shows its midpoint-continuation plots. The complex solver studies a fixed arc length and midpoint continuation, saving a verification CSV and PNG plots. `spline_plots` reads that CSV and generates the thesis figures.

Solver outputs and thesis figures go under `data/spline/`, with thesis-ready PNG and PDF files under `data/spline/thesis_figures/`. These outputs are generated locally and are not source inputs.

## Files

- `spline/solve_spline_problem.py`: state-constrained spline problem and midpoint continuation.
- `spline/solve_complex_spline_problem.py`: fixed arc-length variant, continuation, and verification.
- `spline/spline_plots.py`: thesis figures based on the complex-solver verification output.
