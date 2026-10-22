# aircraft-landing-optimal-control
Optimal control and uncertainty quantification for aircraft landing under uncertain wind disturbances.

## Setup (for conda)
conda create --name aircraft-oc python=3.11
conda activate aircraft-oc 
pip install -r requirements.txt

## Toy problem

```bash
python examples/toy_problem.py      # Solution
python examples/toy_animator.py     # Animation
```

## State-constrained spline problem

Minimize $\frac12\int_0^1 \ddot y(t)^2\,dt$ subject to $y(t)\le L$, $y(0)=y(1)=0$, $\dot y(0)=1$, $\dot y(1)=-1$.
With $x_1=y$, $x_2=\dot y$, $u=\ddot y$ this is a 2-state, 1-control problem: $\dot x_1=x_2$, $\dot x_2=u$, $J=\frac12\int_0^1 u^2\,dt$.

```bash
python examples/spline_problem.py   # Solve (Euler + RK4), print dynamics analysis, plot
python examples/spline_animator.py  # Animation
python examples/spline_plots.py     # Save report figures to data/ (git-ignored)
```

**Method.** Direct multiple shooting with $N=100$ intervals and piecewise constant controls, solved with CasADi/IPOPT. The bound $x_1\le L$ is enforced at the shooting nodes only. `N_INTERVALS` and `L_BOUND` at the top of `spline_problem.py` are easy to change.

- Forward Euler: $x_{k+1}=\begin{bmatrix}1&h\\0&1\end{bmatrix}x_k+\begin{bmatrix}0\\h\end{bmatrix}u_k$
- RK4 (equals the exact zero-order-hold update here): $x_{k+1}=\begin{bmatrix}1&h\\0&1\end{bmatrix}x_k+\begin{bmatrix}h^2/2\\h\end{bmatrix}u_k$

**Analytical reference.** Euler-Lagrange gives $y''''=0$. Unconstrained: $y=t(1-t)$, $J=2$, maximum $1/4$.

| $L$ | Solution |
|---|---|
| $L\ge 1/4$ | Constraint inactive, $y=t(1-t)$ |
| $1/6\le L<1/4$ | Touches the bound only at $t=1/2$ |
| $L<1/6$ | Flat arc $y=L$ on $[3L,\,1-3L]$, $J=4/(9L)$ |

**Results ($N=100$).**

| $L$ | $J_\text{ana}$ | $J_\text{Euler}$ | $J_\text{RK4}$ | Max error Euler / RK4 |
|---|---|---|---|---|
| 0.25 | 2.0000 | 2.0006 | 2.0000 | 9.8e-4 / 7.0e-6 |
| 0.20 | 2.2400 | 2.2415 | 2.2401 | 1.4e-3 / 1.9e-6 |
| 1/6 | 2.6667 | 2.6693 | 2.6669 | 1.5e-3 / 3.2e-6 |
| 0.10 | 4.4444 | 4.4568 | 4.4457 | 1.6e-3 / 2.4e-5 |
| 0.05 | 8.8889 | 8.9892 | 8.8988 | 1.6e-3 / 1.8e-4 |

RK4 is clearly more accurate than Euler, which omits the $h^2u_k/2$ term. Euler's error halves when $N$ is doubled. Since the bound is only enforced at the nodes, $L=0$ is feasible in the discrete problem but not in the continuous one, so very small $L$ should be treated with care. Report figures are generated locally into `data/` by `spline_plots.py`.

### Dependencies
| Package | Purpose |
|---|---|
| [CasADi](https://web.casadi.org/) | Symbolic expressions, automatic differentiation, and nonlinear optimization / optimal control |
| [NumPy](https://numpy.org/) | Numerical arrays and general numerical computations |
| [SciPy](https://scipy.org/) | Scientific computing and supporting numerical methods |
| [Matplotlib](https://matplotlib.org/) | Plotting trajectories, controls, convergence, and other results |