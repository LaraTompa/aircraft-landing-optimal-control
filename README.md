# aircraft-landing-optimal-control
Optimal control and uncertainty quantification for aircraft landing under uncertain wind disturbances.

## Setup (for conda)
conda create --name aircraft-oc python=3.11
conda activate aircraft-oc 
pip install -r requirements.txt

## Toy problem

```bash
python examples/toy_problem.py      # Solve, plot, and animate the toy problem
```

## State-constrained spline problem

Minimize $\frac12\int_0^1 \ddot y(t)^2\,dt$ subject to $y(t)\le L$, $y(0)=y(1)=0$, $\dot y(0)=1$, $\dot y(1)=-1$.
With $x_1=y$, $x_2=\dot y$, $u=\ddot y$ this is a 2-state, 1-control problem: $\dot x_1=x_2$, $\dot x_2=u$, $J=\frac12\int_0^1 u^2\,dt$.

```bash
python examples/solve_spline_problem.py  # Solve, print results and show the comparison plot
python -m examples.spline_problem_scripts.spline_animator  # Show the animation
python -m examples.spline_problem_scripts.spline_plots  # Save comparison figures
python -m examples.spline_problem_scripts.spline_initializations  # Compare initial guesses
```

Run these commands from the repository root. The plotting and initialization scripts create
`data/spline/` as needed and write their PNG figures and initialization text results there.
This generated directory is git-ignored; the scripts do not generate a LaTeX report.
The solver and animator display their results interactively and do not write files.
The helper scripts can also be run directly, for example
`python examples/spline_problem_scripts/spline_plots.py`.

**Methods.** Six formulations with $N=100$ intervals, solved with CasADi (IPOPT with tolerance $10^{-12}$ and bound relaxation off, or `sqpmethod` for the QP; every solve checks the solver status and `solve_spline` returns its wall-clock `time`). The bound $x_1\le L$ is enforced at the nodes only. `N_INTERVALS` and `L_BOUND` at the top of `solve_spline_problem.py` are easy to change; $N\ge2$ is required.

- *Multiple shooting + forward Euler* (piecewise constant $u$): $x_{k+1}=\begin{bmatrix}1&h\\0&1\end{bmatrix}x_k+\begin{bmatrix}0\\h\end{bmatrix}u_k$. **Caveat:** Euler solves a perturbed problem. Its optimal control misses the true terminal state by $h$, and since $x_1^1=h$ for every control the problem is infeasible for $L<h$.
- *Multiple shooting + RK4* (piecewise constant $u$; equals the exact zero-order-hold update here): $x_{k+1}=\begin{bmatrix}1&h\\0&1\end{bmatrix}x_k+\begin{bmatrix}h^2/2\\h\end{bmatrix}u_k$
- *Multiple shooting + RK5* (fixed-step Dormand-Prince, 6 stages, piecewise constant $u$): the same exact update as RK4, since any Runge-Kutta method of order $\ge2$ is exact for $\dot x=(x_2,u)$ with constant $u$.
- *Direct collocation, Hermite-Simpson* (continuous piecewise linear $u$, midpoint states as extra variables): $x_c^k=\tfrac12(x^k+x^{k+1})+\tfrac h8(f^k-f^{k+1})$ and $x^{k+1}-x^k=\tfrac h6\big(f^k+4f(x_c^k,u_c^k)+f^{k+1}\big)$, with $u_c^k=\tfrac12(u^k+u^{k+1})$. A piecewise linear control is used because with piecewise constant $u$ Hermite-Simpson and RK4 are identical for these dynamics.
- *Condensed QP solved with SQP*: with the exact dynamics all states are affine in $U=(u_0,\dots,u_{N-1})^\top$, $X=\Phi x_0+\Gamma U$, giving the convex QP $\min_U \tfrac12 U^\top HU$ ($H=hI$) s.t. $AU=b$ (terminal condition) and $GU\le g$ ($x_1[k]\le L$). Solved with CasADi's `sqpmethod` (qpOASES); it converges in one iteration. It is the same discrete problem as RK4 shooting.
- *Single shooting with an ODE solver* (cost discretized, dynamics not): piecewise constant $u$, cost $\tfrac h2\sum u_k^2$, and the dynamics $\dot x=(x_2,u)$ are integrated interval by interval with CVODES (tolerance $10^{-12}$) instead of being constraints; only the $N$ controls are decision variables, with the terminal condition and $x_1[k]\le L$ as constraints, solved with IPOPT (limited-memory Hessian, since CVODES has no second-order sensitivities). Same discrete problem as RK4/QP.

In the plots each method has its own colour, line style and marker (Euler: blue solid squares, RK4: orange dashed circles, RK5: brown dashed triangles, collocation: green dash-dot triangles, QP/SQP: purple dotted crosses, ODE shooting: red dash-dot diamonds). The QP, ODE and RK5 curves lie on top of RK4 by construction.

**Analytical reference.** Euler-Lagrange gives $y''''=0$. Unconstrained: $y=t(1-t)$, $J=2$, maximum $1/4$.

| $L$ | Solution |
|---|---|
| $L\ge 1/4$ | Constraint inactive, $y=t(1-t)$ |
| $1/6\le L<1/4$ | Touches the bound only at $t=1/2$ |
| $L<1/6$ | Flat arc $y=L$ on $[3L,\,1-3L]$, $J=4/(9L)$ |

**Results ($N=100$).** $J_\text{QP}$, $J_\text{ODE}$ and $J_\text{RK5}$ equal $J_\text{RK4}$ to the digits shown, and the RK5 error equals the RK4 error. Times are the median of three runs of the complete `solve_spline` call (seconds; indicative only).

| $L$ | $J_\text{ana}$ | $J_\text{Euler}$ | $J_\text{RK4}$ | $J_\text{HS}$ | Max error Euler / RK4 (= RK5) / HS / QP / ODE | Time Euler / RK4 / RK5 / HS / QP / ODE |
|---|---|---|---|---|---|---|
| 0.25 | 2.0000 | 2.0006 | 2.0000 | 2.0000 | 9.8e-4 / 5.7e-8† / 5.7e-8† / 7.8e-16 / 5.6e-8† | 0.050 / 0.100 / 0.228 / 0.121 / 0.051 / 0.423 |
| 0.20 | 2.2400 | 2.2415 | 2.2401 | 2.2400 | 1.4e-3 / 1.9e-6 / 1.7e-13 / 1.9e-6 / 1.9e-6 | 0.052 / 0.109 / 0.248 / 0.122 / 0.065 / 0.384 |
| 1/6 | 2.6667 | 2.6693 | 2.6669 | 2.6667 | 1.5e-3 / 3.2e-6 / 1.1e-12 / 3.2e-6 / 3.2e-6 | 0.064 / 0.100 / 0.256 / 0.127 / 0.075 / 0.377 |
| 0.113 | 3.9331 | 3.9417 | 3.9340 | 3.9331 | 1.5e-3 / 3.1e-5 / 2.1e-5 / 3.1e-5 / 3.1e-5 | 0.069 / 0.109 / 0.257 / 0.131 / 0.076 / 0.410 |
| 0.10 | 4.4444 | 4.4568 | 4.4457 | 4.4444 | 1.6e-3 / 2.0e-5 / 9.9e-8† / 2.0e-5 / 2.0e-5 | 0.070 / 0.109 / 0.258 / 0.148 / 0.079 / 0.477 |
| 0.05 | 8.8889 | 8.9892 | 8.8988 | 8.8889 | 1.6e-3 / 1.8e-4 / 2.6e-7† / 1.8e-4 / 1.8e-4 | 0.075 / 0.114 / 0.275 / 0.190 / 0.089 / 0.453 |

† IPOPT floor (weakly active constraints), not discretization error: solving the same problems exactly as QPs gives $\le10^{-12}$ (HS) and $8\cdot10^{-16}$ (RK4/QP at $L=0.25$).

**What was verified** (sweep over 26 values of $L\in[0.011,0.5]$, $N=25$–$400$, all solves converged):

- The analytical solution satisfies the boundary conditions, $y\le L$, has continuous $y''$, and its cost matches a finite-difference evaluation; Richardson extrapolation of the QP cost converges to it ($6\cdot10^{-10}$ at $L=0.2$, $10^{-8}$ at $L=0.1$).
- The QP solutions satisfy the KKT conditions and agree with DAQP to $3\cdot10^{-13}$ ($N=100$). RK4, QP and HS are consistent with exact propagation of the ODE to $10^{-15}$; Euler is not (terminal miss $h$).
- Orders: Euler 1, RK4 = QP 2 (their errors are identical for $L<1/4$), HS 2 in general.
- Initialization: all six methods were started from 9 very different guesses (zeros, infeasible, far above the bound, random) at $L=0.2$ and $0.1$. All 108 solves converged to the same solution for each method (objectives agree to $\le5\cdot10^{-11}$ across initializations, trajectories to $\le2\cdot10^{-8}$, the IPOPT floor); the guess only changes the IPOPT iteration count by a few (e.g. RK4: 17 vs. 19–20 at $L=0.1$), and the QP/SQP always needs 1 iteration. `solve_spline(..., guess=f)` takes `f(t) -> (y, y', y'')`.

**Findings.**

- Ranking: Euler < RK4 = RK5 = QP = ODE ≲ Hermite-Simpson. HS is better than RK4/QP by only a factor 1.3–1.5 in general (e.g. $L=0.113$), but it is exact ($\sim10^{-13}$) when the kinks of the optimal $u=y''$ (at $t=3L$, $1-3L$) lie on grid nodes (all $1/6\le L<1/4$ for even $N$; $L<1/6$ when $3LN$ is an integer, e.g. $L=0.1$, $N\ge50$), because its piecewise linear control then represents the optimal $u$ exactly. For $L=0.1$ and $N=25$ the kinks are not nodes and HS is only slightly better than RK4.
- The condensed QP is the same discrete problem as RK4 and converges in one SQP iteration, but it does not scale: with CasADi `sqpmethod`/qpOASES it is the fastest method at $N\le100$ (0.004–0.08 s) but takes 0.72 s at $N=200$ and 8.5 s at $N=400$, versus 0.51 s (RK4) and 0.74 s (collocation) at $N=400$ (indicative only).
- RK5 (fixed-step Dormand-Prince) gives exactly the same discrete problem and solution as RK4 (differences $\sim10^{-16}$): for constant $u$ every Runge-Kutta method of order $\ge2$ is exact here, so the higher order brings no accuracy gain, only cost (≈2.3× the time of RK4: 0.27 s vs 0.11 s at $N=100$, 1.2 s vs 0.5 s at $N=400$).
- ODE-solver shooting gives the same solution as RK4/QP but is the slowest method for $N\le200$ (0.38–0.48 s at $N=100$, 3–5× RK4; 5.8 s at $N=400$), since every evaluation integrates all intervals with sensitivities and IPOPT uses a limited-memory Hessian.
- IPOPT's defaults distorted earlier numbers (tolerance $10^{-8}$ made RK4/HS errors stall near $10^{-5}$; bound relaxation $10^{-8}$ let $x_1$ exceed $L$); both are changed.
- The bound is only enforced at the nodes: the true trajectory can exceed $L$ between nodes by $\le10^{-6}$ for $L\ge0.05$ (up to $3\cdot10^{-5}$ at $L\approx0.01$). For $L=0$ the discrete problem is feasible for RK4, QP and HS (with different, meaningless costs) but infeasible for Euler, while the continuous problem is infeasible, so very small $L$ should be treated with care.


## Airplane model

Side-view Airbus A320 (true proportions in metres, gear down, flaps extended) for animating the landing trajectory.

```bash
python airplane_animator.py   # Demo on a placeholder flare path
python land_airplane.py       # Solve the landing problem and animate its trajectory
```

Use `animate_airplane(x, y, theta, target=None, scale=1.0)` with the main wheel contact point `(x, y)` in metres and pitch angle `theta` [rad, nose up > 0]; use `scale` > 1 to enlarge the plane for visibility; `make_airplane` / `set_pose` give lower-level access to the patches.

The landing script opens both the solution plots and a following-view animation at approximately 5x real time. It linearly interpolates the optimized states between nodes. The point-mass model does not calculate pitch or wheel position: the animation treats its position as the main-wheel contact point and assumes pitch equals flight-path angle plus a constant 5-degree angle of attack. These are visualization assumptions, not optimized aircraft attitudes or validated gear geometry.

### Dependencies
| Package | Purpose |
|---|---|
| [CasADi](https://web.casadi.org/) | Symbolic expressions, automatic differentiation, and nonlinear optimization / optimal control |
| [NumPy](https://numpy.org/) | Numerical arrays and general numerical computations |
| [SciPy](https://scipy.org/) | Scientific computing and supporting numerical methods |
| [Matplotlib](https://matplotlib.org/) | Plotting trajectories, controls, convergence, and other results |