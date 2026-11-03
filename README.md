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
python examples/spline_problem.py   # Solve (Euler, RK4, Hermite-Simpson, QP/SQP), print dynamics analysis, plot
python examples/spline_animator.py  # Animation
python examples/spline_plots.py     # Save report figures to data/ (git-ignored)
```

**Methods.** Four formulations with $N=100$ intervals, solved with CasADi (IPOPT with tolerance $10^{-12}$ and bound relaxation off, or `sqpmethod` for the QP; every solve checks the solver status). The bound $x_1\le L$ is enforced at the nodes only. `N_INTERVALS` and `L_BOUND` at the top of `spline_problem.py` are easy to change; $N\ge2$ is required.

- *Multiple shooting + forward Euler* (piecewise constant $u$): $x_{k+1}=\begin{bmatrix}1&h\\0&1\end{bmatrix}x_k+\begin{bmatrix}0\\h\end{bmatrix}u_k$. **Caveat:** Euler solves a perturbed problem. Its optimal control misses the true terminal state by $h$, and since $x_1^1=h$ for every control the problem is infeasible for $L<h$.
- *Multiple shooting + RK4* (piecewise constant $u$; equals the exact zero-order-hold update here): $x_{k+1}=\begin{bmatrix}1&h\\0&1\end{bmatrix}x_k+\begin{bmatrix}h^2/2\\h\end{bmatrix}u_k$
- *Direct collocation, Hermite-Simpson* (continuous piecewise linear $u$, midpoint states as extra variables): $x_c^k=\tfrac12(x^k+x^{k+1})+\tfrac h8(f^k-f^{k+1})$ and $x^{k+1}-x^k=\tfrac h6\big(f^k+4f(x_c^k,u_c^k)+f^{k+1}\big)$, with $u_c^k=\tfrac12(u^k+u^{k+1})$. A piecewise linear control is used because with piecewise constant $u$ Hermite-Simpson and RK4 are identical for these dynamics.
- *Condensed QP solved with SQP*: with the exact dynamics all states are affine in $U=(u_0,\dots,u_{N-1})^\top$, $X=\Phi x_0+\Gamma U$, giving the convex QP $\min_U \tfrac12 U^\top HU$ ($H=hI$) s.t. $AU=b$ (terminal condition) and $GU\le g$ ($x_1[k]\le L$). Solved with CasADi's `sqpmethod` (qpOASES); it converges in one iteration. It is the same discrete problem as RK4 shooting.

In the plots each method has its own colour, line style and marker (Euler: blue solid squares, RK4: orange dashed circles, collocation: green dash-dot triangles, QP/SQP: purple dotted crosses). The QP curve lies on top of RK4 by construction.

**Analytical reference.** Euler-Lagrange gives $y''''=0$. Unconstrained: $y=t(1-t)$, $J=2$, maximum $1/4$.

| $L$ | Solution |
|---|---|
| $L\ge 1/4$ | Constraint inactive, $y=t(1-t)$ |
| $1/6\le L<1/4$ | Touches the bound only at $t=1/2$ |
| $L<1/6$ | Flat arc $y=L$ on $[3L,\,1-3L]$, $J=4/(9L)$ |

**Results ($N=100$).** $J_\text{QP}$ equals $J_\text{RK4}$ to the digits shown.

| $L$ | $J_\text{ana}$ | $J_\text{Euler}$ | $J_\text{RK4}$ | $J_\text{HS}$ | Max error Euler / RK4 / HS / QP |
|---|---|---|---|---|---|
| 0.25 | 2.0000 | 2.0006 | 2.0000 | 2.0000 | 9.8e-4 / 5.7e-8† / 5.7e-8† / 7.8e-16 |
| 0.20 | 2.2400 | 2.2415 | 2.2401 | 2.2400 | 1.4e-3 / 1.9e-6 / 1.7e-13 / 1.9e-6 |
| 1/6 | 2.6667 | 2.6693 | 2.6669 | 2.6667 | 1.5e-3 / 3.2e-6 / 1.1e-12 / 3.2e-6 |
| 0.113 | 3.9331 | 3.9417 | 3.9340 | 3.9331 | 1.5e-3 / 3.1e-5 / 2.1e-5 / 3.1e-5 |
| 0.10 | 4.4444 | 4.4568 | 4.4457 | 4.4444 | 1.6e-3 / 2.0e-5 / 9.9e-8† / 2.0e-5 |
| 0.05 | 8.8889 | 8.9892 | 8.8988 | 8.8889 | 1.6e-3 / 1.8e-4 / 2.6e-7† / 1.8e-4 |

† IPOPT floor (weakly active constraints), not discretization error: solving the same problems exactly as QPs gives $\le10^{-12}$ (HS) and $8\cdot10^{-16}$ (RK4/QP at $L=0.25$).

**What was verified** (sweep over 26 values of $L\in[0.011,0.5]$, $N=25$–$400$, all solves converged):

- The analytical solution satisfies the boundary conditions, $y\le L$, has continuous $y''$, and its cost matches a finite-difference evaluation; Richardson extrapolation of the QP cost converges to it ($6\cdot10^{-10}$ at $L=0.2$, $10^{-8}$ at $L=0.1$).
- The QP solutions satisfy the KKT conditions and agree with DAQP to $3\cdot10^{-13}$ ($N=100$). RK4, QP and HS are consistent with exact propagation of the ODE to $10^{-15}$; Euler is not (terminal miss $h$).
- Orders: Euler 1, RK4 = QP 2 (their errors are identical for $L<1/4$), HS 2 in general.

**Findings.**

- Ranking: Euler < RK4 = QP ≲ Hermite-Simpson. HS is better than RK4/QP by only a factor 1.3–1.5 in general (e.g. $L=0.113$), but it is exact ($\sim10^{-13}$) when the kinks of the optimal $u=y''$ (at $t=3L$, $1-3L$) lie on grid nodes (all $1/6\le L<1/4$ for even $N$; $L<1/6$ when $3LN$ is an integer, e.g. $L=0.1$, $N\ge50$), because its piecewise linear control then represents the optimal $u$ exactly. For $L=0.1$ and $N=25$ the kinks are not nodes and HS is only slightly better than RK4.
- The condensed QP is the same discrete problem as RK4 and converges in one SQP iteration, but it does not scale: with CasADi `sqpmethod`/qpOASES it takes about 0.3 s at $N=100$ and 26–34 s at $N=400$, versus 1.6–2.6 s for IPOPT with RK4 or collocation (single wall-clock runs, indicative only).
- IPOPT's defaults distorted earlier numbers (tolerance $10^{-8}$ made RK4/HS errors stall near $10^{-5}$; bound relaxation $10^{-8}$ let $x_1$ exceed $L$); both are changed.
- The bound is only enforced at the nodes: the true trajectory can exceed $L$ between nodes by $\le10^{-6}$ for $L\ge0.05$ (up to $3\cdot10^{-5}$ at $L\approx0.01$). For $L=0$ the discrete problem is feasible for RK4, QP and HS (with different, meaningless costs) but infeasible for Euler, while the continuous problem is infeasible, so very small $L$ should be treated with care.

Report figures are generated locally into `data/` by `spline_plots.py`.

### Dependencies
| Package | Purpose |
|---|---|
| [CasADi](https://web.casadi.org/) | Symbolic expressions, automatic differentiation, and nonlinear optimization / optimal control |
| [NumPy](https://numpy.org/) | Numerical arrays and general numerical computations |
| [SciPy](https://scipy.org/) | Scientific computing and supporting numerical methods |
| [Matplotlib](https://matplotlib.org/) | Plotting trajectories, controls, convergence, and other results |