
import casadi as ca
import numpy as np

from aircraft_optimal_control.numerical_integration.solvers import configure_solver
from aircraft_optimal_control.numerical_integration.integrators import get_integrator

from aircraft_optimal_control.optimal_control.objectives import (
    build_objectives,
)

from aircraft_optimal_control.optimal_control.constraints import (
    add_landing_constraints,
    add_minimax_clearance_constraint,
)

from aircraft_optimal_control.optimal_control.transcription import (
    add_multiple_shooting_constraints,
    add_hermite_simpson_constraints,
)


class LandingOptimalControlProblem:
    """
    Aircraft optimal control problem supporting the legacy landing and
    paper minimax abort-landing formulations.

    State and control dimensions and semantic indices come from the selected
    aircraft model, so legacy and angle-of-attack models share this builder.
    """

    def __init__(
        self,
        aircraft_model,
        problem_config,
        integrator,
        objective_names,
        solver_name="ipopt",
        solver_options=None,
        transcription_method="multiple_shooting",
    ):
        self.aircraft_model = aircraft_model
        self.aircraft_state_dim = aircraft_model.STATE_DIM
        self.STATE_INDICES = dict(aircraft_model.STATE_INDICES)
        self.CONTROL_INDICES = aircraft_model.CONTROL_INDICES
        self.config = problem_config
        self.integrator = integrator

        self.objective_names = list(objective_names)
        self.transcription_method = transcription_method.lower()

        if len(self.objective_names) not in {1, 2}:
            raise ValueError(
                "Configure one objective or a two-objective comparison."
            )

        for name in self.objective_names:
            if name not in self.config.objectives:
                raise ValueError(f"Unknown objective: {name}")

        selected_definitions = [
            self.config.objectives[name] for name in self.objective_names
        ]
        minimax_definitions = [
            definition for definition in selected_definitions
            if definition.type == "minimax_clearance"
        ]
        if minimax_definitions and (
            len(self.objective_names) != 1 or len(minimax_definitions) != 1
        ):
            raise ValueError("The minimax clearance objective must be selected alone.")
        self.minimax_definition = minimax_definitions[0] if minimax_definitions else None
        self.has_zeta = self.minimax_definition is not None
        if self.has_zeta:
            if "angle_of_attack" not in self.STATE_INDICES:
                raise ValueError("Minimax clearance requires the paper point-mass model.")
            self.STATE_INDICES["zeta"] = self.aircraft_state_dim
        self.state_dim = self.aircraft_state_dim + int(self.has_zeta)

        if self.transcription_method not in {
            "multiple_shooting",
            "collocation",
        }:
            raise ValueError(
                f"Unsupported transcription method: "
                f"{self.transcription_method}"
            )

        if (
            self.transcription_method == "multiple_shooting"
            and self.integrator is None
        ):
            raise ValueError(
                "Multiple shooting requires an integrator."
            )

        self.solver_name = solver_name
        self.solver_options = dict(solver_options or {})

        self._build()

    # ========================================================
    # Build optimization problem
    # ========================================================

    def _build(self):

        self.opti = ca.Opti()

        # Time discretization
        self.t0 = self.config.time.initial
        self.tf = self.config.time.final
        self.N = self.config.time.intervals
        self.dt = self.config.time.dt

        # Decision variables
        self.X = self.opti.variable(self.state_dim, self.N + 1)
        self.U = self.opti.variable(self.aircraft_model.CONTROL_DIM, self.N)

        # ----------------------------------------------------
        # Dynamic constraints
        # ----------------------------------------------------

        if self.transcription_method == "multiple_shooting":

            add_multiple_shooting_constraints(
                opti=self.opti,
                X=self.X,
                U=self.U,
                dynamics=self._dynamics,
                integrator=self.integrator,
                dt=self.dt,
                N=self.N,
                t0=self.t0,
            )

        elif self.transcription_method == "collocation":

            add_hermite_simpson_constraints(
                opti=self.opti,
                X=self.X,
                U=self.U,
                dynamics=self._dynamics,
                dt=self.dt,
                N=self.N,
                t0=self.t0,
            )

        # ----------------------------------------------------
        # Objective functions
        # ----------------------------------------------------

        selected_definitions = {
            name: self.config.objectives[name]
            for name in self.objective_names
        }

        if self.has_zeta:
            self.objectives = {
                self.objective_names[0]: (
                    self.minimax_definition.theta
                    * self.X[self.STATE_INDICES["zeta"], -1]
                )
            }
        else:
            self.objectives = build_objectives(
                U=self.U,
                dt=self.dt,
                objective_definitions=selected_definitions,
                control_indices=self.CONTROL_INDICES,
            )

        # ----------------------------------------------------
        # Boundary, state and control constraints
        # ----------------------------------------------------

        state_constraints = self.config.state_constraints
        control_constraints = self.config.control_constraints

        state_bounds = {
            "altitude": (
                state_constraints.altitude_min,
                state_constraints.altitude_max,
            ),
            "airspeed": (
                state_constraints.airspeed_min,
                state_constraints.airspeed_max,
            ),
            "flight_path_angle": (
                state_constraints.flight_path_angle_min,
                state_constraints.flight_path_angle_max,
            ),
        }
        if (
            state_constraints.horizontal_position_min is not None
            or state_constraints.horizontal_position_max is not None
        ):
            state_bounds["horizontal_position"] = (
                state_constraints.horizontal_position_min,
                state_constraints.horizontal_position_max,
            )
        if "angle_of_attack" in self.STATE_INDICES:
            alpha_max = state_constraints.angle_of_attack_max
            if alpha_max is None:
                alpha_max = self.aircraft_model.aircraft.aerodynamics.alpha_max
            state_bounds["angle_of_attack"] = (
                state_constraints.angle_of_attack_min,
                alpha_max,
            )
        if "angle_of_attack_rate" in self.CONTROL_INDICES:
            control_bounds = {
                "angle_of_attack_rate": (
                    control_constraints.angle_of_attack_rate_min,
                    control_constraints.angle_of_attack_rate_max,
                )
            }
        else:
            control_bounds = {
                "thrust": (control_constraints.thrust_min, control_constraints.thrust_max),
                "lift_coefficient": (
                    control_constraints.lift_coefficient_min,
                    control_constraints.lift_coefficient_max,
                ),
            }

        add_landing_constraints(
            opti=self.opti,
            X=self.X,
            U=self.U,
            initial_state=self._boundary_values(self.config.initial_state),
            terminal_state=self._boundary_values(self.config.terminal_state),
            state_indices=self.STATE_INDICES,
            control_indices=self.CONTROL_INDICES,
            state_bounds=state_bounds,
            control_bounds=control_bounds,
        )

        if self.has_zeta:
            add_minimax_clearance_constraint(
                opti=self.opti,
                X=self.X,
                altitude_index=self.STATE_INDICES["altitude"],
                zeta_index=self.STATE_INDICES["zeta"],
                runway_height=self.minimax_definition.runway_height,
            )
            self._add_minimax_intermediate_constraints()

        # ----------------------------------------------------
        # Objective selection
        # ----------------------------------------------------

        if len(self.objective_names) == 1:
            self.opti.minimize(self.objectives[self.objective_names[0]])
        else:
            self.alpha = self.opti.parameter()
            first_objective = self.objectives[self.objective_names[0]]
            second_objective = self.objectives[self.objective_names[1]]
            self.opti.minimize(
                self.alpha * first_objective
                + (1.0 - self.alpha) * second_objective
            )

        # ----------------------------------------------------
        # Initial guess
        # ----------------------------------------------------

        self.set_initial_guess()

        # ----------------------------------------------------
        # Solver
        # ----------------------------------------------------

        configure_solver(
            self.opti,
            self.solver_name,
            self.solver_options,
        )

    # ========================================================
    # State conversion
    # ========================================================

    def _boundary_values(self, state):
        values = {
            "horizontal_position": state.horizontal_position,
            "altitude": state.altitude,
            "airspeed": state.airspeed,
            "flight_path_angle": state.flight_path_angle,
        }
        if "angle_of_attack" in self.STATE_INDICES:
            values["angle_of_attack"] = state.angle_of_attack
        if self.has_zeta:
            values["zeta"] = None
        return {
            self.STATE_INDICES[name]: value
            for name, value in values.items()
        }

    def _dynamics(self, x, u, t):
        aircraft_rhs = self.aircraft_model.dynamics(
            x[:self.aircraft_state_dim], u, t
        )
        if self.has_zeta:
            return ca.vertcat(aircraft_rhs, 0.0)
        return aircraft_rhs

    # ========================================================
    # Initial guess
    # ========================================================

    def set_initial_guess(self):

        if self.has_zeta:
            self._set_minimax_initial_guess()
            return

        initial = self._boundary_values(self.config.initial_state)
        terminal = self._boundary_values(self.config.terminal_state)

        for name, i in self.STATE_INDICES.items():
            x0 = initial[i]
            xf = terminal[i]
            if name == "zeta":
                guess_value = (
                    self.minimax_definition.runway_height
                    - self.config.initial_state.altitude
                )
                guess = np.full(self.N + 1, guess_value)
            else:
                if x0 is None and xf is None:
                    guess = np.zeros(self.N + 1)
                else:
                    if x0 is None:
                        x0 = xf
                    if xf is None:
                        xf = x0
                    guess = np.linspace(float(x0), float(xf), self.N + 1)

            self.opti.set_initial(
                self.X[i, :],
                guess,
            )

        initial_guess = self.config.initial_guess

        guess_values = {
            "thrust": initial_guess.thrust,
            "lift_coefficient": initial_guess.lift_coefficient,
            "angle_of_attack_rate": initial_guess.angle_of_attack_rate,
        }
        for name, index in self.CONTROL_INDICES.items():
            self.opti.set_initial(self.U[index, :], guess_values[name])

    def _set_minimax_initial_guess(self):
        """Propagate the initial state under the configured control guess.

        A dynamically consistent path is a much better starting point for
        the minimax multiple-shooting NLP than a constant-state guess.
        """
        initial = self.config.initial_state
        x0 = np.zeros(self.aircraft_state_dim)
        boundary = {
            "horizontal_position": initial.horizontal_position,
            "altitude": initial.altitude,
            "airspeed": initial.airspeed,
            "flight_path_angle": initial.flight_path_angle,
            "angle_of_attack": initial.angle_of_attack,
        }
        for name, index in self.aircraft_model.STATE_INDICES.items():
            x0[index] = boundary[name]

        alpha_rate = self.config.initial_guess.angle_of_attack_rate
        control = ca.DM([alpha_rate])
        guess_integrator = self.integrator or get_integrator("RK4")
        x = ca.DM(np.concatenate((x0, [0.0])))
        path = [np.asarray(x, dtype=float).reshape(-1)]
        interior_altitudes = []
        for k in range(self.N):
            for j in range(1, 8):
                interior = guess_integrator(
                    self._dynamics,
                    x,
                    control,
                    self.t0 + k * self.dt,
                    (j / 8.0) * self.dt,
                )
                interior_altitudes.append(
                    float(interior[self.STATE_INDICES["altitude"]])
                )
            x = guess_integrator(
                self._dynamics,
                x,
                control,
                self.t0 + k * self.dt,
                self.dt,
            )
            path.append(np.asarray(x, dtype=float).reshape(-1))

        path = np.asarray(path).T
        altitude_index = self.STATE_INDICES["altitude"]
        zeta_index = self.STATE_INDICES["zeta"]
        zeta_guess = float(np.max(
            self.minimax_definition.runway_height - np.concatenate((
                path[altitude_index, :], np.asarray(interior_altitudes)
            ))
        ))
        path[zeta_index, :] = zeta_guess

        self.opti.set_initial(self.X, path)
        self.opti.set_initial(self.U, alpha_rate)

    def _add_minimax_intermediate_constraints(self):
        """Enforce clearance at eight uniform points per shooting interval."""
        altitude_index = self.STATE_INDICES["altitude"]
        zeta_index = self.STATE_INDICES["zeta"]
        runway_height = self.minimax_definition.runway_height

        for k in range(self.N):
            x_k = self.X[:, k]
            u_k = self.U[:, k]
            t_k = self.t0 + k * self.dt

            if self.transcription_method == "multiple_shooting":
                for j in range(1, 8):
                    sample = self.integrator(
                        self._dynamics,
                        x_k,
                        u_k,
                        t_k,
                        (j / 8.0) * self.dt,
                    )
                    self.opti.subject_to(
                        runway_height
                        - sample[altitude_index]
                        - sample[zeta_index]
                        <= 0
                    )
            else:
                f_k = self._dynamics(x_k, u_k, t_k)
                f_next = self._dynamics(
                    self.X[:, k + 1], u_k, t_k + self.dt
                )
                for j in range(1, 8):
                    tau = j / 8.0
                    tau2, tau3 = tau * tau, tau * tau * tau
                    sample = (
                        (2 * tau3 - 3 * tau2 + 1) * x_k
                        + (tau3 - 2 * tau2 + tau) * self.dt * f_k
                        + (-2 * tau3 + 3 * tau2) * self.X[:, k + 1]
                        + (tau3 - tau2) * self.dt * f_next
                    )
                    self.opti.subject_to(
                        runway_height
                        - sample[altitude_index]
                        - sample[zeta_index]
                        <= 0
                    )

    # ========================================================
    # Objective selection
    # ========================================================

    def select_objective(self, objective_name):

        if len(self.objective_names) == 1:
            if objective_name != self.objective_names[0]:
                raise ValueError(f"Objective '{objective_name}' is not configured.")
            return

        if objective_name == self.objective_names[0]:
            self.opti.set_value(self.alpha, 1.0)

        elif objective_name == self.objective_names[1]:
            self.opti.set_value(self.alpha, 0.0)

        else:
            raise ValueError(
                f"Objective '{objective_name}' is not configured."
            )

    # ========================================================
    # Warm start
    # ========================================================

    def set_warm_start(self, X_solution, U_solution):

        self.opti.set_initial(self.X, X_solution)
        self.opti.set_initial(self.U, U_solution)

    # ========================================================
    # Solve
    # ========================================================

    def solve(self, objective_name):

        self.select_objective(objective_name)

        return self.opti.solve()

    # ========================================================
    # Extract numerical solution
    # ========================================================

    def extract_solution(self, solution):

        return {
            "X": solution.value(self.X),

            "U": np.asarray(
                solution.value(self.U)
            ),

            "objectives": {
                name: solution.value(expression)
                for name, expression in self.objectives.items()
            },

            "time": np.linspace(
                self.t0,
                self.tf,
                self.N + 1,
            ),

            "control_time": (
                self.t0
                + np.arange(self.N) * self.dt
            ),
        }
