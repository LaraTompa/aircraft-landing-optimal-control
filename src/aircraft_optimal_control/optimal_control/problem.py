
import casadi as ca
import numpy as np

from aircraft_optimal_control.numerical_integration.solvers import configure_solver

from aircraft_optimal_control.optimal_control.objectives import (
    build_objectives,
)

from aircraft_optimal_control.optimal_control.constraints import (
    add_landing_constraints,
)

from aircraft_optimal_control.optimal_control.transcription import (
    add_multiple_shooting_constraints,
    add_hermite_simpson_constraints,
)


class LandingOptimalControlProblem:
    """
    Optimal control problem for aircraft landing.

    State:
        x = [s, h, V, gamma]

    Control:
        u = [T, CL]

    The aircraft model must provide:
        aircraft_model.dynamics(x, u, t)
    """

    STATE_INDICES = {
        "horizontal_position": 0,
        "altitude": 1,
        "airspeed": 2,
        "flight_path_angle": 3,
    }

    CONTROL_INDICES = {
        "thrust": 0,
        "lift_coefficient": 1,
    }

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
        self.config = problem_config
        self.integrator = integrator

        self.objective_names = list(objective_names)
        self.transcription_method = transcription_method.lower()

        if len(self.objective_names) != 2:
            raise ValueError(
                "Exactly two objectives are currently required."
            )

        for name in self.objective_names:
            if name not in self.config.objectives:
                raise ValueError(f"Unknown objective: {name}")

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
        self.X = self.opti.variable(4, self.N + 1)
        self.U = self.opti.variable(2, self.N)

        # ----------------------------------------------------
        # Dynamic constraints
        # ----------------------------------------------------

        if self.transcription_method == "multiple_shooting":

            add_multiple_shooting_constraints(
                opti=self.opti,
                X=self.X,
                U=self.U,
                dynamics=self.aircraft_model.dynamics,
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
                dynamics=self.aircraft_model.dynamics,
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

        control_bounds = {
            "thrust": (
                control_constraints.thrust_min,
                control_constraints.thrust_max,
            ),
            "lift_coefficient": (
                control_constraints.lift_coefficient_min,
                control_constraints.lift_coefficient_max,
            ),
        }

        add_landing_constraints(
            opti=self.opti,
            X=self.X,
            U=self.U,
            initial_state=self._state_vector(
                self.config.initial_state
            ),
            terminal_state=self._state_vector(
                self.config.terminal_state
            ),
            state_indices=self.STATE_INDICES,
            control_indices=self.CONTROL_INDICES,
            state_bounds=state_bounds,
            control_bounds=control_bounds,
        )

        # ----------------------------------------------------
        # Objective selection
        # ----------------------------------------------------

        self.alpha = self.opti.parameter()

        first_objective = self.objectives[
            self.objective_names[0]
        ]

        second_objective = self.objectives[
            self.objective_names[1]
        ]

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

    @staticmethod
    def _state_vector(state):

        return ca.vertcat(
            state.horizontal_position,
            state.altitude,
            state.airspeed,
            state.flight_path_angle,
        )

    # ========================================================
    # Initial guess
    # ========================================================

    def set_initial_guess(self):

        x0 = self._state_vector(self.config.initial_state)
        xf = self._state_vector(self.config.terminal_state)

        for i in range(4):

            guess = np.linspace(
                float(x0[i]),
                float(xf[i]),
                self.N + 1,
            )

            self.opti.set_initial(
                self.X[i, :],
                guess,
            )

        initial_guess = self.config.initial_guess

        self.opti.set_initial(
            self.U[0, :],
            initial_guess.thrust,
        )

        self.opti.set_initial(
            self.U[1, :],
            initial_guess.lift_coefficient,
        )

    # ========================================================
    # Objective selection
    # ========================================================

    def select_objective(self, objective_name):

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