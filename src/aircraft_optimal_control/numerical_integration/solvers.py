def configure_solver(opti, solver_name="ipopt", options=None):
    """
    Configure a CasADi Opti problem with a numerical solver.

    Parameters
    ----------
    opti : casadi.Opti
        Optimization problem to configure.

    solver_name : str
        CasADi solver plugin name, e.g. "ipopt".

    options : dict, optional
        Solver options.

        The "expand" option is passed to the CasADi plugin.
        Remaining options are passed to the solver itself.
    """

    options = dict(options or {})

    # CasADi plugin options
    plugin_options = {
        "expand": options.pop("expand", True)
    }

    # Solver-specific options
    solver_options = options

    opti.solver(
        solver_name,
        plugin_options,
        solver_options,
    )

    return opti
