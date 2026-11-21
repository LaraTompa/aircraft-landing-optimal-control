
from aircraft_optimal_control.config.loader import AircraftConfig


def compute_aerodynamic_forces(V, CL, aircraft: AircraftConfig):
    """
    Compute aerodynamic forces for the point-mass aircraft model.

    Parameters
    ----------
    V : casadi expression or float
        Airspeed [m/s].

    CL : casadi expression or float
        Lift coefficient [-].

    aircraft : AircraftConfig
        Aircraft parameters.

    Returns
    -------
    qbar : casadi expression or float
        Dynamic pressure [Pa].

    CD : casadi expression or float
        Drag coefficient [-].

    L : casadi expression or float
        Lift force [N].

    D : casadi expression or float
        Drag force [N].
    """

    # Dynamic pressure
    qbar = 0.5 * aircraft.air_density * V**2

    # Drag coefficient
    CD = aircraft.CD0 + aircraft.k_drag * CL**2

    # Aerodynamic forces
    L = qbar * aircraft.wing_area * CL
    D = qbar * aircraft.wing_area * CD

    return qbar, CD, L, D
