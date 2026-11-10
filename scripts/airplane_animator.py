"""Side-view Airbus A320 in landing configuration (gear down, flaps extended).

Geometry is in metres and follows the A320 (length 37.57 m, height 11.76 m,
fuselage diameter about 4.1 m, wheelbase 12.6 m). Body frame: x along the
fuselage from the nose, z up. The pose (x, y, theta) places the main wheel
contact point at (x, y) and pitches the plane about it, theta in radians
(nose up > 0). Hook it up to any trajectory with `animate_airplane(x, y, theta)`.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon
from matplotlib.transforms import Affine2D

PLANE_LENGTH = 37.57

_MAIN_GEAR_X = 17.5
_NOSE_GEAR_X = _MAIN_GEAR_X - 12.64
_MAIN_WHEEL_R = 0.585
_NOSE_WHEEL_R = 0.435

# Body-frame outlines (x from nose, z up from the ground with the plane level).
_FUSELAGE = [
    (0.0, 3.4), (0.4, 4.3), (1.3, 5.2), (2.5, 5.75), (4.0, 6.0), (27.0, 6.0),
    (32.0, 5.7), (37.57, 5.2), (37.57, 4.4), (34.0, 3.6), (30.0, 2.6),
    (26.0, 1.9), (4.0, 1.9), (2.0, 1.95), (0.6, 2.4),
]
_FIN = [(27.5, 5.7), (33.3, 11.76), (36.0, 11.76), (37.2, 5.0)]
_TAILPLANE = [(33.0, 4.9), (36.4, 5.25), (37.4, 5.25), (35.6, 4.75)]
_WING = [(12.0, 2.55), (18.2, 2.5), (18.2, 2.2), (12.4, 2.25)]
_SLAT = [(12.0, 2.55), (11.3, 2.3), (11.9, 2.15), (12.4, 2.25)]
_FLAP = [(17.9, 2.45), (18.3, 2.2), (19.7, 1.2), (19.5, 1.55)]
_COCKPIT_WINDOW = [(1.6, 4.95), (2.9, 5.35), (3.1, 5.2), (2.2, 4.8)]
_ENGINE = [
    (10.6, 1.0), (10.5, 1.75), (10.8, 2.7), (12.5, 2.85), (14.5, 2.55),
    (15.8, 2.1), (15.8, 1.45), (14.5, 0.95), (12.5, 0.65), (10.8, 0.7),
]
_ENGINE_INLET = [(10.6, 1.0), (10.5, 1.75), (10.8, 2.7), (11.2, 2.6), (11.0, 1.75), (11.1, 0.85)]
_MAIN_STRUT = [(_MAIN_GEAR_X - 0.15, 2.3), (_MAIN_GEAR_X + 0.15, 2.3),
               (_MAIN_GEAR_X + 0.1, _MAIN_WHEEL_R), (_MAIN_GEAR_X - 0.1, _MAIN_WHEEL_R)]
_NOSE_STRUT = [(_NOSE_GEAR_X - 0.1, 2.0), (_NOSE_GEAR_X + 0.1, 2.0),
               (_NOSE_GEAR_X + 0.07, _NOSE_WHEEL_R), (_NOSE_GEAR_X - 0.07, _NOSE_WHEEL_R)]
_WINDOWS = [(x, 4.85) for x in np.arange(6.0, 33.0, 0.53)]


def make_airplane(ax):
    """Add the airplane patches to `ax` (pose is set separately); returns them in a list."""
    body, wing, dark = "#eef1f4", "#c5ced6", "#3b4a5a"
    livery = "#1b3a6b"

    def to_body(points):
        # Mirror so the nose points along +x, relative to the main wheel contact.
        p = np.atleast_2d(np.asarray(points, dtype=float))
        return np.column_stack((_MAIN_GEAR_X - p[:, 0], p[:, 1])).reshape(np.shape(points))

    def poly(points, color, z):
        return Polygon(to_body(points), closed=True, facecolor=color,
                       edgecolor=dark, linewidth=0.8, zorder=z)

    def circle(center, radius, color, z, edge="black"):
        return Circle(to_body(center), radius, facecolor=color,
                      edgecolor=edge, linewidth=0.8, zorder=z)

    patches = [
        poly(_MAIN_STRUT, "#7f8c8d", 1),
        poly(_NOSE_STRUT, "#7f8c8d", 1),
        circle((_MAIN_GEAR_X, _MAIN_WHEEL_R), _MAIN_WHEEL_R, "#222222", 2),
        circle((_NOSE_GEAR_X, _NOSE_WHEEL_R), _NOSE_WHEEL_R, "#222222", 2),
        circle((_MAIN_GEAR_X, _MAIN_WHEEL_R), 0.25, "#9aa5b1", 3),
        circle((_NOSE_GEAR_X, _NOSE_WHEEL_R), 0.18, "#9aa5b1", 3),
        poly(_TAILPLANE, wing, 2),
        poly(_FIN, livery, 3),
        poly(_FUSELAGE, body, 4),
        poly(_WING, wing, 5),
        poly(_SLAT, "#8e9aa6", 5),
        poly(_FLAP, "#8e9aa6", 5),
        poly(_ENGINE, "#aab4be", 6),
        poly(_ENGINE_INLET, "#4a5560", 7),
        poly(_COCKPIT_WINDOW, "#2c3e50", 7),
    ]
    patches += [circle(w, 0.15, "#2c3e50", 7, edge="none") for w in _WINDOWS]
    for patch in patches:
        ax.add_patch(patch)
    return patches


def set_pose(ax, patches, x, y, theta, scale=1.0):
    """Place the airplane: contact point at (x, y), pitch theta [rad], size factor `scale`."""
    transform = Affine2D().scale(scale).rotate(theta).translate(x, y) + ax.transData
    for patch in patches:
        patch.set_transform(transform)


def animate_airplane(x, y, theta, target=None, scale=1.0, interval=30,
                     title="Aircraft landing", follow=False, window_width=400.0):
    """
    Animate the airplane along a trajectory.

    x, y:   position [m] of the main wheel contact point (arrays of equal length).
    theta:  pitch angle in radians (array, or scalar for a fixed attitude).
    target: optional x position of the touchdown point to mark on the runway.
    scale:  size factor for visibility (1 = true size relative to the axes in metres).
    follow: keep the airplane centered in a moving horizontal viewport.
    window_width: width of the moving viewport in metres.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    theta = np.broadcast_to(np.asarray(theta, dtype=float), x.shape)
    size = PLANE_LENGTH * scale

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_aspect("equal")
    if follow:
        ax.set_xlim(x[0] - window_width / 2, x[0] + window_width / 2)
    else:
        ax.set_xlim(x.min() - 0.6 * size, x.max() + 0.6 * size)
    ax.set_ylim(min(y.min(), 0) - 0.15 * size, max(y.max(), 0) + 0.45 * size)
    ax.set_xlabel("Position [m]")
    ax.set_ylabel("Altitude [m]")
    ax.set_title(title)
    ax.axhline(0, color="black", linewidth=3, zorder=0)  # runway
    if target is not None:
        ax.plot(target, 0, "x", color="red", markersize=10, markeredgewidth=2, zorder=0)
    trail, = ax.plot([], [], color="tab:blue", linestyle=":", linewidth=1.5, zorder=0)

    patches = make_airplane(ax)
    set_pose(ax, patches, x[0], y[0], theta[0], scale)

    def update(frame):
        set_pose(ax, patches, x[frame], y[frame], theta[frame], scale)
        trail.set_data(x[: frame + 1], y[: frame + 1])
        if follow:
            ax.set_xlim(x[frame] - window_width / 2, x[frame] + window_width / 2)
        return (*patches, trail)

    # Caller must keep the returned object alive, or the animation stops.
    return FuncAnimation(fig, update, frames=x.size, interval=interval, blit=False)


if __name__ == "__main__":
    # Placeholder flare path with ~5 degree nose-up attitude; replace with OCP output.
    x = np.linspace(0.0, 300.0, 150)
    y = 12.0 * (1 - x / 300.0) ** 2
    theta = np.arctan(np.gradient(y, x)) + np.deg2rad(5.0)

    ani = animate_airplane(x, y, theta, target=300.0, scale=2.5,
                           title="Airbus A320 model (placeholder path)")
    plt.show()
