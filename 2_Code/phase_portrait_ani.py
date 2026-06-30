import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import solve_ivp


I1 = 105.8
I2 = 434.9
I3 = 537.4

a = I2 / I1 - 1
b = 1 - I2 / I3


def set_chinese_font():
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )


def rhs(theta, psi):
    s = np.sin(psi)
    c = np.cos(psi)

    dtheta = (-b * np.sin(theta) * s * c) / (1 - b**2 * s**2)
    dpsi = (a + b * s**2) * np.cos(theta) / (1 - b * s**2)

    return np.array([dtheta, dpsi])


def rhs_2(phi, y):
    theta = y[0]
    psi = y[1]
    dy = rhs(theta, psi)
    return [dy[0], dy[1]]


def draw_phase_portrait(ax=None):
    if not (a > 0 and 0 < b < 1):
        raise ValueError("Parameters must satisfy a > 0 and 0 < b < 1")

    set_chinese_font()

    psi_bounds = (0, 2 * np.pi)
    theta_bounds = (0, np.pi)

    psi = np.linspace(*psi_bounds, 430)
    theta = np.linspace(*theta_bounds, 260)

    psi_grid, theta_grid = np.meshgrid(psi, theta)
    dtheta, dpsi = rhs(theta_grid, psi_grid)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = ax.figure

    p1 = [[np.pi, k * np.pi / 15] for k in np.arange(4, 12)]
    p2 = [[np.pi / 2, k * np.pi / 40] for k in np.arange(18, 23)]
    p3 = [[3 * np.pi / 2, k * np.pi / 40] for k in np.arange(18, 23)]
    start_points = np.concatenate([p1, p2, p3])

    ax.streamplot(
        psi_grid,
        theta_grid,
        dpsi,
        dtheta,
        density=4,
        linewidth=2,
        color="black",
        arrowsize=1,
        start_points=start_points,
    )

    ax.set_xlim(*psi_bounds)
    ax.set_ylim(*theta_bounds)
    ax.set_xlabel(r"$\psi$/rad", fontsize=16)
    ax.set_ylabel(r"$\theta$/rad", fontsize=16)
    ax.set_title(rf"$a={a:g},\ b={b:g}$", pad=6)

    n_ticks = 10
    x_ticks = np.arange(0, 5) * np.pi / 2
    y_ticks = np.arange(0, n_ticks + 1) * np.pi / n_ticks

    y_labels = []
    for i in range(n_ticks + 1):
        if i == 0:
            y_labels.append("0")
        else:
            y_labels.append(rf"$\frac{{{i}}}{{{n_ticks}}}\pi$")

    x_labels = []
    for i in range(5):
        if i == 0:
            x_labels.append("0")
        elif i % 2 == 0:
            x_labels.append(rf"${i // 2}\pi$")
        else:
            x_labels.append(rf"$\frac{{{i}}}{{2}}\pi$")

    ax.set_xticks(x_ticks, x_labels, fontsize=16)
    ax.set_yticks(y_ticks, y_labels, fontsize=16)

    plot_heteroclinic_orbits(ax)

    ax.plot(
        [0, np.pi, 2 * np.pi],
        [np.pi / 2, np.pi / 2, np.pi / 2],
        "o",
        label="saddles",
        markersize=10,
    )
    ax.plot(
        [np.pi / 2, 3 * np.pi / 2],
        [np.pi / 2, np.pi / 2],
        "o",
        label="centers",
    )

    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(3.5 / 10 * np.pi, 6.5 / 10 * np.pi)
    ax.legend(loc="upper right")

    return fig, ax


def plot_heteroclinic_orbits(ax):
    phi_span = [0, 300]
    phi_eval = np.arange(0, 300, 0.1)

    for i, psi0 in enumerate([np.pi / 2, 3 * np.pi / 2]):
        sol = solve_ivp(
            rhs_2,
            phi_span,
            [1.8039291, psi0],
            method="Radau",
            t_eval=phi_eval,
            rtol=1e-6,
        )
        theta, psi = sol.y[0], sol.y[1]
        ax.plot(
            psi,
            theta,
            color="red",
            linewidth=2,
            label="heteroclinic orbit" if i == 0 else None,
        )


def integrate_phase_points(initial_points, t_eval):
    trajectories = []

    for theta0, psi0 in initial_points:
        sol = solve_ivp(
            rhs_2,
            (t_eval[0], t_eval[-1]),
            [theta0, psi0],
            method="Radau",
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-10,
        )
        if not sol.success:
            raise RuntimeError(f"Failed to integrate from theta={theta0}, psi={psi0}")
        trajectories.append(sol.y)

    return np.array(trajectories)


def wrap_psi(psi):
    return np.mod(psi, 2 * np.pi)


def break_wrapped_line(psi, theta):
    x = wrap_psi(psi)
    y = np.array(theta, copy=True)
    jumps = np.where(np.abs(np.diff(x)) > np.pi)[0]

    if len(jumps) == 0:
        return x, y

    x_parts = []
    y_parts = []
    start = 0
    for jump in jumps:
        x_parts.extend(x[start : jump + 1])
        y_parts.extend(y[start : jump + 1])
        x_parts.append(np.nan)
        y_parts.append(np.nan)
        start = jump + 1
    x_parts.extend(x[start:])
    y_parts.extend(y[start:])

    return np.array(x_parts), np.array(y_parts)


def make_phase_portrait_animation(
    output_path="phase_portrait_animation.mp4",
    initial_points=None,
    t_end=45.0,
    frames=360,
    fps=30,
    dpi=180,
    bitrate=3500,
    codec="mpeg4",
    trail_length=90,
    interval=30,
    show=False,
):
    if initial_points is None:
        initial_points = [
            [np.pi / 2 + 0.060, np.pi / 2 - 0.38],
            [np.pi / 2 - 0.055, np.pi / 2 + 0.34],
            [np.pi / 2 + 0.045, 3 * np.pi / 2 - 0.36],
            [np.pi / 2 - 0.050, 3 * np.pi / 2 + 0.32],
            [np.pi / 2 + 0.090, np.pi - 0.18],
            [np.pi / 2 - 0.080, np.pi + 0.18],
        ]

    output_path = Path(output_path)
    t_eval = np.linspace(0, t_end, frames)
    trajectories = integrate_phase_points(np.asarray(initial_points, dtype=float), t_eval)

    fig, ax = draw_phase_portrait()

    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    trail_lines = []
    point_artists = []

    for color in colors:
        (trail_line,) = ax.plot([], [], color=color, linewidth=2.2, alpha=0.75)
        (point,) = ax.plot(
            [],
            [],
            "o",
            color=color,
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=0.8,
        )
        trail_lines.append(trail_line)
        point_artists.append(point)

    title = ax.text(
        0.02,
        0.96,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 3},
    )

    def init():
        for trail_line, point in zip(trail_lines, point_artists):
            trail_line.set_data([], [])
            point.set_data([], [])
        title.set_text("")
        return [*trail_lines, *point_artists, title]

    def update(frame):
        start = max(0, frame - trail_length)
        for i, (trail_line, point) in enumerate(zip(trail_lines, point_artists)):
            theta = trajectories[i, 0]
            psi = trajectories[i, 1]
            trail_x, trail_y = break_wrapped_line(psi[start : frame + 1], theta[start : frame + 1])
            trail_line.set_data(trail_x, trail_y)
            point.set_data([wrap_psi(psi[frame])], [theta[frame]])

        title.set_text(rf"$\phi={t_eval[frame]:.2f}$")
        return [*trail_lines, *point_artists, title]

    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=frames,
        interval=interval,
        blit=True,
    )

    save_animation(ani, output_path, fps=fps, dpi=dpi, bitrate=bitrate, codec=codec)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def save_animation(ani, output_path, fps=30, dpi=180, bitrate=3500, codec="mpeg4"):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".gif":
        writer = animation.PillowWriter(fps=fps)
    elif suffix in {".mp4", ".m4v", ".mov"}:
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError(
                "Saving MP4/MOV needs ffmpeg. Install ffmpeg or use an output path ending in .gif."
            )
        writer = animation.FFMpegWriter(
            fps=fps,
            bitrate=bitrate,
            codec=codec,
            extra_args=["-pix_fmt", "yuv420p"],
        )
    else:
        raise ValueError("Output path must end with .mp4, .m4v, .mov, or .gif")

    ani.save(output_path, writer=writer, dpi=dpi)


def draw_static_phase_portrait(output_path="phase_explain.png", dpi=300, show=True):
    fig, _ = draw_phase_portrait()
    fig.savefig(output_path, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw a phase portrait and save an animation of multiple moving phase points."
    )
    parser.add_argument("--output", default="phase_portrait_animation.mp4", help="Output .mp4/.mov/.gif file.")
    parser.add_argument("--t-end", type=float, default=45.0, help="Integration end time.")
    parser.add_argument("--frames", type=int, default=360, help="Number of animation frames.")
    parser.add_argument("--fps", type=int, default=30, help="Saved animation frames per second.")
    parser.add_argument("--dpi", type=int, default=180, help="Saved animation resolution quality.")
    parser.add_argument("--bitrate", type=int, default=3500, help="MP4/MOV bitrate in kbps.")
    parser.add_argument("--codec", default="mpeg4", help="FFmpeg video codec, e.g. mpeg4 or libx264.")
    parser.add_argument("--trail-length", type=int, default=90, help="Number of previous frames shown as trails.")
    parser.add_argument("--interval", type=int, default=30, help="Preview interval between frames in milliseconds.")
    parser.add_argument("--show", action="store_true", help="Show the Matplotlib window after saving.")
    parser.add_argument(
        "--static",
        action="store_true",
        help="Save only the static phase portrait instead of the animation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.static:
        draw_static_phase_portrait(output_path="phase_explain.png", dpi=args.dpi, show=args.show)
    else:
        make_phase_portrait_animation(
            output_path=args.output,
            t_end=args.t_end,
            frames=args.frames,
            fps=args.fps,
            dpi=args.dpi,
            bitrate=args.bitrate,
            codec=args.codec,
            trail_length=args.trail_length,
            interval=args.interval,
            show=args.show,
        )
