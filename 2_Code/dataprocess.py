from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


DEFAULT_DATA_PATH = Path(r"data_smallest_1.csv")
SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / "fig"
FIG_DIR.mkdir(exist_ok=True)
I1 = 105.8
I2 = 434.9
I3 = 537.4


def euler_equations(t: float, omega: np.ndarray) -> list[float]:
    omega1, omega2, omega3 = omega
    return [
        ((I2 - I3) / I1) * omega2 * omega3,
        ((I3 - I1) / I2) * omega3 * omega1,
        ((I1 - I2) / I3) * omega1 * omega2,
    ]


def phone_rotate(
    omega_0: list[float] | tuple[float, float, float] | np.ndarray,
    tspan: list[float] | tuple[float, float],
    *,
    points: int = 500,
) -> np.ndarray:
    t_eval = np.linspace(float(tspan[0]), float(tspan[1]), points)
    sol = solve_ivp(
        euler_equations,
        (float(tspan[0]), float(tspan[1])),
        np.asarray(omega_0, dtype=float),
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-11,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")
    return np.column_stack((sol.t, sol.y.T))


def load_omega_data(csv_path: Path = DEFAULT_DATA_PATH) -> tuple[np.ndarray, np.ndarray]:
    csv_path = find_data_file(csv_path, "data_smallest.csv")
    data_smallest = np.genfromtxt(csv_path, delimiter=",", names=True)
    data = np.column_stack([data_smallest[name] for name in data_smallest.dtype.names])

    # MATLAB span = 205:520 is inclusive and 1-based.
    span = slice(204, 520)
    omega = data[span][:, [2, 1, 3]].copy()
    omega[:, 1] = -omega[:, 1]
    t = data[span][:, 0]
    return t, omega


def find_data_file(csv_path: Path, filename: str) -> Path:
    candidates = [
        csv_path,
        SCRIPT_DIR / filename,
        SCRIPT_DIR.parent / filename,
        Path.home() / "Desktop" / "新建文件夹" / "TRE" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Cannot find {filename}. Searched:\n{searched}")


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.1)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.tick_params(labelsize=13)


# def plot_energy_momentum(
#     csv_path: Path = DEFAULT_DATA_PATH,
#     output_path: Path = SCRIPT_DIR / "energy_momentum.png",
#     show: bool = False,
# ) -> Path:
#     t, omega = load_omega_data(csv_path)
#     angular_momentum = np.sqrt(
#         (I1 * omega[:, 0]) ** 2
#         + (I2 * omega[:, 1]) ** 2
#         + (I3 * omega[:, 2]) ** 2
#     )
#     kinetic_energy = (
#         I1 * omega[:, 0] ** 2 / 2
#         + I2 * omega[:, 1] ** 2 / 2
#         + I3 * omega[:, 2] ** 2 / 2
#     )
#     p_resist = np.diff(kinetic_energy) / np.diff(t)
#
#     launch_t = 0.67125
#     landing_t = 1.037
#
#     fig = plt.figure(figsize=(10, 6), constrained_layout=True)
#     grid = fig.add_gridspec(2, 2)
#     ax_energy = fig.add_subplot(grid[0, 0])
#     ax_momentum = fig.add_subplot(grid[1, 0])
#     ax_power = fig.add_subplot(grid[:, 1])
#
#     ax_energy.plot(t, kinetic_energy / 1_000_000, linewidth=2)
#     ax_energy.set_xlabel("t/s")
#     ax_energy.set_ylabel(r"E$_k$/J")
#
#     ax_momentum.plot(t, angular_momentum / 1_000_000, linewidth=2)
#     ax_momentum.set_xlabel("t/s")
#     ax_momentum.set_ylabel(r"AM/Kg$\cdot$m$^2$")
#
#     ax_power.plot(t[:-1], p_resist / 1_000_000, linewidth=2)
#     ax_power.set_xlabel("t/s")
#     ax_power.set_ylabel(r"P$_{resist}$(W)")
#
#     for ax in (ax_energy, ax_momentum, ax_power):
#         ax.axvline(launch_t, color="r", linestyle="--")
#         ax.axvline(landing_t, color="r", linestyle="--")
#         style_axes(ax)
#
#     output_path: Path = FIG_DIR / "smallest_axis.png",
#     fig.savefig(output_path, dpi=1200)
#     if show:
#         plt.show()
#     plt.close(fig)
#     return output_path


def plot_process(
    csv_path: Path = DEFAULT_DATA_PATH,
    output_path: Path = SCRIPT_DIR / "energy.png",
    show: bool = False,
) -> Path:
    t, omega = load_omega_data(csv_path)
    tspan = (float(np.min(t)), float(np.max(t)))
    simu_omega = phone_rotate([-30.0091, 3.42526, -2.15902], [0.67125, 1.037])

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), constrained_layout=True)
    launch_t = 0.67125
    landing_t = 1.037

    configs = [
        (0, r"$\omega_1$/rad$\cdot$s$^{-1}$", (1.2 * np.min(omega[:, 0]), 1.2 * np.max(omega[:, 0]))),
        (1, r"$\omega_2$/rad$\cdot$s$^{-1}$", (1.2 * np.min(omega[:, 1]), 8.27)),
        (2, r"$\omega_3$/rad$\cdot$s$^{-1}$", None),
    ]

    for ax, (idx, ylabel, ylim) in zip(axes, configs):
        ax.plot(t, omega[:, idx], linewidth=2, label="experiment")
        ax.plot(simu_omega[:, 0], simu_omega[:, idx + 1], linewidth=2, label="simulation")
        ax.axvline(launch_t, color="r", linestyle="--", label="launch point")
        ax.axvline(landing_t, color="r", linestyle="--", label="landing point")
        ax.set_xlabel("t/s")
        ax.set_ylabel(ylabel)
        ax.set_xlim(tspan)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.legend()
        style_axes(ax)

    output_path = output_path.resolve()
    fig.savefig(output_path, dpi=1200)
    if show:
        plt.show()
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    saved_paths = [
        plot_process(show=True),
        # plot_energy_momentum(show=True),
    ]
    for saved_path in saved_paths:
        print(f"Saved figure to: {saved_path}")
