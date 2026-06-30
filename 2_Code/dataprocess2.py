from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


DEFAULT_DATA_PATH = Path(r"data_intermidiate_1.csv")
SCRIPT_DIR = Path(__file__).resolve().parent
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
    csv_path = find_data_file(csv_path, "data_intermidiate.csv")
    data_intermidiate = np.genfromtxt(csv_path, delimiter=",", names=True)
    data = np.column_stack([data_intermidiate[name] for name in data_intermidiate.dtype.names])

    # MATLAB span = 1560:1839 is inclusive and 1-based.
    span = slice(1559, 1839)
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


def plot_process(
    csv_path: Path = DEFAULT_DATA_PATH,
    output_path: Path = SCRIPT_DIR / "intermediate_axis.png",
    show: bool = False,
) -> Path:
    t, omega = load_omega_data(csv_path)
    tspan = (float(np.min(t)), float(np.max(t)))
    simu_omega = phone_rotate([-25.3751, -8.58784, -22.9441], [4.04, 4.42])

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), constrained_layout=True)
    launch_t = 4.14
    landing_t = 4.42

    configs = [
        (0, r"$\omega_1$/rad$\cdot$s$^{-1}$", (1.2 * np.min(omega[:, 0]), 1.2 * np.max(omega[:, 0]))),
        (1, r"$\omega_2$/rad$\cdot$s$^{-1}$", (1.2 * np.min(omega[:, 1]), 1.2 * np.max(omega[:, 1]))),
        (2, r"$\omega_3$/rad$\cdot$s$^{-1}$", (1.2 * np.min(omega[:, 2]), 5)),
    ]

    for ax, (idx, ylabel, ylim) in zip(axes, configs):
        ax.plot(t, omega[:, idx], linewidth=2, label="experiment")
        ax.plot(simu_omega[:, 0], simu_omega[:, idx + 1], linewidth=2, label="simulation")
        ax.axvline(launch_t, color="r", linestyle="--", label="launch point")
        ax.axvline(landing_t, color="r", linestyle="--", label="landing point")
        ax.set_xlabel("t/s")
        ax.set_ylabel(ylabel)
        ax.set_xlim(tspan)
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
    saved_path = plot_process(show=True)
    print(f"Saved figure to: {saved_path}")
