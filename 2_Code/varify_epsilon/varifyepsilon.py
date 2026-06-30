from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root


# ============================================================
# 1. 显眼参数区：只改这里即可快速切换实验
# ============================================================

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RigidBodyParams:
    """
    刚体参数

    I1, I2, I3 : 主转动惯量
    M          : 总角动量大小（自由刚体守恒）
    """

    # -------------------------
    # 主转动惯量
    # -------------------------
    I1: float = 105.8
    I2: float = 434.9
    I3: float = 537.4

    # -------------------------
    # 总角动量大小 |L|
    # (自由刚体为常数)
    # -------------------------
    M: float = 1000.0

    @property
    def I(self) -> np.ndarray:
        """主转动惯量数组"""
        return np.array([self.I1, self.I2, self.I3], dtype=float)


@dataclass(frozen=True)
class SimulationConfig:
    """
    数值模拟参数

    状态变量统一采用

        y = (theta, phi, psi)

    其中

        theta : 刚体 x 轴与空间角动量 M 的夹角
        phi   : 绕空间角动量方向的进动角
        psi   : M 在刚体 yz 平面中的方位角
    """

    # =====================================================
    # 时间区间
    # =====================================================
    t0: float = 0.0
    t1: float = 30.0

    # =====================================================
    # 欧拉角初值（单位：rad）
    #
    # y = (theta, phi, psi)
    # =====================================================
    euler0: tuple[float, float, float] = (
        np.deg2rad(87.5),   # theta
        0.0,                # phi
        np.deg2rad(2.5),   # psi
    )

    # =====================================================
    # 固定步长积分
    # =====================================================
    dt_fixed: float = 0.02

    # =====================================================
    # solve_ivp 参数
    # =====================================================
    rtol: float = 1e-9
    atol: float = 1e-11
    solve_ivp_max_step: float = 0.005
    n_plot_points: int = 4000

    # =====================================================
    # Newton 迭代
    # =====================================================
    nonlinear_tol: float = 1e-12

    # =====================================================
    # Adaptive Gauss-Legendre-4
    # =====================================================
    adaptive_h0: float = 0.02
    adaptive_h_min: float = 1e-5
    adaptive_h_max: float = 0.08

    adaptive_safety: float = 0.90
    adaptive_min_factor: float = 0.20
    adaptive_max_factor: float = 3.00

    # =====================================================
    # 输出
    # =====================================================
    output_dir: str = "tennis_racket_output"

    dpi: int = 300

    show_figures: bool = True

    # "absolute"
    # "relative_error"
    invariant_plot_mode: str = "absolute"

    @property
    def L0(self) -> np.ndarray:
        """
        初始角动量在固联坐标系中的分量

            L1 = M cos(theta)
            L2 = M sin(theta) cos(psi)
            L3 = M sin(theta) sin(psi)
        """

        theta, _, psi = self.euler0
        M = PARAMS.M

        return np.array(
            [
                M * np.cos(theta),
                M * np.sin(theta) * np.cos(psi),
                M * np.sin(theta) * np.sin(psi),
            ],
            dtype=float,
        )


# =========================================================
# 全局参数
# =========================================================

PARAMS = RigidBodyParams()

CFG = SimulationConfig()

# ============================================================
# 2. 模型、守恒量与工具函数
# ============================================================

def euler_equations(
    t: float,
    y: np.ndarray,
    p: RigidBodyParams = PARAMS,
) -> np.ndarray:
    """
    自由刚体欧拉动力学（Euler angles）

    状态变量：
        y = [theta, phi, psi]

    M 为总角动量大小（守恒）
    """

    theta, phi, psi = y

    M = p.M

    theta_dot = (
        M * (1.0 / p.I3 - 1.0 / p.I2)
        * np.sin(theta)
        * np.sin(psi)
        * np.cos(psi)
    )

    phi_dot = (
        M
        * (
            np.sin(psi) ** 2 / p.I3
            + np.cos(psi) ** 2 / p.I2
        )
    )

    psi_dot = (
        M
        * (
            1.0 / p.I1
            - np.sin(psi) ** 2 / p.I3
            - np.cos(psi) ** 2 / p.I2
        )
        * np.cos(theta)
    )

    return np.array(
        [theta_dot, phi_dot, psi_dot],
        dtype=float,
    )


def kinetic_energy(
    y: np.ndarray,
    p: RigidBodyParams = PARAMS,
) -> np.ndarray:
    """
    转动能量

        E = 1/2 (
            L1^2/I1 +
            L2^2/I2 +
            L3^2/I3
        )

    y=[theta,phi,psi]
    """

    y = np.asarray(y)

    theta = y[..., 0]
    psi = y[..., 2]

    M = p.M

    return 0.5 * M**2 * (
        np.cos(theta) ** 2 / p.I1
        + np.sin(theta) ** 2 * np.cos(psi) ** 2 / p.I2
        + np.sin(theta) ** 2 * np.sin(psi) ** 2 / p.I3
    )

def angular_momentum_squared(
    y: np.ndarray,
    p: RigidBodyParams = PARAMS,
) -> np.ndarray:
    """
    |L|^2
    """

    y = np.asarray(y)

    shape = y.shape[:-1]

    return np.full(shape, p.M**2)

def angular_momentum_norm(
    y: np.ndarray,
    p: RigidBodyParams = PARAMS,
) -> np.ndarray:
    """
    |L|
    """
    y = np.asarray(y)

    shape = y.shape[:-1]

    return np.full(shape, p.M)


def make_time_grid(t0: float, t1: float, dt: float) -> np.ndarray:
    """生成固定步长时间网格，并保证最后一个点精确等于 t1。"""
    if dt <= 0:
        raise ValueError("dt 必须为正数。")

    ts = [float(t0)]

    while ts[-1] < t1 - 1e-14:
        ts.append(min(ts[-1] + dt, t1))

    return np.array(ts, dtype=float)


@dataclass
class Solution:
    """统一保存一种算法的结果。"""
    name: str
    t: np.ndarray
    y: np.ndarray  # shape = (N, 3)


# ============================================================
# 3. 基础欧拉法、RK45、DOP853
# ============================================================
def solve_with_scipy_ivp(
    method: str,
    t_span: tuple[float, float],
    y0: np.ndarray,
    cfg: SimulationConfig,
    p: RigidBodyParams = PARAMS,
) -> Solution:
    """
    使用 scipy.solve_ivp 求解自由刚体欧拉角动力学。

    状态变量

        y = (theta, phi, psi)

    参数
    ----------
    method : str
        scipy.solve_ivp 的积分方法，可选：
            - RK45
            - DOP853
            - Radau
            - BDF
            - LSODA

    t_span : (t0, tf)
        积分时间区间。

    y0 : ndarray(3,)
        欧拉角初值

            y0 = (theta0, phi0, psi0)

    cfg : SimulationConfig
        数值积分参数。

    p : RigidBodyParams
        刚体参数。
    """

    t_eval = np.linspace(
        t_span[0],
        t_span[1],
        cfg.n_plot_points,
    )

    sol = solve_ivp(
        fun=lambda t, y: euler_equations(t, y, p),
        t_span=t_span,
        y0=y0,
        method=method,
        t_eval=t_eval,
        rtol=cfg.rtol,
        atol=cfg.atol,
        max_step=cfg.solve_ivp_max_step,
    )

    if not sol.success:
        raise RuntimeError(
            f"{method} 求解失败：{sol.message}"
        )

    return Solution(
        name=method,
        t=sol.t,
        y=sol.y.T,
    )
# ============================================================
# 4. 辛型隐式方法：隐式中点法与四阶 Gauss-Legendre 法
# ============================================================
def implicit_midpoint_step(
    t: float,
    y: np.ndarray,
    h: float,
    p: RigidBodyParams = PARAMS,
    nonlinear_tol: float = 1e-12,
) -> np.ndarray:
    """
    隐式中点法一步。

    当前状态变量

        y = (theta, phi, psi)

    满足

        y_{n+1}
            = y_n
            + h f(t_n+h/2, (y_n+y_{n+1})/2)

    它是一阶段 Gauss-Legendre Runge-Kutta 方法，
    为二阶辛积分器。
    """

    y = np.asarray(y, dtype=float)

    # 显式欧拉作为初值猜测
    guess = y + h * euler_equations(t, y, p)

    def residual(y_next: np.ndarray) -> np.ndarray:
        mid = 0.5 * (y + y_next)
        return (
            y_next
            - y
            - h * euler_equations(
                t + 0.5 * h,
                mid,
                p,
            )
        )

    sol = root(
        residual,
        guess,
        tol=nonlinear_tol,
    )

    if not sol.success:
        raise RuntimeError(
            f"隐式中点法非线性方程求解失败：{sol.message}"
        )

    return sol.x


def gauss_legendre4_step(
    t: float,
    y: np.ndarray,
    h: float,
    p: RigidBodyParams = PARAMS,
    nonlinear_tol: float = 1e-12,
) -> np.ndarray:
    """
    二阶段 Gauss-Legendre 四阶辛积分器。

    状态变量

        y = (theta, phi, psi)

    Butcher 系数：

        c1 = 1/2 - sqrt(3)/6
        c2 = 1/2 + sqrt(3)/6

        A = [[1/4, 1/4-sqrt(3)/6],
             [1/4+sqrt(3)/6, 1/4]]

        b = [1/2, 1/2]
    """

    sqrt3 = np.sqrt(3.0)

    c1 = 0.5 - sqrt3 / 6.0
    c2 = 0.5 + sqrt3 / 6.0

    a11 = 0.25
    a12 = 0.25 - sqrt3 / 6.0
    a21 = 0.25 + sqrt3 / 6.0
    a22 = 0.25

    b1 = 0.5
    b2 = 0.5

    y = np.asarray(y, dtype=float)

    # 初始猜测
    f0 = euler_equations(t, y, p)
    k_guess = np.concatenate((f0, f0))

    def residual(k_flat: np.ndarray) -> np.ndarray:

        k1 = k_flat[:3]
        k2 = k_flat[3:]

        y_stage1 = y + h * (a11 * k1 + a12 * k2)
        y_stage2 = y + h * (a21 * k1 + a22 * k2)

        r1 = (
            k1
            - euler_equations(
                t + c1 * h,
                y_stage1,
                p,
            )
        )

        r2 = (
            k2
            - euler_equations(
                t + c2 * h,
                y_stage2,
                p,
            )
        )

        return np.concatenate((r1, r2))

    sol = root(
        residual,
        k_guess,
        tol=nonlinear_tol,
    )

    if not sol.success:
        raise RuntimeError(
            f"Gauss-Legendre-4 非线性方程求解失败：{sol.message}"
        )

    k1 = sol.x[:3]
    k2 = sol.x[3:]

    y_next = y + h * (b1 * k1 + b2 * k2)

    return y_next


def solve_fixed_step_method(
    name: str,
    stepper: Callable[[float, np.ndarray, float], np.ndarray],
    t_span: tuple[float, float],
    y0: np.ndarray,
    dt: float,
) -> Solution:
    """
    固定步长积分器。

    状态变量统一采用

        y = (theta, phi, psi)

    Parameters
    ----------
    name : str
        数值方法名称。

    stepper :
        一步积分器。

    t_span :
        (t0, tf)

    y0 :
        初始欧拉角

            y0 = (theta0, phi0, psi0)

    dt :
        固定步长。
    """

    t = make_time_grid(
        t_span[0],
        t_span[1],
        dt,
    )

    y = np.zeros(
        (len(t), 3),
        dtype=float,
    )

    y[0] = y0

    for n in range(len(t) - 1):

        h = t[n + 1] - t[n]

        y[n + 1] = stepper(
            t[n],
            y[n],
            h,
        )

    return Solution(
        name=name,
        t=t,
        y=y,
    )

# ============================================================
# 5. 高阶自适应步长 + 辛型 GL4 + 守恒量投影
# ============================================================
def solve_adaptive_gl4(
    t_span: tuple[float, float],
    y0: np.ndarray,
    cfg: SimulationConfig,
    p: RigidBodyParams = PARAMS,
) -> Solution:
    """
    自适应 Gauss-Legendre 四阶方法。

    状态变量

        y = (theta, phi, psi)

    步长控制：

        一步 h
        与
        两步 h/2

    的差作为局部误差估计。

    不进行守恒量投影。
    """

    t0, t1 = t_span

    t = float(t0)
    y = np.asarray(y0, dtype=float).copy()

    ts = [t]
    ys = [y.copy()]

    h = cfg.adaptive_h0
    order = 4

    max_reject = 200000
    reject_count = 0

    while t < t1 - 1e-14:

        h = min(h, t1 - t)

        # 一步 h
        y_full = gauss_legendre4_step(
            t,
            y,
            h,
            p=p,
            nonlinear_tol=cfg.nonlinear_tol,
        )

        # 两步 h/2
        y_half = gauss_legendre4_step(
            t,
            y,
            0.5 * h,
            p=p,
            nonlinear_tol=cfg.nonlinear_tol,
        )

        y_two_half = gauss_legendre4_step(
            t + 0.5 * h,
            y_half,
            0.5 * h,
            p=p,
            nonlinear_tol=cfg.nonlinear_tol,
        )

        # 四阶误差估计
        err_vec = (
            y_two_half - y_full
        ) / (2.0**order - 1.0)

        scale = (
            cfg.atol
            + cfg.rtol
            * np.maximum(
                np.abs(y),
                np.abs(y_two_half),
            )
        )

        err_norm = np.sqrt(
            np.mean(
                (err_vec / scale) ** 2
            )
        )

        if err_norm <= 1.0 or h <= cfg.adaptive_h_min:

            t += h

            # 保留 GL4 两个半步结果
            y = y_two_half

            ts.append(t)
            ys.append(y.copy())

            if err_norm == 0.0:
                factor = cfg.adaptive_max_factor
            else:
                factor = (
                    cfg.adaptive_safety
                    * err_norm ** (-1.0 / (order + 1))
                )

            factor = min(
                cfg.adaptive_max_factor,
                max(cfg.adaptive_min_factor, factor),
            )

            h = min(
                cfg.adaptive_h_max,
                max(
                    cfg.adaptive_h_min,
                    h * factor,
                ),
            )

        else:

            factor = (
                cfg.adaptive_safety
                * err_norm ** (-1.0 / (order + 1))
            )

            factor = min(
                1.0,
                max(cfg.adaptive_min_factor, factor),
            )

            h = max(
                cfg.adaptive_h_min,
                h * factor,
            )

            reject_count += 1

            if reject_count > max_reject:
                raise RuntimeError(
                    "Adaptive GL4 拒绝步数过多，请检查模型参数。"
                )

    return Solution(
        name="Adaptive GL4",
        t=np.asarray(ts),
        y=np.asarray(ys),
    )
# ============================================================
# 6. 统一运行
# ============================================================

def run_all_methods(
    cfg: SimulationConfig = CFG,
    p: RigidBodyParams = PARAMS,
) -> Dict[str, Solution]:
    """
    使用不同数值算法求解自由刚体欧拉角动力学。

    状态变量统一采用

        y = (theta, phi, psi)
    """

    t_span = (cfg.t0, cfg.t1)

    # 初始欧拉角
    y0 = np.array(cfg.euler0, dtype=float)

    results: Dict[str, Solution] = {}

    # =====================================================
    # 2. RK45
    # =====================================================
    sol = solve_with_scipy_ivp(
        method="RK45",
        t_span=t_span,
        y0=y0,
        cfg=cfg,
        p=p,
    )
    results[sol.name] = sol

    # =====================================================
    # 3. DOP853
    # =====================================================
    sol = solve_with_scipy_ivp(
        method="DOP853",
        t_span=t_span,
        y0=y0,
        cfg=cfg,
        p=p,
    )
    results[sol.name] = sol

    # # =====================================================
    # # 4. 隐式中点法
    # # =====================================================
    # sol = solve_fixed_step_method(
    #     name="Implicit midpoint",
    #     stepper=lambda t, y, h: implicit_midpoint_step(
    #         t,
    #         y,
    #         h,
    #         p=p,
    #         nonlinear_tol=cfg.nonlinear_tol,
    #     ),
    #     t_span=t_span,
    #     y0=y0,
    #     dt=cfg.dt_fixed,
    # )
    # results[sol.name] = sol

    # # =====================================================
    # # 5. Gauss-Legendre 四阶
    # # =====================================================
    # sol = solve_fixed_step_method(
    #     name="Gauss-Legendre 4",
    #     stepper=lambda t, y, h: gauss_legendre4_step(
    #         t,
    #         y,
    #         h,
    #         p=p,
    #         nonlinear_tol=cfg.nonlinear_tol,
    #     ),
    #     t_span=t_span,
    #     y0=y0,
    #     dt=cfg.dt_fixed,
    # )
    # results[sol.name] = sol
    #
    # # =====================================================
    # # 6. Adaptive GL4
    # # =====================================================
    # sol = solve_adaptive_gl4(
    #     t_span=t_span,
    #     y0=y0,
    #     cfg=cfg,
    #     p=p,
    # )
    #
    # results[sol.name] = sol

    return results

# ============================================================
# 7. 守恒量误差统计
# ============================================================

def print_invariant_summary(
    results: Dict[str, Solution],
    p: RigidBodyParams = PARAMS,
) -> None:
    """在控制台打印不同算法的最大相对能量误差与角动量误差。"""
    print("\n===== 守恒量误差统计 =====")
    print(
        f"{'Method':<28s} "
        f"{'max rel. E error':>18s} "
        f"{'max rel. |L| error':>20s}"
    )

    for name, sol in results.items():
        E = kinetic_energy(sol.y, p)
        L = angular_momentum_norm(sol.y, p)

        E0 = E[0]
        L0 = L[0]

        max_E_err = np.max(np.abs((E - E0) / E0))
        max_L_err = np.max(np.abs((L - L0) / L0))

        print(
            f"{name:<28s} "
            f"{max_E_err:18.6e} "
            f"{max_L_err:20.6e}"
        )


# ============================================================
# 8. 绘图函数
# ============================================================

def setup_matplotlib_chinese_font() -> None:
    """尽量兼容 Windows / macOS / Linux 的中文显示。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def plot_euler_angles(
    results: Dict[str, Solution],
    cfg: SimulationConfig = CFG,
) -> Path:
    """
    绘制欧拉角随时间变化。

    状态变量：

        y = (theta, phi, psi)

    三个子图分别绘制

        theta(t)
        phi(t)
        psi(t)

    每个子图叠加所有算法结果。
    """

    setup_matplotlib_chinese_font()

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 9),
        sharex=True,
        constrained_layout=True,
    )

    labels = [
        r"$\theta$ (rad)",
        r"$\phi$ (rad)",
        r"$\psi$ (rad)",
    ]

    titles = [
        r"$\theta(t)$",
        r"$\phi(t)$",
        r"$\psi(t)$",
    ]

    for i, ax in enumerate(axes):

        for name, sol in results.items():

            ax.plot(
                sol.t,
                sol.y[:, i],
                linewidth=1.2,
                label=name,
            )

        ax.set_ylabel(labels[i])

        ax.set_title(titles[i], fontsize=11)

        ax.grid(True, alpha=0.35)

    axes[-1].set_xlabel("时间 $t$")

    axes[0].legend(
        loc="best",
        fontsize=8,
        ncol=2,
    )

    fig.suptitle(
        "自由刚体欧拉角动力学：不同数值方法比较",
        fontsize=14,
    )

    save_path = out_dir / "euler_angle_comparison.png"

    fig.savefig(
        save_path,
        dpi=cfg.dpi,
        bbox_inches="tight",
    )

    fig.savefig(
        out_dir / "euler_angle_comparison.pdf",
        bbox_inches="tight",
    )

    if cfg.show_figures:
        plt.show()
    else:
        plt.close(fig)

    return save_path

def plot_invariants(
    results: Dict[str, Solution],
    cfg: SimulationConfig = CFG,
    p: RigidBodyParams = PARAMS,
) -> Path:
    """
    绘制自由刚体欧拉角动力学的总能量。

    单个子图：

        E : 总能量

    每条曲线对应一种数值方法。

    状态变量：

        sol.y = (theta, phi, psi)
    """

    setup_matplotlib_chinese_font()

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(
        figsize=(11, 5),
        constrained_layout=True,
    )

    mode = cfg.invariant_plot_mode.lower().strip()

    for name, sol in results.items():

        E = kinetic_energy(sol.y, p)

        if mode == "relative_error":

            y = (E - E[0]) / E[0]

            ylabel = r"$(E-E_0)/E_0$"

            title = "自由刚体欧拉角动力学总能量相对误差"

        elif mode == "absolute":

            y = E

            ylabel = r"总能量 $E$"

            title = "自由刚体欧拉角动力学总能量"

        else:

            raise ValueError(
                'cfg.invariant_plot_mode 只能取 "absolute" 或 "relative_error"。'
            )

        ax.plot(
            sol.t,
            y,
            linewidth=1.2,
            label=name,
        )

    ax.set_xlabel("时间 $t$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.grid(True, alpha=0.35)

    ax.legend(
        loc="best",
        fontsize=8,
        ncol=2,
    )

    suffix = (
        "relative_error"
        if mode == "relative_error"
        else "absolute"
    )

    save_path = (
        out_dir
        / f"energy_comparison_{suffix}.png"
    )

    fig.savefig(
        save_path,
        dpi=cfg.dpi,
        bbox_inches="tight",
    )

    fig.savefig(
        out_dir
        / f"energy_comparison_{suffix}.pdf",
        bbox_inches="tight",
    )

    if cfg.show_figures:
        plt.show()
    else:
        plt.close(fig)

    return save_path
# ============================================================
# 9. 主程序
# ============================================================

def main() -> None:
    """主程序入口。"""

    print("开始计算自由刚体欧拉角动力学方程的初值问题...")

    print(
        f"主转动惯量："
        f"I1={PARAMS.I1}, "
        f"I2={PARAMS.I2}, "
        f"I3={PARAMS.I3}"
    )

    print(f"总角动量：M={PARAMS.M}")

    print(f"时间范围：[{CFG.t0}, {CFG.t1}]")

    theta0, phi0, psi0 = CFG.euler0

    print(
        "初始欧拉角：\n"
        f"  theta = {theta0:.6f} rad ({np.rad2deg(theta0):.2f}°)\n"
        f"  phi   = {phi0:.6f} rad ({np.rad2deg(phi0):.2f}°)\n"
        f"  psi   = {psi0:.6f} rad ({np.rad2deg(psi0):.2f}°)"
    )

    print("\n初始角动量（固联坐标系）：")

    L1, L2, L3 = CFG.L0

    print(
        f"  L = ({L1:.6f}, {L2:.6f}, {L3:.6f})"
    )

    # 数值积分
    results = run_all_methods(CFG, PARAMS)

    # 守恒量统计
    print_invariant_summary(results, PARAMS)

    # 绘图
    fig1 = plot_euler_angles(results, CFG)

    fig2 = plot_invariants(results, CFG, PARAMS)

    print("\n图像已保存：")

    print(f"1. {fig1}")

    print(f"2. {fig2}")

    print(f"输出目录：{Path(CFG.output_dir).resolve()}")


if __name__ == "__main__":
    main()