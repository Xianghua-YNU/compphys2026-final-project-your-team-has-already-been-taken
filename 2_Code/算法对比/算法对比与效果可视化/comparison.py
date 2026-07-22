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

figsize1 = (8,5)
figsize2 = (15,10)

@dataclass(frozen=True)
class RigidBodyParams:
    """刚体主转动惯量。这里使用题目给出的网球拍定理参数。"""
    I1: float = 105.8
    I2: float = 434.9
    I3: float = 537.4

    @property
    def I(self) -> np.ndarray:
        return np.array([self.I1, self.I2, self.I3], dtype=float)


@dataclass(frozen=True)
class SimulationConfig:
    """数值模拟与绘图参数。"""

    # 时间范围
    t0: float = 0
    t1: float = 30

    # 初值：
    # 为了明显观察网球拍定理，应让主要角速度沿中间惯量轴 I2，
    # 同时给 I1 / I3 方向一个很小扰动。
    # 如果完全设为 (0, omega2, 0)，理论和数值上都会一直停留在该轴附近。
    omega0: tuple[float, float, float] = (0.01, 10, 0.01)

    # 固定步长方法使用
    dt_fixed: float = 0.02

    # solve_ivp 自适应方法参数
    rtol: float = 1e-9
    atol: float = 1e-11
    solve_ivp_max_step: float = 0.05
    n_plot_points: int = 4000

    # 隐式方程求解参数
    nonlinear_tol: float = 1e-12

    # 自适应 Gauss-Legendre-4 参数
    adaptive_h0: float = 0.02
    adaptive_h_min: float = 1e-5
    adaptive_h_max: float = 0.08
    adaptive_safety: float = 0.90
    adaptive_min_factor: float = 0.20
    adaptive_max_factor: float = 3.00

    # 是否对自适应 GL4 每个接受步做守恒量投影
    project_adaptive_gl4: bool = True

    # 绘图与保存
    output_dir: str = "tennis_racket_output"
    dpi: int = 300
    show_figures: bool = True

    # 守恒量图的模式：
    # "absolute"       -> 画 E 和 |L|
    # "relative_error" -> 画相对误差，便于观察数值漂移
    invariant_plot_mode: str = "absolute"


PARAMS = RigidBodyParams()
CFG = SimulationConfig()


# ============================================================
# 2. 模型、守恒量与工具函数
# ============================================================

def euler_equations(
    t: float,
    omega: np.ndarray,
    p: RigidBodyParams = PARAMS,
) -> np.ndarray:
    """
    刚体自由转动的欧拉动力学方程，角速度写在刚体主轴坐标系中。

    dω1/dt = ((I2 - I3) / I1) ω2 ω3
    dω2/dt = ((I3 - I1) / I2) ω3 ω1
    dω3/dt = ((I1 - I2) / I3) ω1 ω2
    """
    omega1, omega2, omega3 = omega

    return np.array(
        [
            ((p.I2 - p.I3) / p.I1) * omega2 * omega3,
            ((p.I3 - p.I1) / p.I2) * omega3 * omega1,
            ((p.I1 - p.I2) / p.I3) * omega1 * omega2,
        ],
        dtype=float,
    )


def kinetic_energy(
    omega: np.ndarray,
    p: RigidBodyParams = PARAMS,
) -> np.ndarray:
    """
    转动动能：

        E = 1/2 * (I1 ω1^2 + I2 ω2^2 + I3 ω3^2)

    支持 omega.shape == (3,) 或 (N, 3)。
    """
    omega = np.asarray(omega, dtype=float)
    return 0.5 * np.sum(p.I * omega**2, axis=-1)


def angular_momentum_squared(
    omega: np.ndarray,
    p: RigidBodyParams = PARAMS,
) -> np.ndarray:
    """
    角动量模长平方：

        |L|^2 = (I1 ω1)^2 + (I2 ω2)^2 + (I3 ω3)^2

    支持 omega.shape == (3,) 或 (N, 3)。
    """
    omega = np.asarray(omega, dtype=float)
    return np.sum((p.I * omega) ** 2, axis=-1)


def angular_momentum_norm(
    omega: np.ndarray,
    p: RigidBodyParams = PARAMS,
) -> np.ndarray:
    """角动量模长 |L|。"""
    return np.sqrt(angular_momentum_squared(omega, p))


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

def solve_explicit_euler(
    t_span: tuple[float, float],
    omega0: np.ndarray,
    dt: float,
    p: RigidBodyParams = PARAMS,
) -> Solution:
    """
    基础显式欧拉法。

    特点：
    - 一阶精度；
    - 非保结构；
    - 对长期运动通常会出现能量和角动量漂移。
    """
    t = make_time_grid(t_span[0], t_span[1], dt)
    y = np.zeros((len(t), 3), dtype=float)
    y[0] = omega0

    for n in range(len(t) - 1):
        h = t[n + 1] - t[n]
        y[n + 1] = y[n] + h * euler_equations(t[n], y[n], p)

    return Solution("Explicit Euler", t, y)


def solve_with_scipy_ivp(
    method: str,
    t_span: tuple[float, float],
    omega0: np.ndarray,
    cfg: SimulationConfig,
    p: RigidBodyParams = PARAMS,
) -> Solution:
    """
    scipy.solve_ivp 接口。

    method 可取：
    - "RK45"
    - "DOP853"
    - "Radau"
    - "BDF"
    等。
    """
    t_eval = np.linspace(t_span[0], t_span[1], cfg.n_plot_points)

    sol = solve_ivp(
        fun=lambda t, y: euler_equations(t, y, p),
        t_span=t_span,
        y0=omega0,
        method=method,
        t_eval=t_eval,
        rtol=cfg.rtol,
        atol=cfg.atol,
        max_step=cfg.solve_ivp_max_step,
    )

    if not sol.success:
        raise RuntimeError(f"{method} 求解失败：{sol.message}")

    return Solution(method, sol.t, sol.y.T)


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
    隐式中点法一步：

        y_{n+1} = y_n + h f(t_n + h/2, (y_n + y_{n+1})/2)

    它是 Gauss-Legendre 一阶段方法。
    对正则 Hamilton 系统是辛方法。

    对本题：
    - 欧拉刚体方程具有二次守恒量 E 和 |L|^2；
    - 隐式中点作为辛 Runge-Kutta 方法，能很好保持这些二次守恒量。
    """
    y = np.asarray(y, dtype=float)
    guess = y + h * euler_equations(t, y, p)

    def residual(y_next: np.ndarray) -> np.ndarray:
        mid = 0.5 * (y + y_next)
        return y_next - y - h * euler_equations(t + 0.5 * h, mid, p)

    sol = root(residual, guess, tol=nonlinear_tol)

    if not sol.success:
        raise RuntimeError(f"隐式中点法非线性方程求解失败：{sol.message}")

    return sol.x


def gauss_legendre4_step(
    t: float,
    y: np.ndarray,
    h: float,
    p: RigidBodyParams = PARAMS,
    nonlinear_tol: float = 1e-12,
) -> np.ndarray:
    """
    二阶段 Gauss-Legendre 隐式 Runge-Kutta 方法。

    特点：
    - 四阶精度；
    - 辛 Runge-Kutta 格式；
    - 对长期哈密顿系统比普通显式 RK 更稳定；
    - 对本题的二次守恒量保持很好。

    Butcher 系数：

        c1 = 1/2 - sqrt(3)/6
        c2 = 1/2 + sqrt(3)/6

        A = [[1/4, 1/4 - sqrt(3)/6],
             [1/4 + sqrt(3)/6, 1/4]]

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

    # 初始猜测：两个 stage 都取当前点斜率
    f0 = euler_equations(t, y, p)
    k_guess = np.concatenate([f0, f0])

    def residual(k_flat: np.ndarray) -> np.ndarray:
        k1 = k_flat[:3]
        k2 = k_flat[3:]

        y_stage1 = y + h * (a11 * k1 + a12 * k2)
        y_stage2 = y + h * (a21 * k1 + a22 * k2)

        r1 = k1 - euler_equations(t + c1 * h, y_stage1, p)
        r2 = k2 - euler_equations(t + c2 * h, y_stage2, p)

        return np.concatenate([r1, r2])

    sol = root(residual, k_guess, tol=nonlinear_tol)

    if not sol.success:
        raise RuntimeError(f"Gauss-Legendre-4 非线性方程求解失败：{sol.message}")

    k1 = sol.x[:3]
    k2 = sol.x[3:]

    y_next = y + h * (b1 * k1 + b2 * k2)

    return y_next


def solve_fixed_step_method(
    name: str,
    stepper: Callable[[float, np.ndarray, float], np.ndarray],
    t_span: tuple[float, float],
    omega0: np.ndarray,
    dt: float,
) -> Solution:
    """固定步长一步法的统一积分器。"""
    t = make_time_grid(t_span[0], t_span[1], dt)
    y = np.zeros((len(t), 3), dtype=float)
    y[0] = omega0

    for n in range(len(t) - 1):
        h = t[n + 1] - t[n]
        y[n + 1] = stepper(t[n], y[n], h)

    return Solution(name, t, y)


# ============================================================
# 5. 高阶自适应步长 + 辛型 GL4 + 守恒量投影
# ============================================================

def project_to_initial_invariants(
    y: np.ndarray,
    E0: float,
    L20: float,
    p: RigidBodyParams = PARAMS,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    将数值结果轻微投影回初始守恒量曲面：

        E(y) = E0
        |L(y)|^2 = L20

    做法：
        在当前 y 附近，沿 ∇E 与 ∇(|L|^2) 张成的二维方向修正：

            y_new = y + λ ∇E + μ ∇(|L|^2)

        然后通过两个非线性方程求 λ, μ。

    说明：
        对本题 GL 型辛 RK 已经能很好保持二次守恒量。
        投影主要用于抵消浮点误差和非线性求解误差。
    """
    y = np.asarray(y, dtype=float)

    grad_E = p.I * y
    grad_L2 = 2.0 * (p.I ** 2) * y

    if np.linalg.norm(grad_E) < 1e-30 or np.linalg.norm(grad_L2) < 1e-30:
        return y

    def equations(lam_mu: np.ndarray) -> np.ndarray:
        lam, mu = lam_mu

        yc = y + lam * grad_E + mu * grad_L2

        return np.array(
            [
                kinetic_energy(yc, p) - E0,
                angular_momentum_squared(yc, p) - L20,
            ],
            dtype=float,
        )

    sol = root(equations, np.array([0.0, 0.0]), tol=tol)

    if not sol.success:
        return y

    lam, mu = sol.x

    return y + lam * grad_E + mu * grad_L2


def solve_adaptive_gl4_projected(
    t_span: tuple[float, float],
    omega0: np.ndarray,
    cfg: SimulationConfig,
    p: RigidBodyParams = PARAMS,
) -> Solution:
    """
    自适应 Gauss-Legendre-4 + 守恒量投影。

    步长控制：
        使用“一步 h”和“两步 h/2”的差作为局部误差估计。
        GL4 是四阶方法，因此误差估计除以 2^4 - 1。

    结构性质：
        - GL4 本身是四阶辛 Runge-Kutta 方法；
        - 每一步都具有良好的二次守恒量保持性质；
        - 自适应步长严格来说会削弱全局辛结构证明；
        - 因此这里额外加入守恒量投影，使 E 和 |L| 在数值上几乎不漂移。
    """
    t0, t1 = t_span

    t = float(t0)
    y = np.asarray(omega0, dtype=float).copy()

    E0 = float(kinetic_energy(y, p))
    L20 = float(angular_momentum_squared(y, p))

    ts: List[float] = [t]
    ys: List[np.ndarray] = [y.copy()]

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

        # 步长误差估计
        err_vec = (y_two_half - y_full) / (2.0 ** order - 1.0)

        scale = cfg.atol + cfg.rtol * np.maximum(
            np.abs(y),
            np.abs(y_two_half),
        )

        err_norm = float(np.sqrt(np.mean((err_vec / scale) ** 2)))

        if err_norm <= 1.0 or h <= cfg.adaptive_h_min * (1.0 + 1e-14):
            # 接受这一步。
            # 为保留 GL4 的结构性质，不使用 Richardson 外推值，
            # 而采用两次半步得到的 GL4 结果。
            t = t + h
            y = y_two_half

            if cfg.project_adaptive_gl4:
                y = project_to_initial_invariants(
                    y,
                    E0,
                    L20,
                    p=p,
                    tol=cfg.nonlinear_tol,
                )

            ts.append(t)
            ys.append(y.copy())

            # 更新步长
            if err_norm == 0.0:
                factor = cfg.adaptive_max_factor
            else:
                factor = cfg.adaptive_safety * err_norm ** (-1.0 / (order + 1.0))

            factor = min(
                cfg.adaptive_max_factor,
                max(cfg.adaptive_min_factor, factor),
            )

            h = min(
                cfg.adaptive_h_max,
                max(cfg.adaptive_h_min, h * factor),
            )

        else:
            # 拒绝这一步，缩小步长重试
            factor = cfg.adaptive_safety * err_norm ** (-1.0 / (order + 1.0))
            factor = min(1.0, max(cfg.adaptive_min_factor, factor))

            h = max(cfg.adaptive_h_min, h * factor)

            reject_count += 1

            if reject_count > max_reject:
                raise RuntimeError(
                    "自适应 GL4 拒绝步数过多，请放宽容差或检查初值。"
                )

    return Solution(
        "Adaptive GL4 + projection",
        np.array(ts),
        np.vstack(ys),
    )


# ============================================================
# 6. 统一运行
# ============================================================

def run_all_methods(
    cfg: SimulationConfig = CFG,
    p: RigidBodyParams = PARAMS,
) -> Dict[str, Solution]:
    """运行所有算法，并以字典形式返回。"""
    t_span = (cfg.t0, cfg.t1)
    omega0 = np.array(cfg.omega0, dtype=float)

    results: Dict[str, Solution] = {}

    # # 1. 基础欧拉法
    # sol = solve_explicit_euler(
    #     t_span,
    #     omega0,
    #     cfg.dt_fixed,
    #     p,
    # )
    # results[sol.name] = sol

    # 2. RK45：常用自适应 Runge-Kutta
    sol = solve_with_scipy_ivp(
        "RK45",
        t_span,
        omega0,
        cfg,
        p,
    )
    results[sol.name] = sol

    # 3. DOP853：高阶自适应步长方法
    sol = solve_with_scipy_ivp(
        "DOP853",
        t_span,
        omega0,
        cfg,
        p,
    )
    results[sol.name] = sol

    # 4. 常规辛型方法：隐式中点法
    sol = solve_fixed_step_method(
        name="Implicit midpoint",
        stepper=lambda t, y, h: implicit_midpoint_step(
            t,
            y,
            h,
            p=p,
            nonlinear_tol=cfg.nonlinear_tol,
        ),
        t_span=t_span,
        omega0=omega0,
        dt=cfg.dt_fixed,
    )
    results[sol.name] = sol

    # 5. 高阶辛型方法：Gauss-Legendre 四阶
    sol = solve_fixed_step_method(
        name="Gauss-Legendre 4",
        stepper=lambda t, y, h: gauss_legendre4_step(
            t,
            y,
            h,
            p=p,
            nonlinear_tol=cfg.nonlinear_tol,
        ),
        t_span=t_span,
        omega0=omega0,
        dt=cfg.dt_fixed,
    )
    results[sol.name] = sol

    # 6. 高阶自适应步长 + 辛型 GL4 + 守恒量投影
    sol = solve_adaptive_gl4_projected(
        t_span,
        omega0,
        cfg,
        p,
    )
    results[sol.name] = sol

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


def plot_angular_velocity(
    results: Dict[str, Solution],
    cfg: SimulationConfig = CFG,
) -> Path:
    """
    绘制角速度随时间变化：

    - 三个子图分别对应 ω1, ω2, ω3；
    - 每个子图中叠加所有算法的曲线；
    - 高分辨率保存 PNG 和 PDF。
    """
    setup_matplotlib_chinese_font()

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=figsize1,
        sharex=True,
        constrained_layout=True,
    )

    labels = [
        r"$\omega_1$",
        r"$\omega_2$",
        r"$\omega_3$",
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
        ax.grid(True, alpha=0.35)

    axes[-1].set_xlabel("时间 t")
    axes[0].set_title("网球拍定理：不同数值方法得到的角速度分量")
    axes[0].legend(
        loc="best",
        fontsize=8,
        ncol=2,
    )

    save_path = out_dir / "angular_velocity_comparison.png"

    fig.savefig(
        save_path,
        dpi=cfg.dpi,
        bbox_inches="tight",
    )

    fig.savefig(
        out_dir / "angular_velocity_comparison.pdf",
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
    绘制能量曲线与角动量曲线：

    - 两个子图分别对应 E 与 |L|；
    - 每个子图中叠加所有算法；
    - 高分辨率保存 PNG 和 PDF。
    """
    setup_matplotlib_chinese_font()

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=figsize2,
        sharex=True,
        constrained_layout=True,
    )

    mode = cfg.invariant_plot_mode.lower().strip()

    for name, sol in results.items():
        E = kinetic_energy(sol.y, p)
        L = angular_momentum_norm(sol.y, p)

        if mode == "relative_error":
            y_E = (E - E[0]) / E[0]
            y_L = (L - L[0]) / L[0]

            y_label_E = r"$(E-E_0)/E_0$"
            y_label_L = r"$(|\mathbf{L}|-|\mathbf{L}_0|)/|\mathbf{L}_0|$"
            title = "能量与角动量相对误差"

        elif mode == "absolute":
            y_E = E
            y_L = L

            y_label_E = "转动动能 E"
            y_label_L = r"角动量模长 $|\mathbf{L}|$"
            title = "能量曲线与角动量曲线"

        else:
            raise ValueError(
                'cfg.invariant_plot_mode 只能是 "absolute" 或 "relative_error"。'
            )

        axes[0].plot(
            sol.t,
            y_E,
            linewidth=1.2,
            label=name,
        )

        axes[1].plot(
            sol.t,
            y_L,
            linewidth=1.2,
            label=name,
        )

    axes[0].set_ylabel(y_label_E)
    axes[1].set_ylabel(y_label_L)
    axes[1].set_xlabel("时间 t")

    axes[0].set_title(title)
    axes[0].legend(
        loc="best",
        fontsize=8,
        ncol=2,
    )

    for ax in axes:
        ax.grid(True, alpha=0.35)

    suffix = "relative_error" if mode == "relative_error" else "absolute"

    save_path = out_dir / f"invariants_comparison_{suffix}.png"

    fig.savefig(
        save_path,
        dpi=cfg.dpi,
        bbox_inches="tight",
    )

    # 保存向量图
    fig.savefig(
        out_dir / f"invariants_comparison_{suffix}.pdf",
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
    print("开始计算网球拍定理欧拉方程的初值问题...")
    print(f"惯量：I1={PARAMS.I1}, I2={PARAMS.I2}, I3={PARAMS.I3}")
    print(f"时间范围：[{CFG.t0}, {CFG.t1}]")
    print(f"初值 omega0={CFG.omega0}")

    results = run_all_methods(CFG, PARAMS)

    print_invariant_summary(results, PARAMS)

    fig1 = plot_angular_velocity(results, CFG)
    fig2 = plot_invariants(results, CFG, PARAMS)

    print("\n图像已保存：")
    print(f"1. {fig1}")
    print(f"2. {fig2}")
    print(f"输出目录：{Path(CFG.output_dir).resolve()}")


if __name__ == "__main__":
    main()