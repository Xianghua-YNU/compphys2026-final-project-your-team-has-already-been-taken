import numpy as np
import matplotlib.pyplot as plt

# ===================== 1. 参数设置 (和Matlab保持一致) =====================
I1 = 105.8
I2 = 434.9
I3 = 537.4

# 仿真时间与欧拉步长
t_start = 0
t_end = 2.0
dt = 1e-4
t = np.arange(t_start, t_end, dt)
N = len(t)

# 初始化角速度 omega = [ω1, ω2, ω3]
omega = np.zeros((N, 3))
# 自定义初始角速度（刚体典型初值）
omega[0, :] = np.array([2.0, 1.0, 0.5])
w0 = omega[0, :]    # 提取初始角速度（修复未定义w0）
wend = omega[-1, :] # 提取终点角速度（修复未定义wend）

# ===================== 2. 欧拉法求解刚体欧拉方程 =====================
# 刚体自由转动欧拉方程 (外力矩M=0，无能量耗散)：
# I1 dω1/dt = (I2 - I3) ω2 ω3
# I2 dω2/dt = (I3 - I1) ω1 ω3
# I3 dω3/dt = (I1 - I2) ω1 ω2
for i in range(N - 1):
    w1, w2, w3 = omega[i, 0], omega[i, 1], omega[i, 2]
    
    dw1dt = (I2 - I3) * w2 * w3 / I1
    dw2dt = (I3 - I1) * w1 * w3 / I2
    dw3dt = (I1 - I2) * w1 * w2 / I3
    
    # 向前欧拉迭代
    omega[i+1, 0] = w1 + dw1dt * dt
    omega[i+1, 1] = w2 + dw2dt * dt
    omega[i+1, 2] = w3 + dw3dt * dt

# ===================== 3. 计算物理量 Ek、AM、损耗功率 =====================
w1_arr = omega[:, 0]
w2_arr = omega[:, 1]
w3_arr = omega[:, 2]

# 转动动能 Ek = 0.5*I1ω1² + 0.5*I2ω2² + 0.5*I3ω3²
Ek = 0.5 * I1 * w1_arr**2 + 0.5 * I2 * w2_arr**2 + 0.5 * I3 * w3_arr**2

# 角动量矢量 L = [I1ω1, I2ω2, I3ω3]，模长 AM
L1 = I1 * w1_arr
L2 = I2 * w2_arr
L3 = I3 * w3_arr
AM = np.sqrt(L1**2 + L2**2 + L3**2)

# 差分求功率 P = dEk/dt
# 理论无阻尼系统dEk/dt=0，非零值仅为欧拉数值离散误差
dEk_dt = np.diff(Ek) / dt
t_power = t[:-1]  # 功率数组长度少1

# 标记竖线时刻（同Matlab）
t_mark1 = 0.67125
t_mark2 = 1.037

# ===================== 4. 绘图通用格式化函数（复用Matlab风格） =====================
def format_ax(ax):
    ax.grid(True, alpha=0.25)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.1)
    ax.spines["top"].set_linewidth(1.2)
    ax.spines["right"].set_linewidth(1.2)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", labelsize=13, width=1.2)
    ax.set_box_aspect(None)

# ===================== 全局绘图设置 =====================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示方块问题

# ===================== 图1：Ek(t) + AM(t) 上下子图 =====================
plt.figure(figsize=(9, 3))

# 子图1：动能 Ek
ax1 = plt.subplot(2, 1, 1)
plt.plot(t, Ek / 1e6, linewidth=2)
plt.xlabel(r'$t/\mathrm{s}$', fontsize=13)
plt.ylabel(r'$E_\mathrm{k}/\mathrm{MJ}$', fontsize=13)
# plt.axvline(t_mark1, c="r", ls="--")
# plt.axvline(t_mark2, c="r", ls="--")
format_ax(ax1)
plt.title("Kinetic Energy vs Time", fontsize=14)

# 子图2：角动量模长 AM
ax2 = plt.subplot(2, 1, 2)
plt.plot(t, AM / 1e6, linewidth=2, color="#2980b9")
plt.xlabel(r'$t/\mathrm{s}$', fontsize=13)
plt.ylabel(r'$|\boldsymbol{L}| \quad (\mathrm{kg\cdot m^2/s} \times 10^{-6})$', fontsize=13)
# plt.axvline(t_mark1, c="r", ls="--")
# plt.axvline(t_mark2, c="r", ls="--")
format_ax(ax2)
plt.title("Magnitude of Angular Momentum vs Time", fontsize=14)

plt.tight_layout()
plt.show()

# ===================== 图2：等效数值误差功率 dEk/dt =====================
plt.figure(figsize=(9, 3))
ax_p = plt.gca()
plt.plot(t_power, dEk_dt / 1e6, linewidth=2, color="#c0392b")
plt.xlabel(r'$t/\mathrm{s}$', fontsize=13)
plt.ylabel(r'$P_\mathrm{error} \quad (\mathrm{MW})$', fontsize=13)
# plt.axvline(t_mark1, c="r", ls="--")
# plt.axvline(t_mark2, c="r", ls="--")
format_ax(ax_p)
plt.title("Numerical Energy Error dEk/dt", fontsize=14)
plt.tight_layout()
plt.show()

# ===================== 图3：相图 ω1-ω2 平面，判断轨迹闭合 =====================
dt = 0.01
steps = 10000  

# ===================== 初始条件 =====================
w = np.array([2.0, 1.0, 0.5])

traj = []

# ===================== 时间推进 =====================
for i in range(steps):
    w1, w2, w3 = w
    traj.append([w1, w2])

    # 欧拉更新
    dw1 = (I2 - I3) * w2 * w3 / I1
    dw2 = (I3 - I1) * w1 * w3 / I2
    dw3 = (I1 - I2) * w1 * w2 / I3

    w = w + dt * np.array([dw1, dw2, dw3])

traj = np.array(traj)

# ===================== 画图 =====================
plt.figure(figsize=(7,7))

# 轨迹点（一步一步）
plt.plot(traj[:,0], traj[:,1], 'o-', markersize=2, linewidth=1, label="trajectory")

# ===================== 关键：逐步箭头 =====================
for i in range(len(traj)-1):
    x, y = traj[i]
    dx = traj[i+1,0] - traj[i,0]
    dy = traj[i+1,1] - traj[i,1]

    plt.arrow(
        x, y,
        dx, dy,
        head_width=0.02,
        alpha=0.4,
        color="blue"
    )

# 起点
plt.plot(traj[0,0], traj[0,1], 'ro', label="start")

plt.title("Step-by-step phase trajectory ")
plt.xlabel(r'$\omega_1$')
plt.ylabel(r'$\omega_2$')
plt.grid(True)
plt.legend()
plt.show()