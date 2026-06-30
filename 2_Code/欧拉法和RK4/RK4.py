import numpy as np
import matplotlib.pyplot as plt

# ===================== 1. 刚体参数 =====================
I1 = 105.8   # 最小转动惯量
I2 = 434.9   # 中间转动惯量（不稳定主轴）
I3 = 537.4   # 最大转动惯量

t_start = 0
t_end = 2.0
dt = 1e-4
t = np.arange(t_start, t_end, dt)
N = len(t)

# 角速度数组初始化（含中间轴分量，理论不稳定轨道）
omega = np.zeros((N, 3))
omega[0, :] = np.array([2.0, 1.0, 0.5])

# 【对照稳定初值，取消注释可观察闭合轨迹】
# omega[0, :] = np.array([2.0, 0.0, 0.05])

# ===================== 2. 欧拉方程导数函数 =====================
def domega_dt(w):
    w1, w2, w3 = w
    dw1 = (I2 - I3) * w2 * w3 / I1
    dw2 = (I3 - I1) * w1 * w3 / I2
    dw3 = (I1 - I2) * w1 * w2 / I3
    return np.array([dw1, dw2, dw3])

# ===================== 3. 标准RK4迭代（数学格式无错） =====================
for i in range(N - 1):
    w = omega[i, :]
    k1 = domega_dt(w)
    k2 = domega_dt(w + dt/2 * k1)
    k3 = domega_dt(w + dt/2 * k2)
    k4 = domega_dt(w + dt * k3)
    omega[i+1, :] = w + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# ===================== 4. 物理量 & 全程守恒误差计算（修复核心） =====================
w1_arr = omega[:, 0]
w2_arr = omega[:, 1]
w3_arr = omega[:, 2]

# 转动动能、角动量模长
Ek = 0.5 * I1 * w1_arr**2 + 0.5 * I2 * w2_arr**2 + 0.5 * I3 * w3_arr**2
L1 = I1 * w1_arr
L2 = I2 * w2_arr
L3 = I3 * w3_arr
AM = np.sqrt(L1**2 + L2**2 + L3**2)

# 全程相对误差（代替错误的损耗功率）
Ek0 = Ek[0]
L0 = AM[0]
rel_err_Ek = np.abs(Ek - Ek0) / Ek0
rel_err_L = np.abs(AM - L0) / L0

# 首尾定量误差
w0 = omega[0, :]
wend = omega[-1, :]
dw_gap = np.linalg.norm(wend - w0)
err_Ek_end = rel_err_Ek[-1]
err_L_end = rel_err_L[-1]

print("========== RK4 首尾守恒误差 ==========")
print(f"初角速度 w0 = {w0}")
print(f"末角速度 w_end = {wend}")
print(f"起点终点角速度距离 |w_end - w0| = {dw_gap:.6e}")
print(f"末端动能相对误差 ΔEk/Ek0 = {err_Ek_end:.6e}")
print(f"末端角动量相对误差 Δ|L|/L0 = {err_L_end:.6e}")
print("说明：极小误差仅来自数值离散+浮点舍入，无真实物理能量耗散")

# ===================== 绘图通用样式 =====================
def format_ax(ax):
    ax.grid(True, alpha=0.25)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.1)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.tick_params(axis="both", labelsize=13, width=1.2)

plt.rcParams["font.family"] = "Times New Roman"

# ===================== 图1：动能、角动量演化曲线 =====================
plt.figure(figsize=(10, 8))
# 子图1 动能
ax1 = plt.subplot(2, 1, 1)
plt.plot(t, Ek / 1e6, linewidth=2, color="#27ae60")
plt.xlabel(r'$t/\mathrm{s}$', fontsize=13)
plt.ylabel(r'$E_\mathrm{k}/\mathrm{MJ}$', fontsize=13)
plt.title("RK4: Kinetic Energy vs Time (Theoretically Strictly Constant)", fontsize=14)
format_ax(ax1)

# 子图2 角动量模长
ax2 = plt.subplot(2, 1, 2)
plt.plot(t, AM / 1e6, linewidth=2, color="#2980b9")
plt.xlabel(r'$t/\mathrm{s}$', fontsize=13)
plt.ylabel(r'$|\boldsymbol{L}| \quad (\mathrm{kg\cdot m^2/s} \times 10^{-6})$', fontsize=13)
plt.title("RK4: Magnitude of Angular Momentum vs Time (Theoretically Strictly Constant)", fontsize=14)
format_ax(ax2)
plt.tight_layout()
plt.show()

# ===================== 图2：全程能量相对误差（替代错误功率图） =====================
plt.figure(figsize=(9, 5))
ax_err = plt.gca()
plt.semilogy(t, rel_err_Ek, linewidth=2, color="#c0392b", label=r"$\Delta E_k/E_{k0}$")
plt.semilogy(t, rel_err_L, linewidth=2, color="#8e44ad", label=r"$\Delta |\boldsymbol{L}|/|\boldsymbol{L}|_0$")
plt.xlabel(r'$t/\mathrm{s}$', fontsize=13)
plt.ylabel("Relative Error (log scale)", fontsize=13)
plt.legend(fontsize=11)
plt.title("Full-Time Numerical Conservation Error (Only Discrete Float Error, No Physical Loss)", fontsize=14)
format_ax(ax_err)
plt.tight_layout()
plt.show()

