import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 参数
I1 = 105.8
I2 = 434.9
I3 = 537.4

a = I2/I1 -1
b = 1 - I2/I3


def set_chinese_font():
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    # 更轻、更小的论文风格
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })


def rhs(theta, psi):
    s = np.sin(psi)
    c = np.cos(psi)

    dtheta = (-b * np.sin(theta) * s * c) / (1 - b**2 * s**2)
    dpsi = (a + b * s**2) * np.cos(theta) / (1 - b * s**2)

    return np.array([dtheta, dpsi])

def rhs_2(phi ,y):
    theta = y[0]
    psi = y[1]
    dy = rhs(theta, psi)
    return [dy[0],dy[1]]

def solve_eps():

    if not (a > 0 and 0 < b < 1):
        raise ValueError("参数必须满足 a > 0 且 0 < b < 1")

    set_chinese_font()

    psi_bounds = (0, 2 * np.pi)
    theta_bounds = (0, np.pi)

    psi = np.linspace(*psi_bounds, 430)
    theta = np.linspace(*theta_bounds, 260)

    PSI, THETA = np.meshgrid(psi, theta)

    DTHETA, DPSI = rhs(THETA, PSI)

    n = 300
    phi_span = [0,n]
    y0 = [1.8039291,np.pi/2]
    phi_eval = np.arange(0,n,0.1)

    sol = solve_ivp(
        rhs_2,
        [0,np.pi()],
        y0,
        method="Radau",
        t_eval=phi_eval,
        rtol=1e-6,
    )
    
    theta, psi = sol.y[0], sol.y[1]

    print(theta)

if __name__ == "__main__":
    solve_eps()