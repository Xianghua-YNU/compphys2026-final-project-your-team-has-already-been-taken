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

# 欧拉动力学方程带入欧拉角之后的简化版本
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

def draw_phase_portrait():

    if not (a > 0 and 0 < b < 1):
        raise ValueError("参数必须满足 a > 0 且 0 < b < 1")

    set_chinese_font()

    psi_bounds = (0, 2 * np.pi)
    theta_bounds = (0, np.pi)

    psi = np.linspace(*psi_bounds, 430)
    theta = np.linspace(*theta_bounds, 260)

    PSI, THETA = np.meshgrid(psi, theta)

    DTHETA, DPSI = rhs(THETA, PSI)

    fig, ax = plt.subplots(figsize=(10, 7))
    
    p1 = [
        [np.pi, k * np.pi / 15]
        for k in np.arange(4,12)]
    
    p2 = [
        [np.pi/2, k * np.pi / 40]
        for k in np.arange(18,23)]
    
    p3 = [
        [3*np.pi/2, k * np.pi / 40]
        for k in np.arange(18,23)]
    
    p = np.concatenate([p1 ,p2 ,p3])

    # 相轨线（更细）
    ax.streamplot(
        PSI,
        THETA,
        DPSI,
        DTHETA,
        density=4,
        linewidth=2,
        color = 'black',
        arrowsize= 1,
        start_points=p
    )
    
    ax.set_xlim(*psi_bounds)
    ax.set_ylim(*theta_bounds)

    ax.set_xlabel(r"$\psi$/rad",fontsize = 16)
    ax.set_ylabel(r"$\theta$/rad",fontsize = 16)

    ax.set_title(
        rf"$a={a:g},\ b={b:g}$",
        pad=6,
    )
    
    n = 10

    x_ticks = np.arange(0, 5) * np.pi/2
    y_ticks = np.arange(0, n+1) * np.pi/n

    #    标签
    ylabels = []

    for i in range(n + 1):

        # 自动生成标签
        if i == 0:
            ylabels.append('0')
        else:
            ylabels.append(rf'$\frac{{{i}}}{{{n}}}\pi$')
            
    xlabels = []

    for i in range(5):

        # 自动生成标签
        if i == 0:
            xlabels.append('0')
        elif i%2 == 0:
            k = i//2
            xlabels.append(rf'${k}\pi$')
        else:
            xlabels.append(rf'$\frac{{{i}}}{{2}}\pi$')
        
    plt.xticks(x_ticks ,xlabels ,fontsize = 16)
    plt.yticks(y_ticks ,ylabels ,fontsize = 16)
    
    plt.xlim(0, 2*np.pi)
    ep = 0.1
    plt.ylim(np.pi/3+ep, 2*np.pi/3-ep)
    
    # plot the hetierclinic
    n = 300
    phi_span = [0,n]
    y0 = [1.8039291,np.pi/2]
    phi_eval = np.arange(0,n,0.1)
    
    
    sol = solve_ivp(
        rhs_2,
        phi_span,
        y0,
        method="Radau",
        t_eval=phi_eval,
        rtol=1e-6,
    )
    
    theta, psi = sol.y[0], sol.y[1]
    
    plt.plot(psi,theta ,color = 'red' ,linewidth = 2 ,label = 'heteroclinic orbit')
    
    # 2times
    n = 300
    phi_span = [0,n]
    y0 = [1.8039291,3*np.pi/2]
    phi_eval = np.arange(0,n,0.1)
    
    
    sol = solve_ivp(
        rhs_2,
        phi_span,
        y0,
        method="Radau",
        t_eval=phi_eval,
        rtol=1e-6,
    )
    
    theta, psi = sol.y[0], sol.y[1]
    
    plt.plot(psi,theta ,color = 'red',linewidth = 2)
    
    # # 绘制定理示意图
    # n = 3
    # k = 2
    # phi_span = [-n,n+k]
    # y0 = [np.pi/2,np.pi/2+1.4]
    # phi_eval = np.arange(-n,n+k,0.1)
    #
    #
    # sol = solve_ivp(
    #     rhs_2,
    #     phi_span,
    #     y0,
    #     method="Radau",
    #     t_eval=phi_eval,
    #     rtol=1e-6,
    # )
    #
    # theta, psi = sol.y[0], sol.y[1]
    #
    # plt.plot(psi,theta ,color = 'blue',linewidth = 2)
    #
    # # 绘制定理示意图
    # n = 3
    # k = -0.4
    # phi_span = [-n,n+k]
    # y0 = [np.pi/2+0.06,np.pi-0.15]
    # phi_eval = np.arange(-n,n+k,0.1)
    #
    #
    # sol = solve_ivp(
    #     rhs_2,
    #     phi_span,
    #     y0,
    #     method="Radau",
    #     t_eval=phi_eval,
    #     rtol=1e-6,
    # )
    #
    # theta, psi = sol.y[0], sol.y[1]
    #
    # plt.plot(psi,theta ,color = 'blue',linewidth = 2)
    
    
    # plot the fixed points
    
    plt.plot([0,np.pi, 2*np.pi] ,[np.pi/2 ,np.pi/2 ,np.pi/2 ] ,'o' ,label = 'saddles' ,markersize = 10)
    
    plt.plot([np.pi/2, 3*np.pi/2] ,[np.pi/2 ,np.pi/2 ] ,'o' ,label = 'centers')
    
    
    # 设置范围
    g = 1.8
    shift = 0.05
    plt.xlim(0, 2*np.pi)
    plt.ylim(3.5/10*np.pi, 6.5/10*np.pi)
    
    plt.legend()
    
    plt.savefig("phase_explain.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    draw_phase_portrait()