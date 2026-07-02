# -*- coding: utf-8 -*-
"""
Created on Tue May 26 09:16:46 2026

@author: 20277
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / "fig"
FIG_DIR.mkdir(exist_ok=True)

eps = 0.01

def equ(ep ,a ,b):
    s = np.sin(ep)
    c = np.cos(ep)
    part1 = (  np.sqrt(  1+(b/a)*s**2  ) -c   )/(  np.sqrt(  1+(b/a)*s**2  ) +c +eps )
    part2 = np.exp(  -2*np.sqrt( a*b ) * ( np.pi + np.asin(eps +  np.sqrt( b/(a+b+eps) ) * c ) ) )
    return part1 - part2

# # 手机参数
# I1 = 105.8
# I2 = 434.9
# I3 = 537.4

# a = I2/I1 -1
# b = 1 - I2/I3

# 解方程

def get_ep0(a,b):
    
    sol = root( equ ,x0 = 0.01 ,args = (a,b) ,method="lm")
    
    ep0 = sol.x
    
    return ep0[0]

gamma = np.linspace(-0.02,0.04,100)

n = 50

eps = 0.01

a = np.linspace(2+eps, 5-eps ,n)

b = np.linspace(0.05+eps, 0.9-eps ,n)

ep = np.zeros((n,n))

for x in range(n):
    for y in range(n):
        ep[x,y] = get_ep0(a[x],b[y])
        
ep[ep>10] = 0


# ======================
# 论文级参数
# ======================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "mathtext.fontset": "stix",
})

# ======================
# 创建数据
# ======================
x = a
y = b

X, Y = np.meshgrid(x, y)

Z = ep

# ======================
# 创建图像
# ======================
fig = plt.figure(figsize=(8, 6), dpi=300)

ax = fig.add_subplot(111, projection='3d')

# ======================
# 绘制曲面
# ======================
surf = ax.plot_surface(
    X,
    Y,
    Z,

    cmap='viridis',

    edgecolor='none',

    antialiased=True,

    rcount=300,
    ccount=300,
    alpha=0.8,
    shade = False
)

# 手机参数
I1 = 105.8
I2 = 434.9
I3 = 537.4

a = I2/I1 -1
b = 1 - I2/I3

# 绘制手机

ep = get_ep0(a, b)

print(ep)

# 垂线

x0 = a
y0 = b
ax.plot(
    [x0, x0],
    [y0, y0],
    [ax.get_zlim()[0], ax.get_zlim()[1]],

    linestyle='--',

    linewidth=1.2,

    color='black'
)

# ======================
# 坐标轴
# ======================
ax.set_xlabel(r'$a$', labelpad=10)
ax.set_ylabel(r'$b$', labelpad=10)
ax.set_zlabel(r'$\epsilon_0$', labelpad=10)

# ======================
# 视角
# ======================
ax.view_init(
    elev=30,
    azim=45
)

# ======================
# 去背景灰色面
# ======================
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# 网格透明
ax.grid(False)

# ======================
# colorbar
# ======================
cbar = fig.colorbar(
    surf,
    shrink=0.6,
    aspect=15,
    pad=0.1
)

cbar.set_label(r'$\epsilon_0$')

# ======================
# 保存
# ======================
plt.tight_layout()

plt.savefig(
    'epsilon_surface_plot_cellphone.png',
    dpi=600,
    bbox_inches='tight'
)

plt.show()



