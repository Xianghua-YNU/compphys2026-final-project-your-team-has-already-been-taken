# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 13:01:47 2026

@author: 20277
"""

import numpy as np
import matplotlib.pyplot as plt

I1 = 105.8
I2 = 434.9
I3 = 537.4
a = I2/I1 - 1
b = 1-I2/I3
N = 10000
scale = 0.5
epsilon_0 = np.linspace(0,0.3,N)
gamma = np.linspace(0.07,-0.02,N)
epsilon_01,gamma1 = np.meshgrid(epsilon_0,gamma)
epsilon = epsilon_01 - gamma1/(4*b*epsilon_01+1e-7)
gamma_all_0 = 4*b*epsilon_0**2

fig, ax = plt.subplots(figsize=(6, 4))
dpi = 300
ax.plot(epsilon_0,gamma_all_0,"b",linewidth = 2)
plt.imshow(
    epsilon
    ,cmap = 'gist_ncar' 
    ,extent = (epsilon_0.min(),epsilon_0.max(),gamma.min(),gamma.max())
    ,vmin = -scale
    ,vmax = scale
    )
ax.set_aspect(aspect='auto', adjustable='box')
plt.colorbar()
ax.set_xlabel("$\\epsilon_0$")
ax.set_ylabel("$\\gamma$")

plt.show()