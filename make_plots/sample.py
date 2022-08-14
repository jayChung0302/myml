# -*- coding: utf-8 -*-

import config.preset as Preset
import matplotlib.pyplot as plt
import numpy as np
import sys
# sys.path.append()

Preset.setPlotStyle()

data = np.genfromtxt(r'./data/tutorial_data.txt', skip_header=1)

x = data[:, 0]
y = data[:, 1]

plt.plot(x, y, '^', label='Raw Data', color='red')

linfit = np.polyfit(x, y, 1)
f_1d = np.poly1d(linfit)

plt.plot(x, f_1d(x), label='Linear Fit', color='black')

plt.legend(loc='best')
# plt.legend(loc = (1.02, 0.02))

plt.xlabel('X [-]', fontsize=20)
plt.ylabel('Y [-]', fontsize=20)

# data = np.genfromtxt(r'data.txt', skip_header = 1)

# x = data[:,0]
# y = data[:,1]

# plt.plot(x,y, 'o',  label = 'Raw Data', color = 'red')

# linfit = np.polyfit(x,y,1)
# f_1d = np.poly1d(linfit)

# # plt.plot(x,f_1d(x), label = 'Linear Fit', color = 'black')

# plt.plot(x,f_1d(x), label = 'Linear Fit', color = 'black')

# plt.xlabel('X [-]', fontsize=20)
# plt.ylabel('Y [-]', fontsize=20)
# plt.legend(loc = 'best')
# plt.legend(loc = (1.02,0.02))

# %%
