# -*- coding: utf-8 -*-


import numpy as np
import matplotlib as mat
from matplotlib import font_manager, rc

# font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/Arial.ttf").get_name()
# font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/NANUMBARUNPENR.TTF").get_name()


def setPlotStyle():
    # plt.figure(dpi=300)
    mat.rcParams['font.family'] = 'Arial'
    mat.rcParams['font.size'] = 15
    mat.rcParams['legend.fontsize'] = 15
    mat.rcParams['lines.linewidth'] = 2
    mat.rcParams['lines.color'] = 'r'
    mat.rcParams['axes.grid'] = 1
    mat.rcParams['axes.xmargin'] = 0.1
    mat.rcParams['axes.ymargin'] = 0.1
    mat.rcParams["mathtext.fontset"] = "dejavuserif"  # "cm", "stix", etc.
    mat.rcParams['figure.dpi'] = 500
    mat.rcParams['savefig.dpi'] = 500
