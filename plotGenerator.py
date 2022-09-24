from matplotlib.pyplot import figure
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes,mark_inset
from matplotlib.animation import FuncAnimation
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from util import *

RADIUS = 250

colors: List = ['r', 'b', "orange", 'g', 'y', 'c']
markers: List = ['o', 'v', '<', '>', 's', 'p']
DPI = 100

def plot(b):
    fig, ax = plt.subplots(figsize = [5, 5], dpi = DPI)

    plots: List = [ax.plot(Util.get_x_coordinates(b.paths[i]), Util.get_y_coordinates(b.paths[i]), markersize=15, color = colors[i])[0] for i in range(5)] 

    def initializeAnimation():
        ax.tick_params(axis='y',
                direction="in",
                right=True, labelsize=18)
        ax.tick_params(axis='x', direction="in" , top=True,bottom=True, labelsize=18)

        ## legends and utilities
        ax.set_xlabel(r"x", fontsize=16)
        ax.set_ylabel(r"y", fontsize=16)

        ## border colors
        ax.patch.set_edgecolor('black')  
        ax.patch.set_linewidth('2') 

        ax.set_xlim(-RADIUS, RADIUS)
        ax.set_ylim(-RADIUS, RADIUS)
        return plots

    def updateAnimation(frame):
        b.update()
        for i, plot in enumerate(plots):
            plot.set_data(Util.get_x_coordinates(b.paths[i]), Util.get_y_coordinates(b.paths[i]))
        return plots

    animation = FuncAnimation(fig, updateAnimation, init_func = initializeAnimation, 
                blit = True, interval = 250)

    plt.show(block=True)
    fig.tight_layout()