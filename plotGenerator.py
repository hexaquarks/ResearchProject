from nanodomainSimulation import Nanodomain
from simulation import *
from util import *

from matplotlib.animation import FuncAnimation
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy as np

RADIUS = 250
colors: List = ['r', 'b', "orange", 'g', 'y', 'c']
markers: List = ['o', 'v', '<', '>', 's', 'p']
DPI = 100

# 1 pixel = 1 nm
# ==> 

def handle_nanodomain(ax, sim):
    nanodomains = [
        plt.Circle(param[0], param[1], color='black', alpha = 0.1) for param in 
        list(map(lambda coord, radius: (coord, radius), sim.get_nanodomain_coordinates(), sim.get_nanodomain_radii()))
    ]
    ax.add_patch( _ for _ in nanodomains)

def handle_hop_diffusion(ax):
    pass

def set_plot_parameters(ax):
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
    
def plot(sim: Simulation, type: SimulationType):
    fig, ax = plt.subplots(figsize = [5, 5], dpi = DPI)

    plots: List = [ax.plot(Util.get_x_coordinates(sim.paths[i]), Util.get_y_coordinates(sim.paths[i]), markersize=15, color = colors[i])[0] for i in range(5)] 

    def initialize_animation():
        set_plot_parameters(ax)
        if type == SimulationType.BROWNIAN: return plots
        handle_nanodomain(ax, sim) if (type == SimulationType.Nanodomain) else handle_hop_diffusion(ax)
        return plots

    def update_animation(frame):
        sim.update()
        for i, plot in enumerate(plots):
            plot.set_data(Util.get_x_coordinates(sim.paths[i]), Util.get_y_coordinates(sim.paths[i]))
        return plots

    animation = FuncAnimation(fig, update_animation, init_func = initialize_animation, 
                blit = True, interval = 250)

    plt.show(block=True)
    fig.tight_layout()