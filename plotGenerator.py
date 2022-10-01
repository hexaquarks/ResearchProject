from simulations.hopDiffusionSimulation import HopDiffusion
from simulations.nanodomainSimulation import Nanodomain
from simulations.simulation import *
from util import *

from matplotlib.animation import FuncAnimation # type: ignore
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy as np
from matplotlib import rcParams # type: ignore

colors: List[str] = ['r', 'b', "orange", 'g', 'y', 'c']
markers: List[str] = ['o', 'v', '<', '>', 's', 'p']
                
def handle_nanodomain(ax, sim: Nanodomain):
    nanodomains = [
        plt.Circle( # type: ignore
            *param,
            color = 'black', 
            alpha = 0.2) 
        for param in sim.get_nanodomain_attributes()
    ]
    [ax.add_patch(nanodomain) for nanodomain in nanodomains]

def handle_hop_diffusion(ax, sim: HopDiffusion):
    compartments = [
        plt.Rectangle( # type: ignore
            tuple((param[0], param[1])),
            param[2], param[3],
            color = 'black',
            alpha = 0.7,
            clip_on = False)
        for param in sim.boundary_coordinates_for_plot
    ]
    [ax.add_patch(boundary) for boundary in compartments]

def get_coordinates_for_plot(sim, idx: int):
    return Util.get_x_coordinates(sim.paths[idx]), Util.get_y_coordinates(sim.paths[idx])

def get_coordinates_for_heads(sim, idx: int):
    return Util.get_last_point(sim.paths[idx])

def set_plot_parameters(ax):
    ax.tick_params(axis = 'y', direction = "in", right = True, labelsize = 16, pad = 20)
    ax.tick_params(axis = 'x', direction = "in", top = True, bottom = True, labelsize = 16, pad = 20)

    ## legends and utilities
    ax.set_xlabel(r"nm", fontsize=16)
    ax.set_ylabel(r"nm", fontsize=16)

    ## border colors
    ax.patch.set_edgecolor('black')  
    ax.patch.set_linewidth('2') 

    ax.set_xlim(-RADIUS, RADIUS)
    ax.set_ylim(-RADIUS, RADIUS)
    
def plot(sim: Simulation, type: SimulationType):
    fig, ax = plt.subplots(figsize = [5, 5], dpi = DPI) # type: ignore

    path_plots: List = [
        ax.plot(
            *get_coordinates_for_plot(sim, i), 
            markersize=15, color = colors[i])[0] 
        for i in range(5)
    ] 
    
    head_plots: List = [
        ax.plot(
            *get_coordinates_for_heads(sim, i), 
            markersize=7, color = colors[i], marker = markers[i], 
            markerfacecolor="white")[0] 
        for i in range(5)
    ]

    def initialize_animation():
        set_plot_parameters(ax)
        if type == SimulationType.NANODOMAIN: handle_nanodomain(ax, sim)
        elif type == SimulationType.HOPDIFFUSION: handle_hop_diffusion(ax, sim)
        return path_plots

    def update_animation(frame):
        sim.update()
        for i, plot in enumerate(path_plots):
            plot.set_data(*get_coordinates_for_plot(sim, i))
        for i, head_marker in enumerate(head_plots):
            head_marker.set_data(*get_coordinates_for_heads(sim, i))
        return path_plots

    animation = FuncAnimation(
        fig, 
        update_animation, 
        init_func = initialize_animation, 
        interval = 20
    )

    plt.show(block = True) # type: ignore
    fig.tight_layout()

rcParams.update({'figure.autolayout': True})