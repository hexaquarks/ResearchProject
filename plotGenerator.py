from simulations.hopDiffusion import HopDiffusion
from simulations.nanodomain import Nanodomain
from simulations.simulation import *
from util import *

from matplotlib.animation import FuncAnimation # type: ignore
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams # type: ignore

colors: list[str] = ['r', 'b', "orange", 'g', 'y', 'c']
markers: list[str] = ['o', 'v', '<', '>', 's', 'p']
                
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

class PlotGenerator:
    def __init__(self, sim: Simulation, type: SimulationType):
        self.fig, self.ax = plt.subplots(figsize = [5, 5], dpi = DPI) # type: ignore
        self.sim = sim
        self.type = type
        
        self.path_plots: list = [
            self.ax.plot(
                *get_coordinates_for_plot(sim, i), 
                markersize=15, color = colors[i])[0] 
            for i in range(5)
        ] 
        
        self.head_plots: list = [
            self.ax.plot(
                *get_coordinates_for_heads(sim, i), 
                markersize=7, color = colors[i], marker = markers[i], 
                markerfacecolor="white")[0] 
            for i in range(5)
        ]
        
    def set_plot_parameters(self):
        self.ax.tick_params(axis = 'y', direction = "in", right = True, labelsize = 16, pad = 20)
        self.ax.tick_params(axis = 'x', direction = "in", top = True, bottom = True, labelsize = 16, pad = 20)

        ## legends and utilities
        self.ax.set_xlabel(r"nm", fontsize=16)
        self.ax.set_ylabel(r"nm", fontsize=16)

        ## border colors
        self.ax.patch.set_edgecolor('black')  
        self.ax.patch.set_linewidth('2') 

        self.ax.set_xlim(-RADIUS, RADIUS)
        self.ax.set_ylim(-RADIUS, RADIUS)
        
    def initialize_animation(self):
        self.set_plot_parameters()
        if self.type == SimulationType.NANODOMAIN: handle_nanodomain(self.ax, self.sim)
        elif self.type == SimulationType.HOPDIFFUSION: handle_hop_diffusion(self.ax, self.sim)
        return self.path_plots

    def update_animation(self, *args):
        self.sim.update()
        for i, plot in enumerate(self.path_plots):
            plot.set_data(*get_coordinates_for_plot(self.sim, i))
        for i, head_marker in enumerate(self.head_plots):
            head_marker.set_data(*get_coordinates_for_heads(self.sim, i))
        return self.path_plots

    def start_animation(self):
        self.animation = FuncAnimation(
            fig = self.fig,
            func = self.update_animation, 
            init_func = self.initialize_animation, 
            interval = 20
        )
        plt.show(block = True) # type: ignore
        self.fig.tight_layout()


rcParams.update({'figure.autolayout': True})