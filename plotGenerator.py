from simulations.hopDiffusion import HopDiffusion
from simulations.nanodomain import Nanodomain
from simulations.simulation import *
from simulations.spaceTimeCorrelationManager import SpaceTimeCorrelationManager
import util

from matplotlib.animation import FuncAnimation # type: ignore
from matplotlib.pyplot import figure
from matplotlib import colors
import matplotlib.pyplot as plt

import numpy as np
from matplotlib import rcParams # type: ignore

#path_colors: tuple[str, ...] = ('r', 'b', 'orange', 'g', 'y', 'c')
markers: tuple[str, ...] = ('o', 'v', '<', '>', 's', 'p')


def handle_nanodomain(ax: plt.Axes, sim: Nanodomain) -> None:
    nanodomains = [
        plt.Circle(
            *param,
            color='black',
            alpha=0.2,
        )
        for param in sim.get_nanodomain_attributes()
    ]
    for nanodomain in nanodomains:
        ax.add_patch(nanodomain)

def handle_hop_diffusion(ax: plt.Axes, sim: HopDiffusion) -> None:
    for param in sim.boundary_coordinates_for_plot:
        boundary = plt.Rectangle(
            tuple((param[0], param[1])),
            param[2], param[3],
            color='black',
            alpha=0.7,
            clip_on=False,
        )
        ax.add_patch(boundary)

def get_coordinates_for_plot(sim: Simulation, idx: int):
    return util.get_x_coordinates(sim.paths[idx]), util.get_y_coordinates(sim.paths[idx])

def get_coordinates_for_heads(sim, idx: int):
    return util.get_last_point(sim.paths[idx])

def get_matrix_for_plot(spc_manager: SpaceTimeCorrelationManager):
    return spc_manager.calculate_matrix()


class PlotGenerator:
    def __init__(self, sim: Simulation, spc_manager: SpaceTimeCorrelationManager):
        self.fig, self.ax = plt.subplots(1, 2, figsize = [9, 5], dpi = DPI) # type: ignore
        self.sim = sim
        self.spc_manager = spc_manager

        path_colors = [
            colors.to_hex(util.get_random_gray_shade()) for _ in range(sim.n_particles)
        ]
        
        self.path_plots = [
            self.ax[0].plot(
                *get_coordinates_for_plot(sim, i),
                markersize=15, color = path_colors[i])[0]
            for i in range(sim.n_particles)
        ]

        self.head_plots = [
            self.ax[0].plot(
                *get_coordinates_for_heads(sim, i),
                markersize=7, color = path_colors[i], marker = 'o',
                markerfacecolor="white")[0]
            for i in range(sim.n_particles)
        ]
        
        self.matrix = self.ax[1].imshow(
            get_matrix_for_plot(spc_manager),
            cmap = "viridis", interpolation = "none",
            aspect = "auto", origin = "lower"
        )
        self.fig.colorbar(self.matrix, ax = self.ax[1])

    def set_plot_parameters(self):
        self.ax[0].tick_params(axis = 'y', direction = "in", right = True, labelsize = 16, pad = 20)
        self.ax[0].tick_params(axis = 'x', direction = "in", top = True, bottom = True, labelsize = 16, pad = 20)

        ## legends and utilities
        self.ax[0].set_xlabel(r"nm", fontsize=16)
        self.ax[0].set_ylabel(r"nm", fontsize=16)

        ## border colors
        self.ax[0].patch.set_edgecolor('black')
        self.ax[0].patch.set_linewidth(2)

        self.ax[0].set_xlim(-RADIUS, RADIUS)
        self.ax[0].set_ylim(-RADIUS, RADIUS)

    def initialize_animation(self):
        self.set_plot_parameters()
        if isinstance(self.sim, Nanodomain): handle_nanodomain(self.ax[0], self.sim)
        elif isinstance(self.sim, HopDiffusion): handle_hop_diffusion(self.ax[0], self.sim)
        return self.path_plots

    def update_animation(self, *args):
        self.sim.update()
        for i, axes in enumerate(self.path_plots):
            coords = get_coordinates_for_plot(self.sim, i)
            axes.set_data(*coords)
        for i, head_marker in enumerate(self.head_plots):
            coords = get_coordinates_for_heads(self.sim, i)
            head_marker.set_data(*coords)
        self.matrix.set_data(get_matrix_for_plot(self.spc_manager))
        self.spc_manager.reset_local_matrix()
        return self.path_plots

    def start_animation(self):
        self.animation = FuncAnimation(
            fig = self.fig,
            func = self.update_animation,
            init_func = self.initialize_animation,
            interval = 100
        )

        plt.show(block = True) # type: ignore
        self.fig.tight_layout()
