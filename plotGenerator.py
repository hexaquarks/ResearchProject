from simulations.hopDiffusion import HopDiffusion
from simulations.nanodomain import Nanodomain
from simulations.simulation import *
from simulations.imageManager import ImageManager

from matplotlib.animation import FuncAnimation # type: ignore
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable # type: ignore
from matplotlib.pyplot import figure
from matplotlib import colors

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import util

path_colors2: tuple[str, ...] = ('r', 'b', 'orange', 'g', 'y', 'c')
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

def get_matrix_for_plot(image_manager: ImageManager):
    return image_manager.calculate_matrix()

class PlotGenerator:
    def __init__(self, sim: Simulation, image_manager: ImageManager):
        self.fig, self.ax = plt.subplots(1, 2, figsize = [10, 5], dpi = DPI, gridspec_kw={'wspace' : 0.2}) # type: ignore
        self.sim = sim
        self.image_manager = image_manager

        path_colors = [
            colors.to_hex(util.get_random_gray_shade()) for _ in range(sim.n_particles)
        ]
        self.path_plots, self.head_plots, self.matrix = self.generate_figure_elements()
        self.adjust_colorbar()
        self.transform_image_axes()

    def generate_figure_elements(self):
        path_plots = [
            self.ax[0].plot(
                *get_coordinates_for_plot(self.sim, i),
                markersize=15, color = path_colors2[i])[0]
            for i in range(self.sim.n_particles)
        ]
        head_plots = [
            self.ax[0].plot(
                *get_coordinates_for_heads(self.sim, i),
                markersize=7, color = path_colors2[i], marker = 'o',
                markerfacecolor="white")[0]
            for i in range(self.sim.n_particles)
        ]
        matrix = self.ax[1].imshow(
            get_matrix_for_plot(self.image_manager),
            cmap = "viridis", interpolation = "none",
            aspect = "auto", origin = "lower"
        )
        return path_plots, head_plots, matrix
    
    def adjust_colorbar(self):
        divider = make_axes_locatable(self.ax[1])
        cax = divider.append_axes('right', size="5%", pad=0.1)
        self.fig.colorbar(self.matrix, cax = cax)
    
    def transform_image_axes(self):
        self.ax[1].set_xticks([32 * _ for _ in range(5)])
        self.ax[1].set_yticks([32 * _ for _ in range(5)])
        self.ax[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: ('%d') % float(x * (2 * RADIUS / 128) - RADIUS)))
        self.ax[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: ('%d') % float(x * (2 * RADIUS / 128) - RADIUS)))
        self.ax[1].get_yaxis().set_visible(False)
        
        self.ax[1].tick_params(axis = 'y', labelsize = 16)
        self.ax[1].tick_params(axis = 'x', labelsize = 16, pad = 17.5)
        
        
    def set_plot_parameters(self):
        self.ax[0].tick_params(axis = 'y', direction = "in", right = True, labelsize = 16, pad = 20)
        self.ax[0].tick_params(axis = 'x', direction = "in", top = True, bottom = True, labelsize = 16, pad = 20)
        self.ax[0].set_xticks([-RADIUS + (375 * _) for _ in range(5)])
        self.ax[0].set_yticks([-RADIUS + (375 * _) for _ in range(5)])
        
        ## legends and utilities
        for ax in self.ax:
            ax.set_xlabel(r"nm", fontsize=15)
            ax.set_ylabel(r"nm", fontsize=15)

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
        self.matrix.set_data(get_matrix_for_plot(self.image_manager))
        self.image_manager.matrix = self.image_manager.reset_local_matrix(True)
        
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
