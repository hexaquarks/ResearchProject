import matplotlib.tri as mtri
from simulations.hopDiffusion import HopDiffusion
from simulations.nanodomain import Nanodomain
from simulations.simulation import *
from simulations.imageManager import ImageManager
from simulations.spaceCorrelationManager import SpaceCorrelationManager

from matplotlib.animation import FuncAnimation # type: ignore
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable # type: ignore
from matplotlib.pyplot import figure
from matplotlib import colors, projections, cm
from mpl_toolkits.mplot3d import axes3d

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from simulations.spaceCorrelationManager import SpaceCorrelationManager
import util

path_colors2: tuple[str, ...] = ('r', 'b', 'orange', 'g', 'y', 'c', 'tan', 'lime', 'brown', 'navy')
markers: tuple[str, ...] = ('o', 'v', '<', '>', 's', 'p')

NM_IN_BETWEEN_AXIS_TICKS = 800
N_PIXEL = 32
CMAP = colors.LinearSegmentedColormap.from_list('my_colormap',
                                                    ['black','green','white'],
                                                    128)

ANIMATION_FRAMES: int = 200
ANIMATION_INTERVAL: int = int(TIME_PER_FRAME * 1000) # second to millisecond

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
        self.spc_manager = None 

        self.path_colors = [
            colors.to_hex(util.get_random_gray_shade()) for _ in range(sim.n_particles)
        ]
        self.path_plots, self.head_plots, self.matrix = self.generate_figure_elements()
        self.adjust_colorbar()
        self.transform_image_axes()

    def generate_figure_elements(self):
        path_plots = [
            self.ax[0].plot(
                *get_coordinates_for_plot(self.sim, i),
                markersize=15, color = self.path_colors[i])[0]
            for i in range(self.sim.n_particles)
        ]
        head_plots = [
            self.ax[0].plot(
                *get_coordinates_for_heads(self.sim, i),
                markersize=9, color = self.path_colors[i], marker = 'o',
                markerfacecolor="black")[0]
            for i in range(self.sim.n_particles)
        ]
        matrix = self.ax[1].imshow(
            get_matrix_for_plot(self.image_manager),
            cmap = CMAP, interpolation = "gaussian",
            aspect = "auto", origin = "lower"
        )
        return path_plots, head_plots, matrix
    
    def adjust_colorbar(self):
        divider = make_axes_locatable(self.ax[1])
        cax = divider.append_axes('right', size="5%", pad=0.1)
        self.fig.colorbar(self.matrix, cax = cax)
    
    def transform_image_axes(self):
        # self.ax[1].set_xticks([32 * _ for _ in range(5)])
        # self.ax[1].set_yticks([32 * _ for _ in range(5)])
        # self.ax[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: ('%d') % float(x * (2 * RADIUS / 128) - RADIUS)))
        # self.ax[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: ('%d') % float(x * (2 * RADIUS / 128) - RADIUS)))
        # self.ax[1].get_yaxis().set_visible(False)
        
        self.ax[1].tick_params(axis = 'y', labelsize = 16)
        self.ax[1].tick_params(axis = 'x', labelsize = 16, pad = 17.5)
        
        
    def set_plot_parameters(self):
        self.ax[0].tick_params(axis = 'y', direction = "in", right = True, labelsize = 16, pad = 20)
        self.ax[0].tick_params(axis = 'x', direction = "in", top = True, bottom = True, labelsize = 16, pad = 20)
        self.ax[0].set_xticks([-RADIUS + (NM_IN_BETWEEN_AXIS_TICKS * _) for _ in range(5)])
        self.ax[0].set_yticks([-RADIUS + (NM_IN_BETWEEN_AXIS_TICKS * _) for _ in range(5)])
        
        ## legends and utilities
        for ax in self.ax:
            ax.set_xlabel(r"nm", fontsize=15)
            ax.set_ylabel(r"nm", fontsize=15)

        ## border colors
        self.ax[0].patch.set_edgecolor('black')
        self.ax[0].patch.set_linewidth(2)

        self.ax[0].set_xlim(-RADIUS, RADIUS)
        self.ax[0].set_ylim(-RADIUS, RADIUS)

    def show_peak_decay_plots(self, ax: plt.Axes, peak_decay_list: list):
        n_peaks = len(peak_decay_list)
        scatter_plot_frame_numbers = np.arange(0, n_peaks, 1)
        
        polyfit_function = np.poly1d(np.polyfit(scatter_plot_frame_numbers, peak_decay_list, 3))
        polyfit_linspace = np.linspace(0, n_peaks, 50)
        ax.plot(
            scatter_plot_frame_numbers,
            peak_decay_list, 'ko', 
            polyfit_linspace,
            polyfit_function(polyfit_linspace), 'r--'
        )
        ax.set_xlabel(r"$\tau \ (s)$", fontsize = 14)
        ax.set_ylabel(r"$G(0, 0, \tau)$", fontsize = 14)
        formatted_ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * TIME_PER_FRAME))
        ax.xaxis.set_major_formatter(formatted_ticks)
        ax.patch.set_edgecolor('black')
        ax.patch.set_linewidth(2)
        ax.tick_params(axis = 'y', direction = "in", right = True, labelsize = 16, pad = 20)
        ax.tick_params(axis = 'x', direction = "in", top = True, bottom = True, labelsize = 16, pad = 20)
        
    
    def show_STICS_plot(self, ax, x, y, data) -> list[plt.Axes]:
        plot = [ax.plot_trisurf(x.flatten(), y.flatten(), data.flatten(), cmap=cm.jet,
                       linewidth = 0.75, edgecolor='black')]
        ax.set_xlabel(r"$\zeta$", fontsize = 14)
        ax.set_ylabel(r"$\eta$", fontsize = 14)
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_axis_off()
        return plot
        
    def initialize_space_correlation_manager(self) -> None:
        plt.close()
        frames = self.spc_manager.get_corr_function_frames
        peak_decay_list = self.spc_manager.get_peak_decay_list()
        
        fig = plt.figure(figsize = [10, 5], dpi = DPI) # type: ignore
        
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.set_zlim(0, max(peak_decay_list) + 2)

        data = frames[0]    
        X, Y = np.meshgrid(
            range(len(frames[0])),
            range(len(frames[0][0]))
        )  
        plot_corr = self.show_STICS_plot(ax1, X, Y, data)
        self.show_peak_decay_plots(ax2, peak_decay_list)
        
        def update_STICS_animation(frame_number): 
            print(frame_number) 
            data = frames[frame_number] 
            plot_corr[0].remove()
            plot_corr[0] = self.show_STICS_plot(ax1, X, Y, data)[0]
                
            if frame_number == 0 or frame_number == ANIMATION_FRAMES / 2 or frame_number == ANIMATION_FRAMES - 1:
                fig.savefig(fname=f"data/figures/figSTICSHop_{frame_number}.png")
            return frames
        
        def initialize_STICS_animation(): return frames
        
        animation = FuncAnimation(
            fig = fig,
            func = update_STICS_animation,
            init_func = initialize_STICS_animation,
            interval = ANIMATION_INTERVAL,
            frames = ANIMATION_FRAMES,
            repeat = False
        )
        plt.show() # type: ignore
        
    def initialize_animation(self):
        self.set_plot_parameters()
        if isinstance(self.sim, Nanodomain): handle_nanodomain(self.ax[0], self.sim)
        elif isinstance(self.sim, HopDiffusion): handle_hop_diffusion(self.ax[0], self.sim)
        return self.path_plots

    def update_animation(self, frame_number):
        print(frame_number)
        if frame_number + 1 == ANIMATION_FRAMES: 
            self.spc_manager = SpaceCorrelationManager(self.image_manager)
            #util.export_images_to_tiff(self.image_manager.images_without_background)
            
        self.sim.update()
        for i, axes in enumerate(self.path_plots):
            coords = get_coordinates_for_plot(self.sim, i)
            axes.set_data(*coords)
        for i, head_marker in enumerate(self.head_plots):
            coords = get_coordinates_for_heads(self.sim, i)
            head_marker.set_data(*coords)
            
        self.matrix.set_data(get_matrix_for_plot(self.image_manager))
        self.image_manager.increment_image_counter()
        
        if frame_number == 0 or frame_number == ANIMATION_FRAMES / 2 or frame_number == ANIMATION_FRAMES - 1:
            self.fig.savefig(fname=f"data/figures/figNormalHop_{frame_number}.png")
        
        return self.path_plots

    def start_animation(self):
        self.animation = FuncAnimation(
            fig = self.fig,
            func = self.update_animation,
            init_func = self.initialize_animation,
            interval = ANIMATION_INTERVAL,
            frames = ANIMATION_FRAMES,
            repeat = False
        )

        plt.show() # type: ignore
        #plt.pause(5)
        self.fig.tight_layout()