from simulations.brownian import Brownian
from simulations.hopDiffusion import HopDiffusion
from simulations.nanodomain import Nanodomain
from simulations.simulation import *
from simulations.imageManager import ImageManager
from simulations.spaceCorrelationManager import SpaceCorrelationManager

from matplotlib.animation import FuncAnimation # type: ignore
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable # type: ignore
from matplotlib.pyplot import figure
from matplotlib import colors, cm
from scipy.optimize import curve_fit, fsolve

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy.typing as np_t
import numpy as np
import util

FloatMatrix = np_t.NDArray[np.float32]
path_colors2: tuple[str, ...] = ('r', 'b', 'orange', 'g', 'y', 'c', 'tan', 'lime', 'brown', 'navy')
markers: tuple[str, ...] = ('o', 'v', '<', '>', 's', 'p')

NM_IN_BETWEEN_AXIS_TICKS = 800
CONFINEMENT_WIDTH = 700 # in nm
LASER_RADIUS = 350 # in nm
N_PIXEL = 32
CMAP = colors.LinearSegmentedColormap.from_list(
    'my_colormap', ['black','green','white'], 128
)

ANIMATION_FRAMES: int = 5
ANIMATION_INTERVAL: int = int(TIME_PER_FRAME * 1000) # second to millisecond

def handle_landscape_nanodomain(ax: plt.Axes, sim: Nanodomain) -> None:
    nanodomains = [
        plt.Circle(
            *param,
            color = 'black',
            alpha = 0.2,
        )
        for param in sim.get_nanodomain_attributes()
    ]
    for nanodomain in nanodomains:
        ax.add_patch(nanodomain)

def handle_landscape_hop_diffusion(ax: plt.Axes, sim: HopDiffusion) -> None:
    for param in sim.boundary_coordinates_for_plot:
        boundary = plt.Rectangle(
            tuple((param[0], param[1])),
            param[2], param[3],
            color = 'black',
            alpha = 0.7,
            clip_on = False,
        )
        ax.add_patch(boundary)

def get_coordinates_for_landscape_plot(sim: Simulation, idx: int):
    return util.get_x_coordinates(sim.paths[idx]), util.get_y_coordinates(sim.paths[idx])

def get_coordinates_for_landscape_head_markers(sim, idx: int):
    return util.get_last_point(sim.paths[idx])

def get_image_matrix(image_manager: ImageManager):
    return image_manager.calculate_matrix()

def is_animation_ended(frame_number: int) -> bool:
    return frame_number + 1 == ANIMATION_FRAMES

def is_animation_frame_at_critical_number(frame_number: int) -> bool:
    return (frame_number == 0 or 
            frame_number == ANIMATION_FRAMES / 2 or 
            frame_number == ANIMATION_FRAMES - 1)
    
def gaussian(xy, amplitude, x0, y0, sigma_x, sigma_y, offset):
    x, y = xy
    x_kernel =  (x - x0) ** 2 / (2 * sigma_x ** 2 )
    y_kernel =  (y - y0) ** 2 / (2 * sigma_y ** 2 )
    return amplitude * np.exp(-(x_kernel + y_kernel)) + offset

def decayed_gaussian(xy, amplitude, x0, y0, sigma_x, sigma_y, offset) -> float:
    return gaussian(xy, amplitude, x0, y0, sigma_x, sigma_y, offset) - amplitude / np.exp(1)

def hyperbolic_fit(x, amplitude, tau, offset):
    return (amplitude / (1 + (x / tau))) + offset

def hyperbolic_error(x, x_error, a, a_error, tau, tau_error, offset, offset_error):
    dfdx = (-a * tau) / (tau + x) ** 2
    dfda = tau / (tau + x)
    dfdt = (a * x) / (tau + x) ** 2
    dfdo = 1
    return np.sqrt(
        (dfdx * x_error) ** 2 + (dfda * a_error) ** 2 +\
        (dfdt * tau_error) ** 2 + (dfdo * offset_error) ** 2
    )

def get_STICS_diffusion_coefficient(tau, tau_error, omega, omega_error) -> tuple[float, float]:
    D = omega ** 2 / (4 * tau)
    D_error = np.sqrt(((omega / ( 2 * tau)) * omega_error) ** 2 + ((-omega ** 2 / (4 * tau ** 2)) * tau_error) ** 2)
    
    return D, D_error

def get_IMSD_radius(x0, y0, amplitude, offset, sigma_x):
    xe = np.sqrt(-2 * np.log(1 / np.exp(1) - offset / amplitude) * sigma_x ** 2) + x0
    ye = y0
    return np.abs(xe - x0)

def IMSD_unconfined_fit_func(t, D, sigma_0_squared) -> float:
    return 4 * D * t + sigma_0_squared
    
def IMSD_confined_fit_func(t, L, tau_c, D, sigma_0_squared) -> float:
    return (L ** 2 / 3) * (1 - np.exp(- t / tau_c)) + 4 * D * t + sigma_0_squared


class PlotGenerator:
    def __init__(self, sim: Simulation, image_manager: ImageManager):
        self.fig, self.ax = plt.subplots(1, 2, figsize = [10, 5], dpi = DPI, gridspec_kw={'wspace' : 0.2}) # type: ignore
        self.STICS_fig, self.STICS_axins = None, None # deifne later 
        
        self.sim = sim
        self.image_manager = image_manager
        self.spc_manager = None 

        self.path_colors = [
            colors.to_hex(util.get_random_gray_shade()) for _ in range(sim.n_particles)
        ]
        self.path_plots, self.head_plots, self.matrix = self.generate_landscape_figure_elements()
        self.adjust_landscape_colorbar()
        self.transform_image_axes()

    def initialize_STICS_figure_and_axins(self) -> None:    
        self.STICS_fig = plt.figure(figsize = [15, 5], dpi = DPI) # type: ignore    
        
        ax1 = self.STICS_fig.add_subplot(1, 3, 1, projection='3d')
        ax2 = self.STICS_fig.add_subplot(1, 3, 2)
        ax3 = self.STICS_fig.add_subplot(1, 3, 3)
        
        self.STICS_axins = list([ax1, ax2, ax3])
    
    def generate_landscape_figure_elements(self):
        path_plots = [
            self.ax[0].plot(
                *handle_landscape_nanodomain(self.sim, i),
                markersize = 15, color = self.path_colors[i])[0]
            for i in range(self.sim.n_particles)
        ]
        head_plots = [
            self.ax[0].plot(
                *get_coordinates_for_landscape_head_markers(self.sim, i),
                markersize = 9, color = self.path_colors[i], marker = 'o',
                markerfacecolor = "black")[0]
            for i in range(self.sim.n_particles)
        ]
        matrix = self.ax[1].imshow(
            get_image_matrix(self.image_manager),
            cmap = CMAP, interpolation = "gaussian",
            aspect = "auto", origin = "lower"
        )
        return path_plots, head_plots, matrix
    
    def adjust_landscape_colorbar(self):
        divider = make_axes_locatable(self.ax[1])
        cax = divider.append_axes('right', size = "5%", pad = 0.1)
        self.fig.colorbar(self.matrix, cax = cax)
    
    def transform_image_axes(self):
        self.ax[1].tick_params(axis = 'y', labelsize = 16)
        self.ax[1].tick_params(axis = 'x', labelsize = 16, pad = 17.5)
        
    def set_landscape_plot_parameters(self):
        self.ax[0].tick_params(axis = 'y', direction = "in", right = True, labelsize = 16, pad = 20)
        self.ax[0].tick_params(axis = 'x', direction = "in", top = True, bottom = True, labelsize = 16, pad = 20)
        self.ax[0].set_xticks([-RADIUS + (NM_IN_BETWEEN_AXIS_TICKS * _) for _ in range(5)])
        self.ax[0].set_yticks([-RADIUS + (NM_IN_BETWEEN_AXIS_TICKS * _) for _ in range(5)])
        
        ## legends and utilities
        for ax in self.ax:
            ax.set_xlabel(r"nm", fontsize = 15)
            ax.set_ylabel(r"nm", fontsize = 15)

        ## border colors
        self.ax[0].patch.set_edgecolor('black')
        self.ax[0].patch.set_linewidth(2)

        self.ax[0].set_xlim(-RADIUS, RADIUS)
        self.ax[0].set_ylim(-RADIUS, RADIUS)

    def save_figure_at_critical_frame_number(self, fig: plt.figure, do_save: bool, frame_number: int, isSTICS: bool):
        fig_type_token = 'STICS' if isSTICS else 'Normal'
        sim_type_token = ''
        if isinstance(self.sim, Brownian): sim_type_token = 'Brownian'
        elif isinstance(self.sim, Nanodomain): sim_type_token = 'Nanodomain'
        elif isinstance(self.sim, HopDiffusion): sim_type_token = 'HopDiffusion'
        
        if not do_save: return
        
        file_name = f"data/figures/fig{fig_type_token}{sim_type_token}_{frame_number}.png"
        fig.savefig(file_name)

    def get_diffusion_through_curve_fit_for_IMSD(
        self, 
        frame_numbers: list[int], 
        imsd_list: list[float], 
        is_confined: bool
    ) -> tuple[np.ndarray, float, float]:
        D, D_error = 0, 0
        popt = None
        
        if is_confined:
            initial_guess = (0.1, 1.5, 9)
            popt, pcov = curve_fit(
                lambda t, tau_c, D, sigma_0_squared: 
                    IMSD_confined_fit_func(
                        t, 
                        CONFINEMENT_WIDTH, 
                        tau_c, 
                        D, 
                        sigma_0_squared
                    ),
                frame_numbers,
                imsd_list, 
                p0 = initial_guess,
                maxfev = 5000
            )
            D, D_error = popt[1], pcov[1][1] ** 0.5
        else:
            initial_guess = (1.5, 25)
            popt, pcov = curve_fit(
                IMSD_unconfined_fit_func,
                frame_numbers, imsd_list, p0 = initial_guess
            )
            D, D_error = popt[0], pcov[0][0] ** 0.5
        return popt, D, D_error
    
    def show_IMSD_plot(
        self, 
        ax: plt.Axes , 
        imsd_list: list[float], 
        is_confined: bool
    ) -> None:
        n_peaks = len(imsd_list)
        scatter_plot_frame_numbers = np.arange(0, n_peaks, 1)
        fit_linspace = np.linspace(0, n_peaks, 50)
        
        popt, D, D_error = self.get_diffusion_through_curve_fit_for_IMSD(
            scatter_plot_frame_numbers, imsd_list, is_confined
        )
        print(f'D is {D} , D_error {D_error}')
        
        ax.plot(
            scatter_plot_frame_numbers,
            imsd_list, 
            'ko', 
            label = 'iMSD radii'
        )
        ax.plot(
            fit_linspace,
            IMSD_confined_fit_func(fit_linspace, CONFINEMENT_WIDTH, *popt) if is_confined else IMSD_unconfined_fit_func(fit_linspace, *popt), 
            'r--',
            label = 'Confined fit' if is_confined else 'Unconfined fit'
        )
        self.prettify_STICS_complementary_plot(ax, True)
        
    def prettify_STICS_complementary_plot(self, ax: plt.Axes, is_imsd_plot: bool):
        ax.legend(loc = 'upper right')
        ax.set_xlabel(r"$\tau \ (s)$", fontsize = 14)
        ax.set_ylabel(
            r"$\sigma_r^2(\tau)$" if is_imsd_plot else r"$G(0, 0, \tau)$",
            fontsize = 14
        )
        formatted_ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * TIME_PER_FRAME))
        ax.xaxis.set_major_formatter(formatted_ticks)
        ax.patch.set_edgecolor('black')
        ax.patch.set_linewidth(2)
        ax.tick_params(axis = 'y', direction = "in", right = True, labelsize = 16, pad = 20)
        ax.tick_params(axis = 'x', direction = "in", top = True, bottom = True, labelsize = 16, pad = 20)
        
    def show_peak_decay_plots(
        self, 
        ax: plt.Axes, 
        peak_decay_list: list[float], 
        peak_decay_list_error: list[float]
    ) -> None:
        n_peaks = len(peak_decay_list)
        scatter_plot_frame_numbers = np.arange(0, n_peaks, 1)
        hyperbolic_fit_linspace = np.linspace(0, n_peaks, 50)
        
        initial_guess = (0.02, 0)
        popt, pcov = curve_fit(
            lambda x, tau, offset: 
                hyperbolic_fit(
                    x, 
                    peak_decay_list[0], 
                    tau, 
                    offset
                ), 
            scatter_plot_frame_numbers, 
            peak_decay_list, 
            p0 = initial_guess
        )
        
        tau = popt[0]
        tau_error = pcov[0][0] ** 0.5
        D, D_error = get_STICS_diffusion_coefficient(tau, tau_error, LASER_RADIUS, 0)
        print(f'D_stics is {D}, D_stics_error is {D_error}')
        
        ax.errorbar(
            x = scatter_plot_frame_numbers,
            y = peak_decay_list, 
            xerr = np.repeat(0, (len(scatter_plot_frame_numbers))),
            yerr = peak_decay_list_error,
            fmt = 'ko', 
            label = 'STICS function peaks'
        )
        ax.plot(
            hyperbolic_fit_linspace,
            hyperbolic_fit(hyperbolic_fit_linspace, peak_decay_list[0], *popt), 
            'b--',
            label = 'hyperbolic fit'
        )
        self.prettify_STICS_complementary_plot(ax, False)

    def show_STICS_plot(self, ax: plt.Axes, x, y, data: FloatMatrix) -> list[plt.Axes]:
        plot = [
            ax.plot_trisurf(
                x.flatten(), 
                y.flatten(), 
                data.flatten(), 
                cmap=cm.jet, linewidth = 0.75, edgecolor = 'black'
            )
        ]
        
        ax.set_xlabel(r"$\zeta$", fontsize = 14)
        ax.set_ylabel(r"$\eta$", fontsize = 14)
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_axis_off()
        
        return plot
    
    def compute_IMSD_radius(self, popt: list):
        amplitude, x0, y0, sigma_x, sigma_y, offset = popt
        
        sigma = get_IMSD_radius(x0, y0, amplitude, offset, sigma_x)
        self.spc_manager.update_imsd_list(sigma)
    
    def calculate_STICS_curve_fit(self, data_amplitude, x, y, data) -> tuple[float, float]:
        initial_guess = (data_amplitude, N_PIXEL / 2, N_PIXEL / 2, 5, 5, 0)
        xdata = np.vstack((x.ravel(), y.ravel()))
        
        popt, pcov = curve_fit(
            gaussian, 
            xdata, 
            data.flatten(), 
            p0 = initial_guess, 
            sigma = np.repeat(5, len(data.flatten()))
        )
        
        self.compute_IMSD_radius(popt)
        
        return popt[0], (pcov[0][0] ** 0.5)
    
    def show_STICS_complementary_plots_if_animation_is_ended(self, frame_number: int) -> None:
        if is_animation_ended(frame_number):
            self.show_peak_decay_plots(
                self.STICS_axins[1], 
                self.spc_manager.get_peak_decay_list(), 
                self.spc_manager.get_peak_decay_list_error()
            )
            self.show_IMSD_plot(
                self.STICS_axins[2], 
                self.spc_manager.get_imsd_list(),
                False
            )
                
    def set_STICS_axin_range_limit(self, amplitude_of_STICS_function_at_first_frame: float) -> None:
        self.STICS_axins[0].set_zlim(0, amplitude_of_STICS_function_at_first_frame + 2)
        
    def start_STICS_animation(self) -> None:
        plt.close()
        frames = self.spc_manager.get_corr_function_frames
        current_frame = frames[0]    
        amplitude_of_STICS_function_at_first_frame = current_frame.max()
        
        self.initialize_STICS_figure_and_axins()
        self.set_STICS_axin_range_limit(amplitude_of_STICS_function_at_first_frame)

        X_cells, Y_cells = np.meshgrid( range(len(current_frame)), range(len(current_frame[0])))  
        STICS_surface_plots = self.show_STICS_plot(self.STICS_axins[0], X_cells, Y_cells, current_frame)
        
        def update_STICS_animation(frame_number): 
            current_frame = frames[frame_number] 
            STICS_surface_plots[0].remove()
            STICS_surface_plots[0] = self.show_STICS_plot(self.STICS_axins[0], X_cells, Y_cells, current_frame)[0]
            
            amplitude, amplitude_error = self.calculate_STICS_curve_fit(
                amplitude_of_STICS_function_at_first_frame, 
                X_cells, 
                Y_cells, 
                current_frame
            )
            
            self.spc_manager.update_peak_decay_list(amplitude, amplitude_error)
            
            if is_animation_frame_at_critical_number(frame_number):
                self.show_STICS_complementary_plots_if_animation_is_ended(frame_number)
                self.save_figure_at_critical_frame_number(self.STICS_fig, True, frame_number, True)
            return frames
        
        def initialize_STICS_animation(): return frames
        
        animation = FuncAnimation(
            fig = self.STICS_fig,
            func = update_STICS_animation,
            init_func = initialize_STICS_animation,
            interval = ANIMATION_INTERVAL,
            frames = ANIMATION_FRAMES,
            repeat = False
        )
        plt.show() # type: ignore
        
    def initialize_landscape_animation(self):
        self.set_landscape_plot_parameters()
        if isinstance(self.sim, Nanodomain): handle_landscape_nanodomain(self.ax[0], self.sim)
        elif isinstance(self.sim, HopDiffusion): handle_landscape_hop_diffusion(self.ax[0], self.sim)
        return self.path_plots

    def update_landscape_animation(self, frame_number):
        print(frame_number)
        if is_animation_ended(frame_number):
            self.spc_manager = SpaceCorrelationManager(self.image_manager)
        
        self.sim.update()
        for i, axes in enumerate(self.path_plots):
            coords = handle_landscape_nanodomain(self.sim, i)
            axes.set_data(*coords)
        for i, head_marker in enumerate(self.head_plots):
            coords = get_coordinates_for_landscape_head_markers(self.sim, i)
            head_marker.set_data(*coords)
            
        self.matrix.set_data(get_image_matrix(self.image_manager))
        self.image_manager.increment_image_counter()
        
        if frame_number == 0 or frame_number == ANIMATION_FRAMES / 2 or frame_number == ANIMATION_FRAMES - 1:
            self.save_figure_at_critical_frame_number(self.fig, True, frame_number, False)
        
        return self.path_plots

    def start_animation(self):
        self.animation = FuncAnimation(
            fig = self.fig,
            func = self.update_landscape_animation,
            init_func = self.initialize_landscape_animation,
            interval = ANIMATION_INTERVAL,
            frames = ANIMATION_FRAMES,
            repeat = False
        )

        plt.show() # type: ignore
        self.fig.tight_layout()