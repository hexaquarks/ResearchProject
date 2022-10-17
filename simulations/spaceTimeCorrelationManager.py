import random
import numpy as np
import numpy.typing as np_t
from simulations.simulation import *
from scipy.ndimage import gaussian_filter

N_PIXEL = 128
WIDTH = 2 * RADIUS
VOXEL_SIZE = WIDTH / N_PIXEL # N_PIXEL x N_PIXEL voxels
NUMBER_OF_COLS_OR_ROWS_TO_EXTEND = 14 # arbitrary
CONVOLUTION_SIGMA = 15 # note that a larger value yields a wider spread of the intensity
PARTICLE_INTENSITY = 2800 # adjusted aesthetically

class SpaceTimeCorrelationManager(Simulation):
    def __init__(self, sim: Simulation) -> None:
        self.sim = sim
        self.matrix: np_t.NDArray[np.float32]
        self.reset_local_matrix()
         
    def is_out_of_extended_bounds(self, pos: tuple[int, int]) -> bool:
        x, y = pos[0], pos[1]
        max_index = N_PIXEL + 2 * NUMBER_OF_COLS_OR_ROWS_TO_EXTEND
        return x < 0 or x >= max_index or y < 0 or y >= max_index
    
    def apply_convolution_filter(self) -> None:
        self.matrix = gaussian_filter(self.matrix, sigma = CONVOLUTION_SIGMA) 
    
    def apply_gaussian_noise(self) -> None:
        noise_delta = np.random.normal(0, .1, self.matrix.shape)
        self.matrix += noise_delta
        
    def trim_matrix_for_display(self) -> None: 
        val = NUMBER_OF_COLS_OR_ROWS_TO_EXTEND
        self.matrix = self.matrix[val:-val, val:-val]
        
    def get_pixel_coord(self, x: float) -> int:
        return int(x // VOXEL_SIZE) + NUMBER_OF_COLS_OR_ROWS_TO_EXTEND
    
    def calculate_matrix(self) -> np_t.NDArray[np.float32]:
        # quick fix
        if (len(self.matrix) != N_PIXEL + 2 * NUMBER_OF_COLS_OR_ROWS_TO_EXTEND): 
            self.reset_local_matrix()
        
        for i in range(self.sim.n_particles):
            x, y = self.sim.get_last_particle_coordinate(i)
            x, y = self.get_pixel_coord(x + RADIUS), self.get_pixel_coord(y + RADIUS)
            if (self.is_out_of_extended_bounds((x, y))): continue
            self.matrix[y][x] += PARTICLE_INTENSITY
            
        self.apply_convolution_filter()
        self.apply_gaussian_noise()
        self.trim_matrix_for_display()
        
        return self.matrix
    
    def reset_local_matrix(self) -> None:
        self.matrix: np_t.NDArray[np.float32] = np.array([
                [0. for _ in range(N_PIXEL + 2 * NUMBER_OF_COLS_OR_ROWS_TO_EXTEND)] 
                for _ in range(N_PIXEL + 2 * NUMBER_OF_COLS_OR_ROWS_TO_EXTEND)
            ], 
            dtype = np.float32
        )