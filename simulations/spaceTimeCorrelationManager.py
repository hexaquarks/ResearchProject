import random
import numpy as np
import numpy.typing as np_t
from enum import Enum
from simulations.simulation import *
from scipy.ndimage import gaussian_filter

N_VOXEL = 128
WIDTH = 2 * RADIUS
VOXEL_SIZE = WIDTH / N_VOXEL # N_VOXEL x N_VOXEL voxels
MATRIX_EXTENSION_FACTOR = N_VOXEL / 14 # extend the matrix by 10% of N_VOXEL all directions
SIZE_OF_COLS_AND_ROWS = 14

class SpaceTimeCorrelationManager(Simulation):
    def __init__(self, sim: Simulation) -> None:
        self.sim = sim
        self.matrix: np_t.NDArray[np.float32] = np.array([
            [0. for _ in range(N_VOXEL + 2 * SIZE_OF_COLS_AND_ROWS)] 
            for _ in range(N_VOXEL + 2 * SIZE_OF_COLS_AND_ROWS)], 
            dtype = np.float32
        )
        
    def is_out_of_extended_bounds(self, pos: tuple[int, int]) -> bool:
        x, y = pos[0], pos[1]
        factor = SIZE_OF_COLS_AND_ROWS * WIDTH / N_VOXEL
        new_radius = RADIUS * factor
        return x < -new_radius or x > new_radius or y > new_radius or y < -new_radius
    
    def apply_convolution_filter(self) -> None:
        self.matrix = gaussian_filter(self.matrix, sigma = 15) 
    
    def apply_gaussian_noise(self) -> None:
        noise_delta = np.random.normal(0, .1, self.matrix.shape)
        self.matrix += noise_delta
        
    def calculate_matrix(self) -> list[list[float]]:
        # quick fix
        if (len(self.matrix) != N_VOXEL + 2 * SIZE_OF_COLS_AND_ROWS): self.reset_local_matrix()
        
        for i in range(self.sim.n_particles):
            x, y = self.sim.get_last_particle_coordinate(i)
            if (self.is_out_of_extended_bounds((x, y))): continue
            
            x += RADIUS
            y += RADIUS
            PIXEL_X = int(int(x // VOXEL_SIZE) + SIZE_OF_COLS_AND_ROWS)
            PIXEL_Y = int(int(y // VOXEL_SIZE) + SIZE_OF_COLS_AND_ROWS)
            self.matrix[PIXEL_Y][PIXEL_X] += 2600.
            
        self.apply_convolution_filter()
        self.apply_gaussian_noise()
        self.matrix = self.matrix[SIZE_OF_COLS_AND_ROWS:-SIZE_OF_COLS_AND_ROWS, SIZE_OF_COLS_AND_ROWS:-SIZE_OF_COLS_AND_ROWS]
        
        # note that a larger sigma induces a larger kernel such that
        # the pixel gets blurred over a wider distance
        return self.matrix
    
    def reset_local_matrix(self) -> None:
        self.matrix: np_t.NDArray[np.float32] = np.array([
            [0. for _ in range(N_VOXEL + 2 * SIZE_OF_COLS_AND_ROWS)] 
            for _ in range(N_VOXEL + 2 * SIZE_OF_COLS_AND_ROWS)], 
            dtype = np.float32
        )