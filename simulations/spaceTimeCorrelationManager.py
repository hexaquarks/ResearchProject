import random
import numpy as np
import numpy.typing as np_t
from enum import Enum
from simulations.simulation import *
from scipy.ndimage import gaussian_filter

N_VOXEL = 128
WIDTH = 2 * RADIUS
VOXEL_SIZE = WIDTH / N_VOXEL # N_VOXEL x N_VOXEL voxels

class SpaceTimeCorrelationManager(Simulation):
    def __init__(self, sim: Simulation) -> None:
        self.sim = sim
        self.matrix: np_t.NDArray[np.float64] = np.ndarray([[0. for _ in range(N_VOXEL)] for _ in range(N_VOXEL)], dtype = np.float64)
        
    def is_out_of_bounds(self, pos: tuple[int, int]) -> bool:
        x, y = pos[0], pos[1]
        return x < -RADIUS or x > RADIUS or y > RADIUS or y < -RADIUS
    
    def apply_convolution_filter(self) -> None:
        self.matrix = gaussian_filter(self.matrix, sigma = 10) 
    
    def apply_gaussian_noise(self) -> None:
        noise_delta = np.random.normal(0, .1, self.matrix.shape)
        self.matrix += noise_delta
        
    def resize_matrix_for_proper_convolution(self, x: float, y: float) -> np_t.NDArray[np.float64]:
        value_to_extend_x = int(x / VOXEL_SIZE) + (x % VOXEL_SIZE > 0)
        value_to_extend_y = int(y / VOXEL_SIZE) + (y % VOXEL_SIZE > 0)
        
        directions_extended: tuple[int, int] = (0, 0)
        new_mat = self.matrix
        
        if value_to_extend_x < 0:
            new_cols = [[0 for _ in range(-1 * value_to_extend_x)] for _ in range(N_VOXEL)]
            new_mat = np.append(new_cols, self.matrix, axis = 1)
            directions_extended[0] -= value_to_extend_x
        elif value_to_extend_x >= N_VOXEL:
            new_cols = [[0 for _ in range(value_to_extend_x)] for _ in range(N_VOXEL)]
            new_mat = np.append(self.matrix, new_cols, axis = 1)
            directions_extended[0] += value_to_extend_x
        if value_to_extend_y < 0:
            new_rows = [[0 for _ in range(N_VOXEL)] for _ in range(-1 * value_to_extend_y)]
            new_mat = np.vstack([new_rows, self.matrix])
            directions_extended[1] -= value_to_extend_y
        elif value_to_extend_y >= N_VOXEL:
            new_rows = [[0 for _ in range(N_VOXEL)] for _ in range(value_to_extend_y)]
            new_mat = np.vstack([self.matrix, new_rows])
            directions_extended[1] += value_to_extend_y
            
        return new_mat
    
    def calculate_matrix(self) -> list[list[float]]:
        occupied_squares: set[tuple[int, int]] = set()
        
        for i in range(self.sim.n_particles):
            x, y = self.sim.get_last_particle_coordinate(i)
            if (self.is_out_of_bounds((x, y))): continue
            
            x += RADIUS
            y += RADIUS
            self.resize_matrix_for_proper_convolution(x, y)
                
            PIXEL_X = int(x // VOXEL_SIZE)
            PIXEL_Y = int(y // VOXEL_SIZE)
            self.matrix[PIXEL_Y][PIXEL_X] += 2000.
            
            occupied_squares.add(tuple[PIXEL_X, PIXEL_Y])
        
        for i in range(N_VOXEL):
            for j in range(N_VOXEL):
                pair = tuple[i, j]
                if pair not in occupied_squares:
                    pass
                else:
                    pass
                    # self.matrix[i][j] = self.matrix[i][j] - (self.matrix[i][j] / N_VOXEL ** 2)
                    
        self.apply_convolution_filter()
        self.apply_gaussian_noise()
        # note that a larger sigma induces a larger kernel such that
        # the pixel gets blurred over a wider distance
        
        return self.matrix
    
    def reset_local_matrix(self) -> None:
        self.matrix = np.ndarray([[0. for _ in range(N_VOXEL)] for _ in range(N_VOXEL)], dtype = np.float64)