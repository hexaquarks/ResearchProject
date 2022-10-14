import random
import numpy as np
from enum import Enum
from simulations.simulation import *
from scipy.ndimage import gaussian_filter

N_VOXEL = 128
WIDTH = 2 * RADIUS
VOXEL_SIZE = WIDTH / N_VOXEL # N_VOXEL x N_VOXEL voxels

class SpaceTimeCorrelationManager(Simulation):
    def __init__(self, sim: Simulation) -> None:
        self.sim = sim
        self.matrix: list[list[float]] = [[0. for _ in range(N_VOXEL)] for _ in range(N_VOXEL)]
        
    def is_out_of_bounds(self, pos: tuple[int, int]) -> bool:
        x, y = pos[0], pos[1]
        return x < -RADIUS or x > RADIUS or y > RADIUS or y < -RADIUS
    
    def apply_convolution_filter(self) -> None:
        self.matrix = gaussian_filter(self.matrix, sigma = 10) 
    
    def apply_gaussian_noise(self) -> None:
        noise_delta = np.random.normal(0, .1, self.matrix.shape)
        self.matrix += noise_delta
        
    def calculate_matrix(self) -> list[list[float]]:
        occupied_squares: set[tuple[int, int]] = set()
        
        for i in range(self.sim.n_particles):
            x, y = self.sim.get_last_particle_coordinate(i)
            if (self.is_out_of_bounds((x, y))): continue
            
            x += RADIUS
            y += RADIUS
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
        self.matrix = [[0 for _ in range(N_VOXEL)] for _ in range(N_VOXEL)]