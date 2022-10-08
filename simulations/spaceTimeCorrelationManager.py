import random
import numpy as np
from enum import Enum
from simulations.simulation import *

N_VOXEL = 24
WIDTH = 2 * RADIUS
VOXEL_SIZE = WIDTH / N_VOXEL # 24 x 24 voxels

class SpaceTimeCorrelationManager(Simulation):
    def __init__(self, sim: Simulation) -> None:
        self.sim = sim
        self.matrix = [[0 for _ in range(N_VOXEL)] for _ in range(N_VOXEL)]
        
    def calculate_matrix(self) -> list[list[float]]:
        occupied_squares: set[tuple[int, int]] = set()
        
        for i in range(self.sim.n_particles):
            x, y = self.sim.paths[i][-1]
            x += RADIUS
            y += RADIUS
            PIXEL_X = int(x // VOXEL_SIZE)
            PIXEL_Y = int(y // VOXEL_SIZE)
            self.matrix[PIXEL_X][PIXEL_Y] += 1
            
            occupied_squares.add(tuple[PIXEL_X, PIXEL_Y])
        
        for i in range(N_VOXEL):
            for j in range(N_VOXEL):
                pair = tuple[i, j]
                if pair not in occupied_squares:
                    pass
                else:
                    self.matrix[i][j] = self.matrix[i][j] - (self.matrix[i][j] / N_VOXEL ** 2)
        return self.matrix
    
    def reset_local_matrix(self) -> None:
        self.matrix = [[0 for _ in range(N_VOXEL)] for _ in range(N_VOXEL)]