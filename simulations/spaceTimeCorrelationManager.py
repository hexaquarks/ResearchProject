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
        self.matrix = gaussian_filter(self.matrix, sigma = 10) 
    
    def apply_gaussian_noise(self) -> None:
        noise_delta = np.random.normal(0, .1, self.matrix.shape)
        self.matrix += noise_delta
        
    def resize_matrix_for_proper_convolution(
        self, x: float, y: float, directions_extended: tuple[int, int]
    ) -> tuple[np_t.NDArray[np.float64], tuple[int, int]]:
        
        value_to_extend_x = 0
        value_to_extend_y = 0
        
        if x > 0:
            value_to_extend_x = int(x / VOXEL_SIZE) + (x % VOXEL_SIZE > 0)
        else:
            value_to_extend_x = int(x // VOXEL_SIZE)
        if y > 0:
            value_to_extend_y = int(y / VOXEL_SIZE) + (y % VOXEL_SIZE > 0)
        else:
            value_to_extend_x = int(y // VOXEL_SIZE)
    
        new_mat = self.matrix
        
        if value_to_extend_x < 0:
            new_cols = [[0 for _ in range(-1 * value_to_extend_x)] for _ in range(N_VOXEL)]
            new_mat = np.append(new_cols, self.matrix, axis = 1)
            directions_extended = (directions_extended[0] - value_to_extend_x, directions_extended[1])
        elif value_to_extend_x >= N_VOXEL:
            new_cols = [[0 for _ in range(value_to_extend_x)] for _ in range(N_VOXEL)]
            new_mat = np.append(self.matrix, new_cols, axis = 1)
            directions_extended = (directions_extended[0] + value_to_extend_x, directions_extended[1])
        if value_to_extend_y < 0:
            new_rows = [[0 for _ in range(N_VOXEL)] for _ in range(-1 * value_to_extend_y)]
            new_mat = np.vstack([new_rows, self.matrix])
            directions_extended = (directions_extended[0], directions_extended[1] - value_to_extend_y)
        elif value_to_extend_y >= N_VOXEL:
            new_rows = [[0 for _ in range(N_VOXEL)] for _ in range(value_to_extend_y)]
            new_mat = np.vstack([self.matrix, new_rows])
            directions_extended = (directions_extended[0], directions_extended[1] + value_to_extend_y)
            
        # omg just extend all sides and calculate all space anyways...
        # transform at the end. No need for dynamic resizing. static is fine
        return new_mat, directions_extended
    
    def extend_matrix_size(self):
        # horizontal direction
        new_cols = [[0 for _ in range(SIZE_OF_COLS_AND_ROWS)] for _ in range(N_VOXEL)]
        self.matrix = np.append(new_cols, self.matrix, axis = 1)
        new_cols = [[0 for _ in range(SIZE_OF_COLS_AND_ROWS)] for _ in range(N_VOXEL)]
        self.matrix = np.append(self.matrix, new_cols, axis = 1)
        
        # vertical direction
        new_rows = [[0 for _ in range(int(N_VOXEL + 2 * SIZE_OF_COLS_AND_ROWS))] for _ in range(SIZE_OF_COLS_AND_ROWS)]
        self.matrix = np.vstack([self.matrix, new_rows])
        new_rows = [[0 for _ in range(int(N_VOXEL + 2 * SIZE_OF_COLS_AND_ROWS))] for _ in range(SIZE_OF_COLS_AND_ROWS)]
        self.matrix = np.vstack([new_rows, self.matrix])
        
    def calculate_matrix(self) -> list[list[float]]:
        # self.extend_matrix_size()
        if (len(self.matrix) != N_VOXEL + 2 * SIZE_OF_COLS_AND_ROWS): self.reset_local_matrix()
        # assert(len(self.matrix) != N_VOXEL + 2 * SIZE_OF_COLS_AND_ROWS)
        # assert(len(self.matrix[0]) != N_VOXEL + 2 * SIZE_OF_COLS_AND_ROWS)
        
        for i in range(self.sim.n_particles):
            x, y = self.sim.get_last_particle_coordinate(i)
            if (self.is_out_of_extended_bounds((x, y))): continue
            
            x += RADIUS
            y += RADIUS
            PIXEL_X = int(int(x // VOXEL_SIZE) + SIZE_OF_COLS_AND_ROWS)
            PIXEL_Y = int(int(y // VOXEL_SIZE) + SIZE_OF_COLS_AND_ROWS)
            self.matrix[PIXEL_Y][PIXEL_X] += 2000.
            
        self.apply_convolution_filter()
        self.apply_gaussian_noise()
        self.matrix = self.matrix[SIZE_OF_COLS_AND_ROWS:-SIZE_OF_COLS_AND_ROWS, SIZE_OF_COLS_AND_ROWS:-SIZE_OF_COLS_AND_ROWS]
        
        # if dirs[0] < 0: self.matrix = self.matrix[:, np.abs(dirs[0]):]
        # elif dirs[0] > 0: self.matrix = self.matrix[:, :(-1 * dirs[0])]
        # if dirs[1] < 0: self.matrix = self.matrix[np.abs(dirs[0]):, :]
        # elif dirs[1] > 0: self.matrix = self.matrix[:(-1 * dirs[0]), :]
        
                
            # if dir[0] < 0 then a[x:y, abs(dir[0]):]
            # if dir[0] > 0 then a[x:y, :(-1 * dir[0])]
        # note that a larger sigma induces a larger kernel such that
        # the pixel gets blurred over a wider distance
        return self.matrix
    
    def reset_local_matrix(self) -> None:
        self.matrix: np_t.NDArray[np.float32] = np.array([
            [0. for _ in range(N_VOXEL + 2 * SIZE_OF_COLS_AND_ROWS)] 
            for _ in range(N_VOXEL + 2 * SIZE_OF_COLS_AND_ROWS)], 
            dtype = np.float32
        )