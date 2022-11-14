import numpy as np
import numpy.typing as np_t
from simulations.simulation import *
from scipy.ndimage import gaussian_filter

N_PIXEL = 128
WIDTH = 2 * RADIUS
VOXEL_SIZE = WIDTH / N_PIXEL # N_PIXEL x N_PIXEL voxels
NUMBER_OF_COLS_OR_ROWS_TO_EXTEND = 14 # arbitrary
CONVOLUTION_SIGMA = 10 # note that a larger value yields a wider spread of the intensity
PARTICLE_INTENSITY = 30000 # adjusted aesthetically

class ImageManager(Simulation):
    def __init__(self, sim: Simulation) -> None:
        self.sim = sim
        self.images: list[np_t.NDArray[np.float32]] = []
        self.intensity_matrices: list[np_t.NDArray[np.float32]] = []
        
        self.intensity_matrix: np_t.NDArray[np.float32] = self.reset_local_matrix(False)
        self.pixel_fluctuation_matrix: np_t.NDArray[np.float32] = self.reset_local_matrix(False)
        
        self.image_counter = 0;
         
    def increment_image_counter(self) -> None:
        self.image_counter += 1
    
    def update_pixel_fluctuation_space(self, row: int, col: int):  
        avg = self.intensity_matrix.sum() / len(self.intensity_matrix) ** 2
        new_val = self.intensity_matrix[col][row] - avg
        return 0 if new_val < 0 else new_val
    
    def update_pixel_fluctuation_time(self, row: int, col: int):
        new_intensity = ((self.pixel_fluctuation_matrix[row][col] * self.image_counter) +\
            self.intensity_matrix[row][col]) / (self.image_counter + 1)
        self.pixel_fluctuation_matrix[row][col] = new_intensity
        return new_intensity
                    
    def update_pixel_flucuation_matrix(self):
        # self.intensity_matrix size should be correctly trimmed when we get into this function
        assert(len(self.intensity_matrix) == N_PIXEL)
        return [
            [self.update_pixel_fluctuation_space(i, j) for j in range(N_PIXEL)]
            for i in range(N_PIXEL)
        ]

    def add_pixel_fluctuation_matrix_to_images(self, new_mat) -> None:
        self.images.append( [row[:] for row in new_mat] )
    
    def add_intensity_matrix_to_storage(self) -> None:
        self.intensity_matrices.append( [row[:] for row in self.intensity_matrix] )
        
    def is_out_of_extended_bounds(self, pos: tuple[int, int]) -> bool:
        x, y = pos[0], pos[1]
        max_index = N_PIXEL + 2 * NUMBER_OF_COLS_OR_ROWS_TO_EXTEND
        return x < 0 or x >= max_index or y < 0 or y >= max_index
    
    def apply_convolution_filter(self) -> None:
        self.intensity_matrix = gaussian_filter(self.intensity_matrix, sigma = CONVOLUTION_SIGMA) 
    
    def apply_gaussian_noise(self) -> None:
        noise_delta = np.random.normal(0, .1, self.intensity_matrix.shape)
        self.intensity_matrix += noise_delta
        
    def trim_matrix_for_display(self) -> None: 
        val = NUMBER_OF_COLS_OR_ROWS_TO_EXTEND
        self.intensity_matrix = self.intensity_matrix[val:-val, val:-val]
        
    def get_pixel_coord(self, x: float) -> int:
        return int(x // VOXEL_SIZE) + NUMBER_OF_COLS_OR_ROWS_TO_EXTEND
    
    def calculate_matrix(self) -> np_t.NDArray[np.float32]:
        # quick fix
        if (len(self.intensity_matrix) != N_PIXEL + 2 * NUMBER_OF_COLS_OR_ROWS_TO_EXTEND): 
            self.intensity_matrix = self.reset_local_matrix(True)
        
        for i in range(self.sim.n_particles):
            x, y = self.sim.get_last_particle_coordinate(i)
            x, y = self.get_pixel_coord(x + RADIUS), self.get_pixel_coord(y + RADIUS)
            if (self.is_out_of_extended_bounds((x, y))): continue
            self.intensity_matrix[y][x] += PARTICLE_INTENSITY
            
        self.apply_convolution_filter()
        self.apply_gaussian_noise()
        self.trim_matrix_for_display()
        
        #new_mat = self.update_pixel_flucuation_matrix()
        avg = self.intensity_matrix.sum() / len(self.intensity_matrix) ** 2
        new_mat = [
            [0 if self.intensity_matrix[i][j] - avg < 0 else self.intensity_matrix[i][j] - avg for j in range(N_PIXEL)] 
            for i in range(N_PIXEL)
        ]
        self.add_pixel_fluctuation_matrix_to_images(new_mat)
        self.add_intensity_matrix_to_storage()
        
        return self.intensity_matrix
    
    def reset_local_matrix(self, is_extended: bool) -> np_t.NDArray[np.float32]:
        _range = N_PIXEL + (2 * NUMBER_OF_COLS_OR_ROWS_TO_EXTEND if is_extended else 0)
        matrix: np_t.NDArray[np.float32] = np.array([
                [0. for _ in range(_range)] 
                for _ in range(_range)
            ], 
            dtype = np.float32
        )   
        return matrix