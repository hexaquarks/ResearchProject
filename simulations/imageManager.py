import numpy as np
import numpy.typing as np_t

from simulations.simulation import *
from scipy.ndimage import gaussian_filter

__all__ = (
    "N_PIXEL",
    "ImageManager",
    "FloatMatrix"
)

N_PIXEL = 32
WIDTH = 2 * RADIUS
VOXEL_SIZE = WIDTH / N_PIXEL # N_PIXEL x N_PIXEL voxels
NUMBER_OF_COLS_OR_ROWS_TO_EXTEND = 14 # arbitrary
CONVOLUTION_SIGMA = 2 # note that a larger value yields a wider spread of the intensity
PARTICLE_INTENSITY = 500 # adjusted aesthetically
N_PIXEL_EXTENDED = N_PIXEL + 2 * NUMBER_OF_COLS_OR_ROWS_TO_EXTEND # to fix convolutions at edge of ROI

FloatMatrix = np_t.NDArray[np.float32]

class ImageManager(Simulation):
    def __init__(self, sim: Simulation) -> None:
        self.sim = sim
        self.intensity_matrices: list[FloatMatrix] = []
        self.intensity_matrices_without_background: list[FloatMatrix] = []
        
        self.intensity_matrix: FloatMatrix = self.reset_local_matrix(False)
        self.intensity_matrix_background: FloatMatrix = self.generate_random_background()
        self.intensity_matrix_without_background: FloatMatrix = self.reset_local_matrix(False)
        
        self.pixel_fluctuation_matrices: list[FloatMatrix] = []
        self.pixel_fluctuation_matrix: FloatMatrix = self.reset_local_matrix(False)
        
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
        
    def calculate_pixel_fluctuation_matrix(self) -> FloatMatrix: 
        avg = self.intensity_matrix.sum() / len(self.intensity_matrix) ** 2
        return [
            [0 if self.intensity_matrix[i][j] - avg < 0 
               else self.intensity_matrix[i][j] - avg for j in range(N_PIXEL)] 
            for i in range(N_PIXEL)
        ]

    def add_pixel_fluctuation_matrix_to_images(self, new_mat) -> None:
        self.pixel_fluctuation_matrices.append( [row[:] for row in new_mat] )
    
    def add_intensity_matrix_to_storage(self) -> None:
        self.intensity_matrices_without_background.append( [row[:] for row in (self.intensity_matrix_without_background)] )
        self.intensity_matrices.append( [row[:] for row in self.intensity_matrix] )
        
    def is_out_of_extended_bounds(self, pos: tuple[int, int]) -> bool:
        x, y = pos[0], pos[1]
        max_index = N_PIXEL_EXTENDED
        return x < 0 or x >= max_index or y < 0 or y >= max_index
    
    def apply_convolution_filter(self) -> None:
        self.intensity_matrix = gaussian_filter(
            self.intensity_matrix, 
            sigma = CONVOLUTION_SIGMA
        ) 
        self.intensity_matrix_without_background = gaussian_filter(
            self.intensity_matrix_without_background, 
            sigma = CONVOLUTION_SIGMA
        ) 
    
    def apply_gaussian_noise(self) -> None:
        noise_delta = np.abs(np.random.normal(0, 1, self.intensity_matrix.shape))
        self.intensity_matrix += noise_delta
    
    def generate_random_background(self) -> FloatMatrix:
        return [
            [np.random.choice(np.arange(0, 6)*125, p=[0.85, 0.03, 0.03, 0.03, 0.03, 0.03]) 
             for _ in range(N_PIXEL_EXTENDED)]
             for _ in range(N_PIXEL_EXTENDED)
        ]
        
    def apply_discrete_noise_from_custom_probability_function(self) -> None:
        for i in range(len(self.intensity_matrix)):
            for j in range(len(self.intensity_matrix)):
                noise_delta = np.random.choice(np.arange(0, 6)*5, p=[0.9, 0.02, 0.02, 0.02, 0.02, 0.02])
                self.intensity_matrix[i][j] += noise_delta
                self.intensity_matrix_without_background[i][j] += noise_delta
    
    def apply_background_matrix(self) -> None:
        for i in range(len(self.intensity_matrix)):
            for j in range(len(self.intensity_matrix)):
                self.intensity_matrix[i][j] += self.intensity_matrix_background[i][j]
                
    def trim_matrix_for_display(self) -> None: 
        val = NUMBER_OF_COLS_OR_ROWS_TO_EXTEND
        self.intensity_matrix = self.intensity_matrix[val:-val, val:-val]
        self.intensity_matrix_without_background = self.intensity_matrix_without_background[val:-val, val:-val]
        
    def get_pixel_coord(self, x: float) -> int:
        return int(x // VOXEL_SIZE) + NUMBER_OF_COLS_OR_ROWS_TO_EXTEND
    
    def calculate_matrix(self) -> FloatMatrix:
        # Protective guard
        if (len(self.intensity_matrix) != N_PIXEL_EXTENDED): 
            self.intensity_matrix = self.reset_local_matrix(True)
        if (len(self.intensity_matrix_without_background) != N_PIXEL_EXTENDED): 
            self.intensity_matrix_without_background = self.reset_local_matrix(True)
        
        for i in range(self.sim.n_particles):
            x, y = self.sim.get_last_particle_coordinate(i)
            x, y = self.get_pixel_coord(x + RADIUS), self.get_pixel_coord(y + RADIUS)
            if (self.is_out_of_extended_bounds((x, y))): continue
            self.intensity_matrix[y][x] += PARTICLE_INTENSITY
            self.intensity_matrix_without_background[y][x] += PARTICLE_INTENSITY
            
        self.apply_background_matrix()
        self.apply_convolution_filter()
        self.trim_matrix_for_display()
        self.apply_discrete_noise_from_custom_probability_function()
        
        self.add_pixel_fluctuation_matrix_to_images(self.calculate_pixel_fluctuation_matrix())
        self.add_intensity_matrix_to_storage()
        
        return self.intensity_matrix
    
    def reset_local_matrix(self, is_extended: bool) -> FloatMatrix:
        range_ = N_PIXEL + (2 * NUMBER_OF_COLS_OR_ROWS_TO_EXTEND if is_extended else 0)
        matrix: FloatMatrix = np.array([
                [0. for _ in range(range_)] 
                for _ in range(range_)
            ], 
            dtype = np.float32
        )   
        return matrix