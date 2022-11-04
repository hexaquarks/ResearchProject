import numpy as np
import numpy.typing as np_t
from simulations.imageManager import *
from scipy import signal
from numpy import fft as fft

class SpaceCorrelationManager(ImageManager):
    def __init__(self, image_manager: ImageManager) -> None:
        self.images: list[np_t.NDArray[np.float32]] = image_manager.images
        
    def correlate(
        self,
        im1: np_t.NDArray[np.float32], 
        im2: np_t.NDArray[np.float32]
    ) -> np_t.NDArray[np.float32]: return signal.correlate2d(im1, im2)  
    
    def compute_average(self, matrices):
        return [
            [sum(list(matrices[k][i][j] for k in range(len(matrices)))) / len(matrices)
            for j in range(len(matrices[0]))]
            for i in range(len(matrices[0]))
        ]
    
    def get_frame_bruteforce(self) -> list[np_t.NDArray[np.float32]]:
        frame = []
        shift: int = 0
        to_iterate = len(self.images)
        while to_iterate != 0:
            curr_matrices = []
            curr_idx = 0
            for _ in range(to_iterate):
                curr_matrices.append(
                    self.correlate(self.images[curr_idx], self.images[curr_idx + shift])
                )
                curr_idx += 1
            to_iterate -= 1
            shift += 1
            frame.append(self.compute_average(curr_matrices))
        
        return frame
    
    def get_frame(self) -> list[np_t.NDArray[np.float32]]:
        fft_images = []
        normalization_factor = (np.mean(image) * len(image) ** 2)
        
        for image in self.images:
            fft_images.append(
                fft.fftshift(
                    fft.irfft2(
                        np.matmul(
                            fft.fft2(image),
                            np.matrix.conjugate(
                                fft.fft2(image)
                            )   
                        )
                    )  
                ) / normalization_factor - 1
            )
        return fft_images