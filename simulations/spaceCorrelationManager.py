import numpy as np

from simulations.imageManager import *
from scipy import signal
from numpy import fft as fft

class SpaceCorrelationManager(ImageManager):
    def __init__(self, image_manager: ImageManager) -> None:
        self.images: list[FloatMatrix] = image_manager.intensity_matrices_without_background
        self.corr_function_frames: list[FloatMatrix] = self.get_frames()
        self.peak_decay_list: list[float] = []
        self.peak_decay_list_error: list[float] = []
        self.imsd_list: list[float] = []
        
    @property
    def get_corr_function_frames(self):
        return self.corr_function_frames
    
    @property
    def get_peak_decay_list(self):
        return self.peak_decay_list
    
    def correlate(
        self,
        im1: FloatMatrix,
        im2: FloatMatrix
    ) -> FloatMatrix: return signal.correlate2d(im1, im2)

    def compute_average(self, matrices):
        return [
            [sum(list(matrices[k][i][j] for k in range(len(matrices)))) / len(matrices)
             for j in range(len(matrices[0]))]
            for i in range(len(matrices[0]))
        ]

    def get_frame_bruteforce(self) -> list[FloatMatrix]:
        frame = []
        shift: int = 0
        to_iterate = len(self.images)
        while to_iterate != 0:
            curr_matrices = []
            curr_idx = 0
            for _ in range(to_iterate):
                curr_matrices.append(
                    self.correlate(
                        self.images[curr_idx], self.images[curr_idx + shift])
                )
                curr_idx += 1
            to_iterate -= 1
            shift += 1
            frame.append(self.compute_average(curr_matrices))

        return frame

    def get_frames(self) -> list[FloatMatrix]:
        fft_images = []

        for image in self.images:
            normalization_factor = ((np.mean(image) * len(image)) ** 2)
            image = fft.irfft2(
                        np.real(
                            np.matmul(
                                fft.fft2(image),
                                np.matrix.conjugate(
                                    fft.fft2(image)
                                )
                            )
                        ), 
                        s = (N_PIXEL, N_PIXEL)
            ) / normalization_factor - 1
            fft_images.append(image)
            
        return fft_images
    
    def update_peak_decay_list(self, amplitude: float, amplitude_error: float):
        self.peak_decay_list.append(amplitude)
        self.peak_decay_list_error.append(amplitude_error)
        
    def update_imsd_list(self, sigma: float):
        self.imsd_list.append(sigma ** 2)
        
    def get_peak_decay_list(self) -> list[float]:
        return self.peak_decay_list
    
    def get_peak_decay_list_error(self) -> list[float]:
        return self.peak_decay_list_error
    
    def get_imsd_list(self) -> list[float]:
        return self.imsd_list
