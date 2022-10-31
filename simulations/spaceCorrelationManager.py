import numpy as np
import numpy.typing as np_t
from simulations.imageManager import *

class SpaceCorrelationManager(ImageManager):
    def __init__(self, images: ImageManager) -> None:
        self.images: list[np_t.NDArray[np.float32]] = images
        print('in')
        
    