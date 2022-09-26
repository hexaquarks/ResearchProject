from typing import List, Tuple
import numpy as np

from simulation import *
from util import *

class Brownian(Simulation):
    
    def __init__(self, n = 5):
        super().__init__(n)
        
    def updatePath(self, idx):
        x_dir, y_dir = [
            Util.get_random_normal_direction() * MEMBRANE_DIFFUSION_FATOR_CORRECTED for _ in range(2)]
        x, y = self.paths[idx][-1]
        self.paths[idx].append((x + x_dir, y + y_dir))
        
    def update(self):
        [self.updatePath(i) for i in range(self.numberOfParticles)]

