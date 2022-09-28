from typing import List, Tuple
from xmlrpc.client import Boolean, boolean
import numpy as np
import matplotlib.pyplot as plt

from simulation import *
from util import *
        
class HopDiffusion(Simulation):
    
    def __init__(self, n: int = 5):
        super().__init__(n)
        
    def is_particle_in_compartment(self, particle) -> bool:
        pass
    
    def update_path(self, idx):
        x, y = self.paths[idx][-1]
        diffusion_factor = MEMBRANE_DIFFUSION_FATOR_CORRECTED
        x_dir, y_dir = [Util.get_random_normal_direction() * diffusion_factor for _ in range(2)]
        self.paths[idx].append((x + x_dir, y + y_dir))
        
    def update(self):
        [self.update_path(i) for i in range(self.numberOfParticles)]
    
