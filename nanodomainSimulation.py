from typing import List, Tuple
from xmlrpc.client import Boolean, boolean
import numpy as np
import matplotlib.pyplot as plt

from simulation import *
from util import *
        
class Nanodomain(Simulation):
    global NANODOMAIN_DIFFUSION_FATOR_CORRECTED
    NANODOMAIN_DIFFUSION_FATOR_CORRECTED = MEMBRANE_DIFFUSION_FATOR_CORRECTED * 0.4
    
    def __init__(self, n: int = 5):
        super().__init__(n)
        self.nanodomain_coordinates: List[Tuple[int]] = [ 
            (-100, 100), (0, 0), (150, -60), (-130, -160)
        ]
        self.nanodomain_radii: List[int] = [80, 20, 50, 140]
        
    @property
    def get_nanodomain_coordinates(self) -> List[Tuple[int]]:
        return self.nanodomain_coordinates
    
    @property
    def get_nanodomain_radii(self) -> List[int]:
        return self.nanodomain_radii
    
    def is_particle_in_nanodomain(self, particle) -> bool:
        return any(
            Util.compute_distance(particle, circle_center) <= radius 
            for circle_center, radius in 
            zip(self.get_nanodomain_coordinates, self.get_nanodomain_radii)
        )
    
    def update_path(self, idx):
        x, y = self.paths[idx][-1]
        diffusion_factor = NANODOMAIN_DIFFUSION_FATOR_CORRECTED if (self.is_particle_in_nanodomain((x, y))) else MEMBRANE_DIFFUSION_FATOR_CORRECTED
        x_dir, y_dir = [Util.get_random_normal_direction() * diffusion_factor for _ in range(2)]
        self.paths[idx].append((x + x_dir, y + y_dir))
        
    def update(self):
        [self.update_path(i) for i in range(self.numberOfParticles)]
    
