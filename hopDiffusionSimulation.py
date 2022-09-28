from typing import List, Tuple
from xmlrpc.client import Boolean, boolean
import numpy as np
import matplotlib.pyplot as plt

from simulation import *
from util import *
        
class HopDiffusion(Simulation):
    global THICKNESS
    THICKNESS = 6
    
    def __init__(self, n: int = 5):
        super().__init__(n)
        self.rectangle_coordinates: List[List] = []
        self.generate_rectangle_attributes()
        
    def generate_rectangle_attributes(self):
        step: float = float(2 * RADIUS / 3)
        for i in range(3):
            width, height = THICKNESS, 2 * RADIUS
            x = i * step - RADIUS - (THICKNESS >> 1)
            y = -RADIUS 
            self.rectangle_coordinates.append(list([x, y, width, height]))
        for i in range(3):
            width, height = 2 * RADIUS, THICKNESS
            x = -RADIUS 
            y = i * step - RADIUS - (THICKNESS >> 1)
            self.rectangle_coordinates.append(list([x, y, width, height]))
    
    @property
    def get_rectangle_coordinates(self):
        return self.rectangle_coordinates
        
    def is_particle_in_compartment(self, particle) -> bool:
        pass
    
    def update_path(self, idx):
        x, y = self.paths[idx][-1]
        diffusion_factor = MEMBRANE_DIFFUSION_FATOR_CORRECTED
        x_dir, y_dir = [Util.get_random_normal_direction() * diffusion_factor for _ in range(2)]
        self.paths[idx].append((x + x_dir, y + y_dir))
        
    def update(self):
        [self.update_path(i) for i in range(self.numberOfParticles)]
    
