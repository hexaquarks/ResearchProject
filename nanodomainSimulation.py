from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

from simulation import *

class Nanodomain(Simulation):
    global STEPS
    STEPS = 500
    
    def __init__(self, n = 5):
        super().__init__(n)
        
    def updatePath(self, idx):
        x_dir, y_dir = [np.random.normal() * np.random.choice([1, -1]) * 3 for _ in range(2)]
        x, y = self.paths[idx][-1]
        
        self.paths[idx].append((x + x_dir, y + y_dir))
        
    def update(self):
        [self.updatePath(i) for i in range(self.numberOfParticles)]
    
