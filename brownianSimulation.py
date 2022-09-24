from typing import List, Tuple
from matplotlib.pyplot import figure
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes,mark_inset
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
import random

from simulation import *
import plotGenerator

PATH = Tuple[List[float]]
DPI = 100
RADIUS_PADDING = 10
RADIUS = 250
CORRECTED_RADIUS = RADIUS - RADIUS_PADDING

class Brownian(Simulation):
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
    
    # def generateRandomWalks(self, steps:int = 100) -> PATH:
    #     w = np.ones(steps) * self.x0 # array of size steps of els 0.
        
    #     for i in range(1, steps):
    #         # Sampling from the Normal distribution with probability 1/2
    #         yi = np.random.choice([1, -1])
    #         # Weiner process
    #         w[i] = w[i - 1] + (yi / np.sqrt(steps))
    #     return w
    
    # def generateNormalPath(self) -> List[float]:
    #     w = np.ones(STEPS) * self.x0
    #     for i in range(1, STEPS):
    #         p = np.random.normal()
    #         w[i] = w[i - 1] + (p / np.sqrt(STEPS))
    #     return w
    
    # def generateNormal(self, steps = 100) -> PATH:
    #     wx = self.generateNormalPath()
    #     wy = self.generateNormalPath()
    #     return wx, wy 

b = Brownian()
plotGenerator.plot(b)
