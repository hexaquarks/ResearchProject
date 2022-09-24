from typing import List, Tuple
from matplotlib.pyplot import figure
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes,mark_inset
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
import random

import plotGenerator

PATH = Tuple[List[float]]
DPI = 100
RADIUS_PADDING = 10
RADIUS = 250
CORRECTED_RADIUS = RADIUS - RADIUS_PADDING

class Util:
    @staticmethod
    def get_bounds(lists: List[PATH]) -> Tuple[int]:
        X_MAX = max([max(elem[0]) for elem in lists])
        Y_MAX = max([max(elem[1]) for elem in lists])
        X_MIN = min([min(elem[0]) for elem in lists])
        Y_MIN = min([min(elem[1]) for elem in lists])
        return X_MIN, X_MAX, Y_MIN, Y_MAX
    
    @staticmethod
    def get_last_point(path: PATH) -> Tuple[int]:
        return path[0][-1], path[1][-1]
    
    @staticmethod
    def get_x_coordinates(path) -> List:
        return list(list(zip(*path))[0])
    
    @staticmethod
    def get_y_coordinates(path) -> List:
        return list(list(zip(*path))[1])

class Brownian:
    global STEPS
    STEPS = 500
    
    def __init__(self, n = 5):
        self.numberOfParticles = n
        self.particlesLocation: List[Tuple[int]] = [] 
        self.paths: List[List[Tuple[int]]] = []
        
        self.initializeParticles()
        self.initializePaths()
    
    def initializePaths(self):
        for coordinate in self.particlesLocation:
            self.paths.append([coordinate])
            
    def initializeParticles(self):
        mem: List[Tuple] = []
        
        def recc(self, x = 0, y = 0):
            x, y = [int(random.randint(-(CORRECTED_RADIUS), CORRECTED_RADIUS)) for _ in range(2)]
            while (x, y) in mem:
                return recc(self, x, y)
            mem.append((x, y))
            return x,y
        
        self.particlesLocation.extend([recc(self) for _ in range(5)])
        
    def updatePath(self, idx):
        x_dir, y_dir = [np.random.normal() * np.random.choice([1, -1]) * 3 for _ in range(2)]
        x, y = self.paths[idx][-1]
        
        self.paths[idx].append((x + x_dir, y + y_dir))
        
    def update(self):
        [self.updatePath(i) for i in range(self.numberOfParticles)]
    
        
    def generateRandomWalks(self, steps:int = 100) -> PATH:
        w = np.ones(steps) * self.x0 # array of size steps of els 0.
        
        for i in range(1, steps):
            # Sampling from the Normal distribution with probability 1/2
            yi = np.random.choice([1, -1])
            # Weiner process
            w[i] = w[i - 1] + (yi / np.sqrt(steps))
        return w
    
    def generateNormalPath(self) -> List[float]:
        w = np.ones(STEPS) * self.x0
        for i in range(1, STEPS):
            p = np.random.normal()
            w[i] = w[i - 1] + (p / np.sqrt(STEPS))
        return w
    
    def generateNormal(self, steps = 100) -> PATH:
        wx = self.generateNormalPath()
        wy = self.generateNormalPath()
        return wx, wy 

b = Brownian()
plotGenerator.plot(b)

class AbstractClass:
    def __init__(self):
        raise NotImplementedError
