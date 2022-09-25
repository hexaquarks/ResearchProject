from typing import List, Tuple
import random
from enum import Enum

class SimulationType(Enum):
     BROWNIAN = 1
     NANODOMAIN = 2
     HOPDIFFUSION = 3     

PATH = Tuple[List[float]]
DPI = 100
RADIUS_PADDING = 10
RADIUS = 250
CORRECTED_RADIUS = RADIUS - RADIUS_PADDING

class Simulation:
    def __init__(self, n = 5):
        self.numberOfParticles = n
        self.particlesLocation: List[Tuple[int]] = [] 
        self.paths: List[List[Tuple[int]]] = []
        
        self.initializeParticles()
        self.initializePaths()
        
    def initializePaths(self):
        self.paths.extend([[coordinate] for coordinate in self.particlesLocation])
            
    def initializeParticles(self):
        mem: List[Tuple] = []
        
        def getRandomCanvasValue(self):
            return int(random.randint(-(CORRECTED_RADIUS), CORRECTED_RADIUS))
        
        def rec(self, x = 0, y = 0):
            x, y = [getRandomCanvasValue(self) for _ in range(2)]
            while (x, y) in mem:
                return rec(self, x, y)
            mem.append((x, y))
            return x,y
        
        self.particlesLocation.extend([rec(self) for _ in range(5)])
        print(self.particlesLocation)
        