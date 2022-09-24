from typing import List, Tuple
import random

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
        print(self.particlesLocation)
        
    