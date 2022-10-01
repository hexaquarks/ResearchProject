from typing import List, Tuple
import random
import numpy as np
from enum import Enum

class SimulationType(Enum):
     BROWNIAN = 1
     NANODOMAIN = 2
     HOPDIFFUSION = 3     

PATH = Tuple[List[float]]
DPI = 100
RADIUS_PADDING = 10
RADIUS = 250
CORRECTED_CANVAS_RADIUS = RADIUS - RADIUS_PADDING

MEMBRANE_DIFFUSION_COEFFICIENT: float
MEMBRANE_DIFFUSION_FATOR_CORRECTED: float
MEMBRANE_DIFFUSION_FACTOR: float
TIME_PER_FRAME: float
DIFFUSION_SPEED_CORRECTION: int

class Simulation:
    
    def __init__(self, n: int = 5):
        self.numberOfParticles: int = n
        self.particlesLocation: List[Tuple[int, int]] = [] 
        self.paths: List[List[Tuple[int, int]]] = []
        
        self.initGlobalVariables()
        self.initializeParticles()
        self.initializePaths()
        
    def initGlobalVariables(self):
        TIME_PER_FRAME = 0.02 # 20 ms
        DIFFUSION_SPEED_CORRECTION = 35 # arbitrary
        MEMBRANE_DIFFUSION_COEFFICIENT = 0.1 # micrometer^2 / s
        MEMBRANE_DIFFUSION_FACTOR = 2 * np.sqrt(MEMBRANE_DIFFUSION_COEFFICIENT * TIME_PER_FRAME)
        MEMBRANE_DIFFUSION_FATOR_CORRECTED = MEMBRANE_DIFFUSION_FACTOR * DIFFUSION_SPEED_CORRECTION
        
    def initializePaths(self):
        self.paths.extend([[coordinate] for coordinate in self.particlesLocation])
            
    def initializeParticles(self) -> None:
        mem: List[Tuple] = []
        
        def getRandomCanvasValue(self) -> int:
            return int(random.randint(-(CORRECTED_CANVAS_RADIUS), CORRECTED_CANVAS_RADIUS))
        
        def rec(self, x: int = 0, y: int = 0) -> Tuple[int, int]:
            x, y = [getRandomCanvasValue(self) for _ in range(2)]
            while (x, y) in mem:
                return rec(self, x, y)
            mem.append((x, y))
            return x,y
        
        self.particlesLocation.extend([rec(self) for _ in range(5)])
        