from typing import List, Tuple
from xmlrpc.client import Boolean, boolean
import numpy as np
import matplotlib.pyplot as plt

from simulations.simulation import *
from util import *
        
class HopDiffusion(Simulation):
    global BOUNDARY_THICKNESS
    global NUMBER_OF_BOUNDARIES_PER_DIRECTION
    global BOUNDARY_JUMP
    BOUNDARY_THICKNESS = 6
    NUMBER_OF_BOUNDARIES_PER_DIRECTION = 3
    BOUNDARY_JUMP = BOUNDARY_THICKNESS
    
    def __init__(self, n: int = 5):
        self.boundary_coordinates_for_plot: List[List] = []
        self.boundary_coordinates: List[Tuple[Tuple]] = []
        self.generate_boundaries()
        super().__init__(n)
        
    def generate_boundaries(self):
        step: int = int((RADIUS << 1) / NUMBER_OF_BOUNDARIES_PER_DIRECTION)
        
        for i in range(6):
            horizontal: bool = i < NUMBER_OF_BOUNDARIES_PER_DIRECTION
            curr = i * step if horizontal else (i - NUMBER_OF_BOUNDARIES_PER_DIRECTION) * step
            
            width = BOUNDARY_THICKNESS if horizontal else RADIUS << 1
            height = BOUNDARY_THICKNESS if not horizontal else RADIUS << 1
            x = curr - RADIUS - (BOUNDARY_THICKNESS >> 1) if horizontal else -RADIUS
            y = curr - RADIUS - (BOUNDARY_THICKNESS >> 1) if not horizontal else -RADIUS
            
            self.boundary_coordinates_for_plot.append(list([x, y, width, height]))
            self.boundary_coordinates.append(list([tuple((x, x + width)), tuple((y, y + height))]))
    
    @property
    def get_boundary_coordinates(self):
        return self.boundary_coordinates_for_plot
    
    def is_particle_on_specific_boudnary(self, pos: Tuple, idx: int):
        return Util.is_point_within_bounds(pos, self.boundary_coordinates[idx])
    
    def is_particle_on_boundary(self, pos: Tuple):   
        return any(
            Util.is_point_within_bounds(pos, bounds_of_boundary)
            for bounds_of_boundary in self.boundary_coordinates
        ) 
    
    def is_particle_in_compartment(self, particle) -> bool:
        return not self.is_particle_on_boundary(particle)
    
    def get_surrounding_boundary_of_particle(self, pos: Tuple) -> int:
        for idx, bounds_of_boundary in enumerate(self.boundary_coordinates):
            if Util.is_point_within_bounds(pos, bounds_of_boundary):
                return idx
        return -1
    
    def make_particle_jump(self, newPos: Tuple, x_dir: int, y_dir: int):
        surrounding_boundary_idx = self.get_surrounding_boundary_of_particle(newPos)
        
        while (self.is_particle_on_specific_boudnary(newPos, surrounding_boundary_idx)):
            newPos = Util.increment_tuple_by_val(
                newPos, tuple((Util.sign(x_dir), Util.sign(y_dir)))
            )
            
        newPos = Util.increment_tuple_by_val(
            newPos, tuple(
                (Util.sign(x_dir) * BOUNDARY_JUMP, 
                 Util.sign(y_dir) * BOUNDARY_JUMP)
            )
        )
        
        return newPos
    
    def update_path(self, idx):
        x, y = self.paths[idx][-1]
        assert(not self.is_particle_on_boundary(tuple((x ,y))))
        
        diffusion_factor = MEMBRANE_DIFFUSION_FATOR_CORRECTED
        x_dir, y_dir = [Util.get_random_normal_direction() * diffusion_factor for _ in range(2)]
        newPos = tuple((x + x_dir, y + y_dir))
        
        if self.is_particle_on_boundary(newPos):
            newPos = self.make_particle_jump(newPos, x_dir, y_dir)
            
        self.paths[idx].append(newPos)
        
    def update(self):
        [self.update_path(i) for i in range(self.numberOfParticles)]
        
    def initializeParticles(self) -> None:
        mem: List[Tuple] = []
        
        def getRandomCanvasValue(self) -> int:
            return int(random.randint(-(CORRECTED_CANVAS_RADIUS), CORRECTED_CANVAS_RADIUS))
        
        def rec(self, x: int = 0, y: int = 0) -> Tuple[int]:
            x, y = [getRandomCanvasValue(self) for _ in range(2)]
            while (x, y) in mem or self.is_particle_on_boundary(tuple((x, y))):
                return rec(self, x, y)
            mem.append((x, y))
            return x,y
        
        self.particlesLocation.extend([rec(self) for _ in range(5)])
    
