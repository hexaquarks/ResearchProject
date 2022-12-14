import random
import numpy as np

__all__ = (
    "DPI",
    "RADIUS_PADDING",
    "RADIUS",
    "TIME_PER_FRAME",
    "DIFFUSION_SPEED_CORRECTION",
    "MEMBRANE_DIFFUSION_COEFFICIENT",
    "MEMBRANE_DIFFUSION_FACTOR",
    "MEMBRANE_DIFFUSION_FACTOR_CORRECTED",
    "Simulation"
)

DPI = 100
RADIUS_PADDING = 20
RADIUS = 1600
CORRECTED_CANVAS_RADIUS = RADIUS - RADIUS_PADDING
PATH_LIMIT = 20

TIME_PER_FRAME: float = 0.05 # 50 ms
DIFFUSION_SPEED_CORRECTION: int = 1 # arbitrary
MEMBRANE_DIFFUSION_COEFFICIENT: float = 100000 # nanometer^2 / s <=> 0.1 micrometer^2 / s
MEMBRANE_DIFFUSION_FACTOR: float = 2 * np.sqrt(MEMBRANE_DIFFUSION_COEFFICIENT * TIME_PER_FRAME)
MEMBRANE_DIFFUSION_FACTOR_CORRECTED: float = MEMBRANE_DIFFUSION_FACTOR * DIFFUSION_SPEED_CORRECTION


class Simulation:
    def __init__(self, n: int = 5, spawn_in_center: bool = False) -> None:
        self.n_particles: int = n
        self.particle_locations: list[tuple[float, float]] = list(self.init_particles(spawn_in_center))
        self.paths: list[list[tuple[float, float]]] = [
            [coordinate] for coordinate in self.particle_locations
        ]

    def get_last_particle_coordinate(self, idx: int) -> tuple[float, float]:
        return self.paths[idx][-1]
    
    @staticmethod
    def get_random_canvas_value() -> int:
        return int(random.randint(-CORRECTED_CANVAS_RADIUS, CORRECTED_CANVAS_RADIUS))
    
    @staticmethod
    def get_random_center_canvas_value() -> int:
        return int(random.randint(-300, 300))  
    
    def init_particles(self, spawn_in_center: bool = False) -> set[tuple[float, float]]:
        mem: set[tuple[float, float]] = set()
            
        for _ in range(self.n_particles):
            while True:
                if spawn_in_center:
                    pair = (self.get_random_center_canvas_value(), self.get_random_center_canvas_value())
                else:
                    pair = (self.get_random_canvas_value(), self.get_random_canvas_value())
                if pair not in mem: break
            mem.add(pair)

        return mem
    
    def trim_paths(self, idx: int):
        if (len(self.paths[idx]) >= PATH_LIMIT): self.paths[idx] = self.paths[idx][-PATH_LIMIT:]

    def update(self) -> None:
        """Abstract class"""
        raise NotImplementedError()
