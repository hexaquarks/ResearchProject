import random
import numpy as np

DPI = 100
RADIUS_PADDING = 20
RADIUS = 750
CORRECTED_CANVAS_RADIUS = RADIUS - RADIUS_PADDING

TIME_PER_FRAME: float = 0.02 # 20 ms
DIFFUSION_SPEED_CORRECTION: int = 1000 # arbitrary
MEMBRANE_DIFFUSION_COEFFICIENT: float = 0.1 # micrometer^2 / s
MEMBRANE_DIFFUSION_FACTOR: float = 2 * np.sqrt(MEMBRANE_DIFFUSION_COEFFICIENT * TIME_PER_FRAME)
MEMBRANE_DIFFUSION_FACTOR_CORRECTED: float = MEMBRANE_DIFFUSION_FACTOR * DIFFUSION_SPEED_CORRECTION


class Simulation:
    def __init__(self, n: int = 5) -> None:
        self.n_particles: int = n
        self.particle_locations: list[tuple[float, float]] = list(self.init_particles())
        self.paths: list[list[tuple[float, float]]] = [
            [coordinate] for coordinate in self.particle_locations
        ]

    def get_last_particle_coordinate(self, idx: int) -> tuple[float, float]:
        return self.paths[idx][-1]
    
    @staticmethod
    def get_random_canvas_value() -> int:
        return int(random.randint(-CORRECTED_CANVAS_RADIUS, CORRECTED_CANVAS_RADIUS))

    def init_particles(self) -> set[tuple[float, float]]:
        mem: set[tuple[float, float]] = set()

        for _ in range(self.n_particles):
            while True:
                pair = (self.get_random_canvas_value(), self.get_random_canvas_value())
                if pair not in mem: break
            mem.add(pair)

        return mem

    def update(self) -> None:
        """Abstract class"""
        raise NotImplementedError()
