from simulations.simulation import *
from util import *
        
NANODOMAIN_DIFFUSION_FACTOR_CORRECTED: float = MEMBRANE_DIFFUSION_FACTOR_CORRECTED * 0.4 # type : ignore

class Nanodomain(Simulation):
    
    def __init__(self, n: int = 5):
        super().__init__(n)
        self.nanodomain_coordinates: list[tuple[int, int]] = [ 
            (-100, 100), (0, 0), (150, -60), (-130, -160)
        ]
        self.nanodomain_radii: list[int] = [80, 20, 50, 140]
        
    @property
    def get_nanodomain_coordinates(self) -> list[tuple[int, int]]:
        return self.nanodomain_coordinates
    
    @property
    def get_nanodomain_radii(self) -> list[int]:
        return self.nanodomain_radii
    
    def get_nanodomain_attributes(self) -> list[tuple]:
        return list(map(
            lambda coord, radius: (coord, radius), 
            self.get_nanodomain_coordinates, 
            self.get_nanodomain_radii
        ))
    
    def is_particle_in_nanodomain(self, particle: tuple) -> bool:
        return any(
            Util.compute_distance(particle, circle_center) <= radius 
            for circle_center, radius in 
            zip(self.get_nanodomain_coordinates, self.get_nanodomain_radii)
        )
    
    def update_path(self, idx):
        x, y = self.paths[idx][-1]
        diffusion_factor = NANODOMAIN_DIFFUSION_FACTOR_CORRECTED if (self.is_particle_in_nanodomain((x, y))) else MEMBRANE_DIFFUSION_FACTOR_CORRECTED
        x_dir, y_dir = [Util.get_random_normal_direction() * diffusion_factor for _ in range(2)]
        self.paths[idx].append((x + x_dir, y + y_dir))
        
    def update(self):
        [self.update_path(i) for i in range(self.number_of_particles)]
    
