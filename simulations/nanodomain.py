from simulations.simulation import *
import util

NANODOMAIN_DIFFUSION_FACTOR_CORRECTED: float = MEMBRANE_DIFFUSION_FACTOR_CORRECTED * 0.4 # type : ignore


class Nanodomain(Simulation):
    def __init__(self, n: int = 5, spawn_in_center: bool = False):
        super().__init__(n, spawn_in_center)
        self.nanodomain_coordinates: list[tuple[float, float]] = [
            (-600, 600), (0, 0), (800, -360), (-780, -960)
        ]
        self.nanodomain_radii: list[int] = [480, 120, 300, 940]

    def get_nanodomain_attributes(self) -> list[tuple]:
        return list(map(
            lambda coord, radius: (coord, radius),
            self.nanodomain_coordinates,
            self.nanodomain_radii
        ))

    def is_particle_in_nanodomain(self, particle: tuple) -> bool:
        return any(
            util.compute_distance(particle, circle_center) <= radius
            for circle_center, radius in zip(self.nanodomain_coordinates, self.nanodomain_radii)
        )

    def update_path(self, idx: int) -> None:
        x, y = self.paths[idx][-1]
        diffusion_factor = (
            NANODOMAIN_DIFFUSION_FACTOR_CORRECTED
            if self.is_particle_in_nanodomain((x, y))
            else MEMBRANE_DIFFUSION_FACTOR_CORRECTED
        )
        x_dir, y_dir = [util.get_random_normal_direction() * diffusion_factor for _ in range(2)]
        self.paths[idx].append((x + x_dir, y + y_dir))

    def update(self) -> None:
        for i in range(self.n_particles):
            self.update_path(i)
