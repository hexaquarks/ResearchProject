from simulations.simulation import *
import util


class Brownian(Simulation):
    def __init__(self, n: int = 5):
        super().__init__(n)

    def update_path(self, idx: int):
        x_dir, y_dir = [
            util.get_random_normal_direction() * MEMBRANE_DIFFUSION_FACTOR_CORRECTED
            for _ in range(2)
        ]
        x, y = self.paths[idx][-1]
        self.paths[idx].append((x + x_dir, y + y_dir))

    def update(self) -> None:
        for i in range(self.n_particles):
            self.update_path(i)
