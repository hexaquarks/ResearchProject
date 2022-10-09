from simulations.simulation import *
import util

BOUNDARY_THICKNESS: int = 45
NUMBER_OF_COMPARTMENTS_PER_DIRECTION: int = 3
BOUNDARY_JUMP: int = BOUNDARY_THICKNESS
BOUNDARY_OVERFLOW: int = 60
HOP_PROBABILITY_PERCENTAGE: float = 0.15


class HopDiffusion(Simulation):
    def __init__(self, n: int = 5) -> None:
        self.boundary_coordinates_for_plot: list[tuple[int, int, int, int]] = []
        self.boundary_coordinates: list[tuple[tuple[float, float], tuple[float, float]]] = []
        self.generate_boundaries()
        super().__init__(n)

    def generate_boundaries(self) -> None:
        step = (RADIUS * 2) // NUMBER_OF_COMPARTMENTS_PER_DIRECTION
        
        for i in range(6):
            dynamic_step = step
            if i % 3 == 0: continue
            if i == 2 or i == 5: dynamic_step = step * 1.25
            
            horizontal = i < NUMBER_OF_COMPARTMENTS_PER_DIRECTION
            curr = i * dynamic_step if horizontal else (i - NUMBER_OF_COMPARTMENTS_PER_DIRECTION) * dynamic_step

            width = BOUNDARY_THICKNESS if horizontal else (RADIUS * 2) + (BOUNDARY_OVERFLOW * 2)
            height = BOUNDARY_THICKNESS if not horizontal else (RADIUS * 2) + (BOUNDARY_OVERFLOW * 2)
            x = curr - RADIUS - (BOUNDARY_THICKNESS // 2) if horizontal else -RADIUS - BOUNDARY_OVERFLOW
            y = curr - RADIUS - (BOUNDARY_THICKNESS // 2) if not horizontal else -RADIUS - BOUNDARY_OVERFLOW

            self.boundary_coordinates_for_plot.append((x, y, width, height))
            self.boundary_coordinates.append(((x, x + width), (y, y + height)))

    @staticmethod
    def can_particle_hop_boundary_probability() -> bool:
        return random.random() < HOP_PROBABILITY_PERCENTAGE

    def is_particle_on_specific_boundary(self, pos: tuple[float, float], idx: int) -> bool:
        return util.is_point_within_bounds(pos, self.boundary_coordinates[idx])

    def is_particle_on_boundary(self, pos: tuple[float, float]) -> bool:
        return any(
            util.is_point_within_bounds(pos, bounds_of_boundary)
            for bounds_of_boundary in self.boundary_coordinates
        )

    def is_particle_in_compartment(self, particle: tuple[float, float]) -> bool:
        return not self.is_particle_on_boundary(particle)

    def get_surrounding_boundary_of_particle(self, pos: tuple[float, float]) -> int:
        for idx, bounds_of_boundary in enumerate(self.boundary_coordinates):
            if util.is_point_within_bounds(pos, bounds_of_boundary):
                return idx
        return -1

    def make_particle_jump(
        self,
        new_pos: tuple[float, float],
        x_dir: float, y_dir: float,
    ) -> tuple[float, float]:
        surrounding_boundary_idx = self.get_surrounding_boundary_of_particle(new_pos)

        while self.is_particle_on_specific_boundary(new_pos, surrounding_boundary_idx):
            new_pos = util.increment_tuple_by_val(
                new_pos, (np.sign(x_dir), np.sign(y_dir))
            )

        new_pos = util.increment_tuple_by_val(
            new_pos, (
                np.sign(x_dir) * BOUNDARY_JUMP,
                np.sign(y_dir) * BOUNDARY_JUMP,
            )
        )

        # Special case: In some instances the jump may land the particle
        # on a subsequent boundary so we repeat the function. We decrement
        # the particle's coordinates until it is out.
        while self.is_particle_on_boundary(new_pos):
            new_pos = util.increment_tuple_by_val(
                new_pos, (np.sign(-x_dir), np.sign(-y_dir))
            )

        return new_pos

    def update_path(self, idx: int) -> None:
        x, y = self.paths[idx][-1]
        assert not self.is_particle_on_boundary((x, y))

        diffusion_factor = MEMBRANE_DIFFUSION_FACTOR_CORRECTED
        x_dir, y_dir = [util.get_random_normal_direction() * diffusion_factor for _ in range(2)]
        new_pos = x + x_dir, y + y_dir

        if self.is_particle_on_boundary(new_pos):
            if self.can_particle_hop_boundary_probability():
                new_pos = self.make_particle_jump(new_pos, x_dir, y_dir)
            else:
                new_pos = util.change_direction((x, y), (x_dir, y_dir))

        self.paths[idx].append(new_pos)

    def update(self) -> None:
        for i in range(self.n_particles):
            self.update_path(i)

    def init_particles(self) -> set[tuple[float, float]]:
        mem: set[tuple[float, float]] = set()

        for _ in range(self.n_particles):
            while True:
                pair = (self.get_random_canvas_value(), self.get_random_canvas_value())
                if not (pair in mem or self.is_particle_on_boundary(pair)):
                    break
            mem.add(pair)

        return mem
