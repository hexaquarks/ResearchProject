from simulations.simulation import *
from util import *
        
BOUNDARY_THICKNESS: int = 15
NUMBER_OF_COMPARTMENTS_PER_DIRECTION: int = 3
BOUNDARY_JUMP: int  = BOUNDARY_THICKNESS
BOUNDARY_OVERFLOW: int = 20
HOP_PROBABILITY_PERCENTAGE: float = 0.15

class HopDiffusion(Simulation):
    def __init__(self, n: int = 5):
        self.boundary_coordinates_for_plot: list[list] = []
        self.boundary_coordinates: list[tuple[tuple]] = []
        self.generate_boundaries()
        
        super().__init__(n)
        
    def generate_boundaries(self):
        step: int = int((RADIUS << 1) / NUMBER_OF_COMPARTMENTS_PER_DIRECTION)
        
        for i in range(6):
            if i % 3 == 0: continue 
            horizontal: bool = i < NUMBER_OF_COMPARTMENTS_PER_DIRECTION
            curr = i * step if horizontal else (i - NUMBER_OF_COMPARTMENTS_PER_DIRECTION) * step
            
            width = BOUNDARY_THICKNESS if horizontal else (RADIUS << 1) + (BOUNDARY_OVERFLOW << 1)
            height = BOUNDARY_THICKNESS if not horizontal else (RADIUS << 1) + (BOUNDARY_OVERFLOW << 1)
            x = curr - RADIUS - (BOUNDARY_THICKNESS >> 1) if horizontal else -RADIUS - BOUNDARY_OVERFLOW
            y = curr - RADIUS - (BOUNDARY_THICKNESS >> 1) if not horizontal else -RADIUS - BOUNDARY_OVERFLOW
            
            self.boundary_coordinates_for_plot.append(list([x, y, width, height]))
            self.boundary_coordinates.append(list([tuple((x, x + width)), tuple((y, y + height))]))
    
    @property
    def get_boundary_coordinates(self):
        return self.boundary_coordinates_for_plot

    def can_particle_hop_boundary_probability(self) -> bool:
        return random.random() < HOP_PROBABILITY_PERCENTAGE

    def is_particle_on_specific_boudnary(self, pos: tuple, idx: int):
        return Util.is_point_within_bounds(pos, self.boundary_coordinates[idx])
    
    def is_particle_on_boundary(self, pos: tuple):   
        return any(
            Util.is_point_within_bounds(pos, bounds_of_boundary)
            for bounds_of_boundary in self.boundary_coordinates
        ) 
    
    def is_particle_in_compartment(self, particle) -> bool:
        return not self.is_particle_on_boundary(particle)
    
    def get_surrounding_boundary_of_particle(self, pos: tuple) -> int:
        for idx, bounds_of_boundary in enumerate(self.boundary_coordinates):
            if Util.is_point_within_bounds(pos, bounds_of_boundary):
                return idx
        return -1
    
    def make_particle_jump(self, newPos: tuple, x_dir: int, y_dir: int):
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
        # Special case: In some instances the jump may land the particle
        # on a subsequent boundary so we repeat the function. We decrement
        # the particle's coordinates until it is out.
        new_surrounding_boundary_idx = self.get_surrounding_boundary_of_particle(newPos)
        while (self.is_particle_on_boundary(newPos)):
            newPos = Util.increment_tuple_by_val(
                newPos, tuple((Util.sign(-x_dir), Util.sign(-y_dir)))
            )
        
        return newPos
    
    def update_path(self, idx: int):
        x, y = self.paths[idx][-1]
        assert(not self.is_particle_on_boundary(tuple((x, y))))
        
        diffusion_factor = MEMBRANE_DIFFUSION_FACTOR_CORRECTED
        x_dir, y_dir = [Util.get_random_normal_direction() * diffusion_factor for _ in range(2)]
        newPos = tuple((x + x_dir, y + y_dir))
        
        if self.is_particle_on_boundary(newPos):
            if self.can_particle_hop_boundary_probability():
                newPos = self.make_particle_jump(newPos, x_dir, y_dir)
            else:
                newPos = Util.change_direction(tuple((x, y)), tuple((x_dir, y_dir)))
            
        self.paths[idx].append(newPos)
        
    def update(self):
        for i in range(self.number_of_particles): self.update_path(i)
        
    def init_particles(self) -> None:
        mem: list[tuple[int, int]] = []
        
        def get_random_canvas_value(self) -> int:
            return int(random.randint(-(CORRECTED_CANVAS_RADIUS), CORRECTED_CANVAS_RADIUS))
        
        def rec(self, x: int = 0, y: int = 0) -> tuple[int, int]:
            x, y = [get_random_canvas_value(self) for _ in range(2)]
            while (x, y) in mem or self.is_particle_on_boundary(tuple((x, y))):
                return rec(self, x, y)
            mem.append((x, y))
            return x, y
        
        self.particles_location.extend([rec(self) for _ in range(5)])
    
