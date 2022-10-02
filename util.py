import numpy as np
import random

class Util:
    @staticmethod
    def get_bounds(lists) -> tuple[int, ...]:
        x_min: int = min([min(elem[0]) for elem in lists])
        x_max: int = max([max(elem[0]) for elem in lists])
        y_min: int = min([min(elem[1]) for elem in lists])
        y_max: int = max([max(elem[1]) for elem in lists])
        return x_min, x_max, y_min, y_max

    @staticmethod
    def compute_distance(p1: tuple, p2: tuple) -> float:
        return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    @staticmethod
    def get_last_point(path: list[tuple]) -> tuple[int, ...]:
        return path[-1][0], path[-1][1]

    @staticmethod
    def get_x_coordinates(path) -> list:
        return list(list(zip(*path))[0])

    @staticmethod
    def get_y_coordinates(path) -> list:
        return list(list(zip(*path))[1])

    @staticmethod
    def get_random_normal_direction():
        return np.random.normal() * np.random.choice([1, -1])

    @staticmethod
    def is_point_within_bounds(pos: tuple, bounds: tuple[tuple, ...]):
        x, y = pos[0], pos[1]
        return x >= bounds[0][0] and x <= bounds[0][1] and y >=  bounds[1][0] and y <= bounds[1][1]
    
    @staticmethod
    def sign(x):
        return ((x >= 0) << 1) - 1
    
    @staticmethod
    def increment_tuple_by_val(tuple_object: tuple, val):
        tuple_object = tuple((tuple_object[0] + val[0], tuple_object[1] + val[1]))
        return tuple_object
    
    @staticmethod
    def change_direction(tuple_object: tuple, dir):
        tuple_object = tuple((tuple_object[0] - dir[0], tuple_object[1] - dir[1]))
        return tuple_object
