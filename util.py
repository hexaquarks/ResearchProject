import numpy as np

class Util:  # convert this to an import of a module with global functions
    @staticmethod
    def compute_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
        return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    @staticmethod
    def get_last_point(path: list[tuple[float, float]]) -> tuple[float, float]:
        return path[-1][0], path[-1][1]

    @staticmethod
    def get_x_coordinates(path: list[tuple[float, float]]) -> list[float]:
        return list(list(zip(*path))[0])

    @staticmethod
    def get_y_coordinates(path: list[tuple[float, float]]) -> list[float]:
        return list(list(zip(*path))[1])

    @staticmethod
    def get_random_normal_direction() -> float:
        return np.random.normal() * np.random.choice([1, -1])

    @staticmethod
    def is_point_within_bounds(
        pos: tuple[float, float],
        bounds: tuple[tuple[float, float], tuple[float, float]],
    ) -> bool:
        x, y = pos[0], pos[1]
        return (
            bounds[0][0] <= x <= bounds[0][1] and
            bounds[1][0] <= y <= bounds[1][1]
        )

    @staticmethod
    def increment_tuple_by_val(tuple_object: tuple[float, float], val: tuple[float, float]) -> tuple[float, float]:
        return tuple_object[0] + val[0], tuple_object[1] + val[1]

    @staticmethod
    def change_direction(
        tuple_object: tuple[float, float],
        direction: tuple[float, float],
    ) -> tuple[float, float]:
        return tuple_object[0] - direction[0], tuple_object[1] - direction[1]