from typing import List, Tuple
import numpy as np
import random

class Util:
    @staticmethod
    def get_bounds(lists) -> Tuple[int]:
        X_MAX = max([max(elem[0]) for elem in lists])
        Y_MAX = max([max(elem[1]) for elem in lists])
        X_MIN = min([min(elem[0]) for elem in lists])
        Y_MIN = min([min(elem[1]) for elem in lists])
        return X_MIN, X_MAX, Y_MIN, Y_MAX
    
    @staticmethod
    def compute_distance(p1: Tuple, p2: Tuple) -> float:
        return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        
    @staticmethod
    def get_last_point(path) -> Tuple[int]:
        return path[-1][0], path[-1][1]
    
    @staticmethod
    def get_x_coordinates(path) -> List:
        return list(list(zip(*path))[0])
    
    @staticmethod
    def get_y_coordinates(path) -> List:
        return list(list(zip(*path))[1])
    
    @staticmethod
    def get_random_normal_direction():
        return np.random.normal() * np.random.choice([1, -1])