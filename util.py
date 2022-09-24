from typing import List, Tuple

class Util:
    @staticmethod
    def get_bounds(lists) -> Tuple[int]:
        X_MAX = max([max(elem[0]) for elem in lists])
        Y_MAX = max([max(elem[1]) for elem in lists])
        X_MIN = min([min(elem[0]) for elem in lists])
        Y_MIN = min([min(elem[1]) for elem in lists])
        return X_MIN, X_MAX, Y_MIN, Y_MAX
    
    @staticmethod
    def get_last_point(path) -> Tuple[int]:
        return path[0][-1], path[1][-1]
    
    @staticmethod
    def get_x_coordinates(path) -> List:
        return list(list(zip(*path))[0])
    
    @staticmethod
    def get_y_coordinates(path) -> List:
        return list(list(zip(*path))[1])