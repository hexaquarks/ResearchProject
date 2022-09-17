from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

PATH = Tuple[List[float]]

class Util:
    @staticmethod
    def get_bounds(lists: List[PATH]) -> Tuple[int]:
        X_MAX = max([max(elem[0]) for elem in lists])
        Y_MAX = max([max(elem[1]) for elem in lists])
        X_MIN = min([min(elem[0]) for elem in lists])
        Y_MIN = min([min(elem[1]) for elem in lists])
        return X_MIN, X_MAX, Y_MIN, Y_MAX
    
    @staticmethod
    def get_last_point(path: PATH) -> Tuple[int]:
        return path[0][-1], path[1][-1]

class Brownian:
    """
    Constructor, note that just like in C++ we can do default 
    initialization in function/construtor parameter
    """
    def __init__(self, x0 = 0):
        assert (type(x0) == float or type(x0) == int or x0 is None), "Expect a float or None for the initial value"
        
        self.x0 = float(x0) 
        self._x, self._y = self.generateNormal()
    
    @property
    def get_path(self) -> PATH:
        return (self._x, self._y)
    
    def generateRandomWalks(self, steps:int = 100) -> PATH:
        """
        steps is the number of data points we consider between each 
        random walk, it's the standard step size, higher for higher precision
        """
        w = np.ones(steps) * self.x0 # array of size steps of els 0.
        
        for i in range(1, steps):
            # Sampling from the Normal distribution with probability 1/2
            yi = np.random.choice([1, -1])
            # Weiner process
            w[i] = w[i - 1] + (yi / np.sqrt(steps))
        return w
    
    def generateNormal(self, steps = 100) -> PATH:
        """
        Generate motion by drawing from the Normal distribution
        """
        wx = np.ones(steps) * self.x0
        wy = np.ones(steps) * self.x0
        
        for i in range(1, steps):
            # Sampling from the Normal distribution
            xi = np.random.normal()
            yi = np.random.normal()
            # Weiner process
            wx[i] = wx[i - 1] + (xi / np.sqrt(steps))
            wy[i] = wy[i - 1] + (yi / np.sqrt(steps))
        return wx, wy

STEPS: int = 1000
NUMBER_OF_PATHS: int = 6

paths: List[PATH] = [Brownian().get_path for i in range(NUMBER_OF_PATHS)]
colors: List = ['r', 'b', "orange", 'g', 'y', 'c']
markers: List = ['o', 'v', '<', '>', 's', 'p']

for i, path in enumerate(paths):
    plt.plot(path[0], path[1], c = colors[i])
    
    last_point = Util.get_last_point(path)
    print("x,y : " ,last_point[0] , " ", last_point[1])
    plt.plot(last_point[0], last_point[1], 
             marker = markers[i], markersize = 12,
             markerfacecolor = colors[i])
    
x_min, x_max, y_min, y_max = Util.get_bounds(paths)
MAX = max(abs(el) for el in [x_min, x_max, y_min, y_max])
MAX *= 1.1
plt.xlim(-MAX, MAX)
plt.ylim(-MAX, MAX)
plt.show(block=True)