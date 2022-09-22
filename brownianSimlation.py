from typing import List, Tuple
from matplotlib.pyplot import figure
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes,mark_inset
import numpy as np
import matplotlib.pyplot as plt

PATH = Tuple[List[float]]
DPI = 120

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
    global STEPS
    STEPS = 500
    
    def __init__(self, x0 = 0):
        assert (type(x0) == float or type(x0) == int or x0 is None), "Expect a float or None for the initial value"

        self.x0 = float(x0) 
        self._x, self._y = self.generateNormal(500)
    
    @property
    def get_path(self) -> PATH:
        return (self._x, self._y)
    
    def generateRandomWalks(self, steps:int = 100) -> PATH:
        w = np.ones(steps) * self.x0 # array of size steps of els 0.
        
        for i in range(1, steps):
            # Sampling from the Normal distribution with probability 1/2
            yi = np.random.choice([1, -1])
            # Weiner process
            w[i] = w[i - 1] + (yi / np.sqrt(steps))
        return w
    
    def generateNormalPath(self) -> List[float]:
        w = np.ones(STEPS) * self.x0
        for i in range(1, STEPS):
            p = np.random.normal()
            w[i] = w[i - 1] + (p / np.sqrt(STEPS))
        return w
    
    def generateNormal(self, steps = 100) -> PATH:
        wx = self.generateNormalPath()
        wy = self.generateNormalPath()
        return wx, wy 
    

NUMBER_OF_PATHS: int = 6

paths: List[PATH] = [Brownian().get_path for i in range(NUMBER_OF_PATHS)]
colors: List = ['r', 'b', "orange", 'g', 'y', 'c']
markers: List = ['o', 'v', '<', '>', 's', 'p']

fig, ax = plt.subplots(figsize=[5, 5])

ax.tick_params(axis='y',
               direction="in",
               right=True, labelsize=18)
ax.tick_params(axis='x', direction="in" , top=True,bottom=True, labelsize=18)

## legends and utilities
ax.set_xlabel(r"x", fontsize=16)
ax.set_ylabel(r"y", fontsize=16)

## border colors
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth('2') 

x = [0, 10]
ax.plot(x, x, markersize=300,markerfacecolor="r", color = 'r')
ax.set_xlim(-250, 250)
ax.set_ylim(-250, 250)
plt.show(block=True)

fig.tight_layout()