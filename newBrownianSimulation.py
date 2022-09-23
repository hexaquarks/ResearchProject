from typing import List, Tuple
from matplotlib.pyplot import figure
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes,mark_inset
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
import random

PATH = Tuple[List[float]]
DPI = 120
RADIUS = 250
RADIUS_PADDING = 10

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

        self.particlesLocation: List[Tuple[int]] = [] # to track particles position in time
        self.paths: List[List[Tuple[int]]] = []
        self.initializeGrid()
        self.initializePaths()
        #self.initializeParticles()
        
        self.x0 = float(x0) 
        self._x, self._y = self.generateNormal(500)
    
    def initializePaths(self):
        for coordinate in self.particlesLocation:
            self.paths.append([coordinate])
            
    def updatePath(self, idx):
        x_dir, y_dir = [np.random.normal() * np.random.choice([1, -1]) * 3 for _ in range(2)]
        x, y = self.paths[idx][-1]
        
        #print("appending at ", idx, " over size : ", len(paths))
        self.paths[idx].append((x + x_dir, y + y_dir))
        
    def update(self):
        for i in range(len(self.paths)):
            self.updatePath(i);
            
    @property
    def get_path(self) -> PATH:
        return (self._x, self._y)
    
    def get_x_coords(self, idx) -> List:
        return list(list(zip(*self.paths[idx]))[0])
    
    def get_y_coords(self, idx) -> List:
        return list(list(zip(*self.paths[idx]))[1])
    
    def initializeGrid(self):
        mem: List[Tuple] = []
        
        def recc(self, x = 0, y = 0):
            x = int(random.randint(-(RADIUS - RADIUS_PADDING), RADIUS - RADIUS_PADDING))
            y = int(random.randint(-(RADIUS - RADIUS_PADDING), RADIUS - RADIUS_PADDING))
            while (x, y) in mem:
                return recc(self, x, y)
            mem.append((x, y))
            return x,y
                
        for i in range(5):
            self.particlesLocation.append(recc(self))
        print(self.particlesLocation)
    
    def initializeParticles(self, nbPoints: int = 10):
        np.put(self.grid, np.random.choice(
            range(RADIUS * RADIUS), nbPoints, replace = False), 1)
        for i in range(RADIUS):
            for j in range(RADIUS):
                if self.grid[i][j] == 1: self.particlesLocation.append((i, j))
        
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

#paths: List[PATH] = [Brownian().get_path for i in range(NUMBER_OF_PATHS)]
colors: List = ['r', 'b', "orange", 'g', 'y', 'c']
markers: List = ['o', 'v', '<', '>', 's', 'p']
b = Brownian()

fig, ax = plt.subplots(figsize=[5, 5], dpi = 100)

plots: List = [ax.plot(b.get_x_coords(i), b.get_y_coords(i), markersize=15, color = colors[i])[0] for i in range(5)] 

def init():
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

    ax.set_xlim(-250, 250)
    ax.set_ylim(-250, 250)
    return plots

def update(frame):
    b.update()
    for i, plot in enumerate(plots):
        plot.set_data(b.get_x_coords(i), b.get_y_coords(i))
    return plots

temp = np.arange(-100, 100, 1.0).tolist()
ani = FuncAnimation(fig, update, frames=temp,
                    init_func=init, blit=True)

# for i, path in enumerate(paths):
#     ax.plot(path[0], path[1], c = colors[i])
    
#     last_point = Util.get_last_point(path)
#     ax.plot(last_point[0], last_point[1], 
#              marker = markers[i], markersize = 17,
#              markerfacecolor = colors[i])
    
# x_min, x_max, y_min, y_max = Util.get_bounds(paths)
# MAX = max(abs(el) for el in [x_min, x_max, y_min, y_max])
# MAX *= 1.1
# ax.set_xlim(-MAX, MAX)
# ax.set_ylim(-MAX, MAX)

#plt.imshow(b.grid, cmap='Greys', interpolation='none')


#plt.figure(figsize = (128 / DPI, 128 / DPI), dpi = DPI)

plt.show(block=True)

fig.tight_layout()