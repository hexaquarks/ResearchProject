from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

class Brownian:
    """
    Constructor, note that just like in C++ we can do default 
    initialization in function/construtor parameter
    """
    def __init__(self, x0 = 0):
        assert (type(x0) == float or type(x0) == int or x0 is None), "Expect a float or None for the initial value"
        
        self.x0 = float(x0)  # this.x = x;
        self.path = []
        
    
    def generateRandomWalks(self, steps:int = 100):
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
    
    def generateNormal(self, steps = 100):
        """
        Generate motion by drawing from the Normal distribution
        """
        w = np.ones(steps) * self.x0
        
        for i in range(1, steps):
            # Sampling from the Normal distribution
            yi = np.random.normal()
            # Weiner process
            w[i] = w[i - 1] + (yi / np.sqrt(steps))
            self.path = w

STEPS: int = 1000
NUMBER_OF_PATHS: int = 7
paths: List[Tuple[Brownian]] = []
colors: List = ['r', 'b', "orange", 'g', 'y', 'c', 'm']

for i in range(NUMBER_OF_PATHS):
    paths.append((Brownian(), Brownian()))
    paths[i][0].generateNormal(STEPS)
    paths[i][1].generateNormal(STEPS)
    
for i in range(NUMBER_OF_PATHS):
    item: Tuple[Brownian] = paths[i]
    plt.plot(item[0].path, item[1].path, c = colors[i])
    
# xmax,xmin,ymax,ymin = x.max(),x.min(),y.max(),y.min()
# scale_factor = 1.25
# xmax,xmin,ymax,ymin = xmax*scale_factor,xmin*scale_factor,ymax*scale_factor,ymin*scale_factor
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.show(block=True)