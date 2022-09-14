import numpy as np
import matplotlib.pyplot as plt

class Brownian():
    """
    Constructor, note that just like in C++ we can do default 
    initialization in function/construtor parameter
    """
    def __init__(self, x0 = 0):
        assert (type(x0) == float or type(x0) == int or x0 is None), "Expect a float or None for the initial value"
        
        self.x0 = float(x0)  # this.x = x;
    
    def generateRandomWalks(self, steps = 100):
        """
        steps is the number of data points we consider between each 
        random walk, it's the standard step size, higher for higher precision
        """
        w = np.ones(steps) * self.x0
        
        for i in range(1, steps):
            # Sampling from the Normal distribution with probability 1/2
            yi = np.random.choice([1,-1])
            # Weiner process
            w[i] = w[i-1] + (yi / np.sqrt(steps))
        
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
            w[i] = w[i-1] + (yi / np.sqrt(steps))
        
        return w
    
b1 = Brownian(1)
b2 = Brownian(1)

x = b1.generateNormal(1000)
y = b2.generateNormal(1000)

plt.plot(x,y,c='b')
xmax,xmin,ymax,ymin = x.max(),x.min(),y.max(),y.min()
scale_factor = 1.25
xmax,xmin,ymax,ymin = xmax*scale_factor,xmin*scale_factor,ymax*scale_factor,ymin*scale_factor
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.show(block=True)