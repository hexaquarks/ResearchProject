from simulations.simulation import *
from simulations.brownian import *
from simulations.nanodomain import *    
from simulations.hopDiffusion import *    

from plotGenerator import *
from util import *

#brownian = Brownian()
#nanoDomain = Nanodomain()
hopDiffusion = HopDiffusion();
plot(hopDiffusion, SimulationType.HOPDIFFUSION)