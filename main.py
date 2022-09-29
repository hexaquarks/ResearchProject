from simulations.simulation import *
from simulations.brownianSimulation import *
from simulations.nanodomainSimulation import *    

from plotGenerator import *
from util import *

#brownian = Brownian()
#nanoDomain = Nanodomain()
hopDiffusion = HopDiffusion();
plot(hopDiffusion, SimulationType.HOPDIFFUSION)