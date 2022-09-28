from simulation import *
from plotGenerator import *
from util import *
from brownianSimulation import *
from nanodomainSimulation import *    

#brownian = Brownian()
#nanoDomain = Nanodomain()
hopDiffusion = HopDiffusion();
plot(hopDiffusion, SimulationType.HOPDIFFUSION)