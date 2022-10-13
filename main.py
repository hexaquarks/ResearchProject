from simulations.simulation import *
from simulations.brownian import *
from simulations.nanodomain import *    
from simulations.hopDiffusion import *    
from simulations.spaceTimeCorrelationManager import *  

from plotGenerator import *
from util import *

def main() -> None:
    rcParams.update({'figure.autolayout': True})
    # brownian = Brownian()
    # nanoDomain = Nanodomain()
    
    hopDiffusion = HopDiffusion(3);
    space_time_correlation_manager = SpaceTimeCorrelationManager(hopDiffusion)
    
    plotGenerator = PlotGenerator(hopDiffusion, space_time_correlation_manager)
    plotGenerator.start_animation()

if __name__ == '__main__':
    main()
