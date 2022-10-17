from simulations.simulation import *
from simulations.brownian import *
from simulations.nanodomain import *    
from simulations.hopDiffusion import *    
from simulations.spaceTimeCorrelationManager import *  

from plotGenerator import *
from util import *

def main() -> None:
    rcParams.update({'figure.autolayout': True})
    # brownian = Brownian(6)
    nanoDomain = Nanodomain(6)
    
    # hopDiffusion = HopDiffusion(6);
    space_time_correlation_manager = SpaceTimeCorrelationManager(nanoDomain)
    
    plotGenerator = PlotGenerator(nanoDomain, space_time_correlation_manager)
    plotGenerator.start_animation()

if __name__ == '__main__':
    main()
