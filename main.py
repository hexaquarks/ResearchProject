from simulations.simulation import Simulation
from simulations.brownian import Brownian
from simulations.nanodomain import Nanodomain
from simulations.hopDiffusion import HopDiffusion   
from simulations.imageManager import ImageManager

from matplotlib import rcParams # type: ignore
from plotGenerator import PlotGenerator
from util import *

def main() -> None:
    rcParams.update({'figure.autolayout': True})
    brownian = Brownian(15, True)
    #nanoDomain = Nanodomain(15, True)
    
    #hopDiffusion = HopDiffusion(15, True);
    image_manager = ImageManager(brownian)
    
    plotGenerator = PlotGenerator(brownian, image_manager)
    plotGenerator.start_animation()
    plotGenerator.initialize_space_correlation_manager()

if __name__ == '__main__':
    main()
