from simulations.simulation import *
from simulations.brownian import *
from simulations.nanodomain import *    
from simulations.hopDiffusion import *    
from simulations.imageManager import *  

from matplotlib import rcParams # type: ignore
from plotGenerator import *
from util import *

def main() -> None:
    rcParams.update({'figure.autolayout': True})
    # brownian = Brownian(6)
    # nanoDomain = Nanodomain(6)
    
    hopDiffusion = HopDiffusion(6);
    image_manager = ImageManager(hopDiffusion)
    
    plotGenerator = PlotGenerator(hopDiffusion, image_manager)
    plotGenerator.start_animation()
    plotGenerator.initialize_space_correlation_manager()
    

if __name__ == '__main__':
    main()
