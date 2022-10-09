from simulations.simulation import *
from simulations.brownian import *
from simulations.nanodomain import *
from simulations.hopDiffusion import *

from plotGenerator import *
from util import *

def main() -> None:
    rcParams.update({'figure.autolayout': True})
    # brownian = Brownian()
    # nanoDomain = Nanodomain()

    hopDiffusion = HopDiffusion(10)
    plotGenerator = PlotGenerator(hopDiffusion, SimulationType.HOPDIFFUSION)
    plotGenerator.start_animation()

if __name__ == '__main__':
    main()
