import planets
import plotting
import param_study

import numpy as np
import matplotlib as plt

from numba import set_num_threads
set_num_threads(6)
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

import time


def main():
    plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{parskip}",
    "axes.labelsize": 22,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.titlesize": 24
    })

    # Geometry and resolution
    R_planet=1
    R_bowshock=12.5
    R_magnetopause=9
    n_r = 200
    xmin = -10
    xmax = 15
    ymin = -20
    ymax = 20


    begin = time.time()
    
    # Fields
    IMFs = [[ 1, 0, 0], # are normalised elsewhere
            [ 0, 1, 0],
            [ 0, 0, 1],
            [ 1, 0, 1],
            [-1, 0, 1],
            [-1, 0,-1],
            [ 1, 0,-1],
            [ 1, 1, 0],
            [ 0, 1, 1],
            [ 1, 1, 1],
            [-1, 0, 0],
            [ 0,-1, 0],
            [ 0, 0,-1],
            [-1,-1, 0],
            [ 0,-1,-1],
            [-1,-1,-1],
            [ 1,-1, 0],
            [ 0, 1,-1],
            [-1, 1, 0],
            [ 0,-1, 1],
            [ 1,-1, 1],
            [-1, 1, 1],
            [ 1, 1,-1],
            [-1,-1, 1],
            [ 1,-1,-1]]
    plot_sigma = 10/6371 # std dev for plotting spatially resolved component errors
    sigmas = np.array([1, 5, 10, 20, 50, 100, 200, 500, 1000, 6371]) / 6371 # std devs for plotting max err scaling

    # plot spatially resolved component errors and max err scaling for all IMFs above
    param_study.study_pos_errors(R_planet, R_bowshock, R_magnetopause, n_r, IMFs, plot_sigma, sigmas, xmin, xmax, ymin, ymax, path="plots/param_study_pos/")
    
    # generate random fields and plot component <-> error correlations
    IMFs = []
    for i in range(100):
        IMFs.append([np.random.normal(0, 1), np.random.normal(0, 1), np.random.normal(0, 1)])
    param_study.err_scale_orientation(R_planet=1, R_bowshock=12.5, R_magnetopause=9, n_r=200, IMFs = IMFs, sigma=10/6731, xmin=-10, xmax = 15, ymin=-20, ymax = 20, path="plots/components/")
    
    # print time elapsed
    end = time.time()
    print(end-begin, "s")
    

if __name__ == "__main__":
    main()


