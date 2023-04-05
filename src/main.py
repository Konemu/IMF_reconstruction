import planets
import plotting
import param_study

import numpy as np
import matplotlib as plt

from numba import set_num_threads
set_num_threads(20)
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
    #Earth = planets.Planet(R_planet=1, R_bowshock=12.5, R_magnetopause=9, IMF=np.asarray([-1, 0, 0], dtype=np.double))
    #plotting.plot_condition(planet=Earth, xmin=-10, xmax = 15, ymin=-20, ymax = 20, n=1000, path="plots/vortrag1/")
    #plotting.plot_determinant(planet=Earth, xmin=-10, xmax = 15, ymin=-20, ymax = 20, n=1000, path="plots/vortrag1/")
    #plotting.plot_rel_errs_geometry(planet=Earth, R_bs_dist=12.6, R_mp_dist=9.1, n_r=1000, xmin=-10, xmax = 15, ymin=-20, ymax = 20, path="plots/vortrag1/")
    #sigma = 0.1
    #plotting.plot_rel_errs_field(planet=Earth, n_r=1000, n_avg=1, sigma=sigma, xmin=-10, xmax = 15, ymin=-20, ymax = 20, path="plots/vortrag1/")

    begin = time.time()
    R_planet=1
    R_bowshock=12.5
    R_magnetopause=9
    n_r = 200
    xmin = -10
    xmax = 15
    ymin = -20
    ymax = 20
    
    IMFs = [[ 1, 0, 0], # are normalised elsewhere
            [ 0, 1, 0],
            [ 0, 0, 1],
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
    plot_sigma = 10/6371
    sigmas = np.array([1, 5, 10, 20, 50, 100, 200, 500, 1000, 6371]) / 6371

    param_study.study_pos_errors(R_planet, R_bowshock, R_magnetopause, n_r, IMFs, plot_sigma, sigmas, xmin, xmax, ymin, ymax, path="plots/param_study_pos/")    

    end = time.time()
    print(end-begin, "s")
    

if __name__ == "__main__":
    main()


