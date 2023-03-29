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
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{parskip}"
    })

    begin = time.time()
    R_planet=1
    R_bowshock=12.5
    R_magnetopause=9
    n_r = 200
    xmin = -10
    xmax = 15
    ymin = -20
    ymax = 20
    
    IMFs = [[ 1, 0, 0],
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
    sigmas = np.array([10, 20, 50, 100, 200, 500]) / 6371

    param_study.study_pos_errors(R_planet, R_bowshock, R_magnetopause, n_r, IMFs, sigmas, xmin, xmax, ymin, ymax, path="plots/param_study_pos/")    

    end = time.time()
    print(end-begin, "s")

if __name__ == "__main__":
    main()


