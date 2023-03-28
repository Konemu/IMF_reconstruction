import planets
import plotting
import errors

import numpy as np
import matplotlib as plt

from numba import set_num_threads
set_num_threads(20)

import time


def main():
    plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{parskip}"
    })

    begin = time.time()

    Earth = planets.Planet(R_planet=1, R_bowshock=12.5, R_magnetopause=9, IMF=np.asarray([-1, 0, 0], dtype=np.double))
    #plotting.plot_field_lines(planet=Earth, xmin=-10, xmax = 15, ymin=-20, ymax = 20, n=200, path="plots/")
    #plotting.plot_determinant(planet=Earth, xmin=-10, xmax = 15, ymin=-20, ymax = 20, n=1000, path="plots/")
    #plotting.plot_rel_errs_geometry(planet=Earth, R_bs_dist=12.6, R_mp_dist=9.1, n_r=1000, xmin=-10, xmax = 15, ymin=-20, ymax = 20, path="plots/")
    #plotting.plot_rel_errs_field(planet=Earth, n_r=200, n_avg=1, sigma=0.1, xmin=-10, xmax = 15, ymin=-20, ymax = 20, path="plots/")
    
    
    sigs = np.array([10, 20, 50, 100, 200, 500]) / 6371

    for sig in sigs:
        plotting.plot_rel_errs_pos(planet=Earth, n_r=200, n_avg=1, sigma=sig, xmin=-10, xmax = 15, ymin=-20, ymax = 20, path="plots/")

    end = time.time()
    print(end-begin, " s")

if __name__ == "__main__":
    main()