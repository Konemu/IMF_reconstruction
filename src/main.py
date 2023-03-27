import planets
import plotting
import errors

import numpy as np
import matplotlib as plt


def main():
    plt.rcParams.update({
    "text.usetex": True,
    })


    Earth = planets.Planet(R_planet=1, R_bowshock=12.5, R_magnetopause=9, IMF=np.asarray([-1, 0, 0], dtype=np.double))
    #plotting.plot_field_lines(planet=Earth, xmin=-10, xmax = 15, ymin=-20, ymax = 20, n=200, path="plots/")
    #plotting.plot_determinant(planet=Earth, xmin=-10, xmax = 15, ymin=-20, ymax = 20, n=200, path="plots/")
    plotting.plot_rel_errs_geometry(planet=Earth, R_bs_dist=12.6, R_mp_dist=9.1, n_r=50, xmin=-10, xmax = 15, ymin=-20, ymax = 20, path="plots/")

if __name__ == "__main__":
    main()