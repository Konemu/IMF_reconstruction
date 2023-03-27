import planet
import plotting

import numpy as np


def main():
    Earth = planet.Planet(R_planet=1, R_bowshock=12.5, R_magnetopause=9, IMF=np.asarray([-1, 0, 0], dtype=np.double))
    plotting.plot_field_lines(planet=Earth, xmin=-10, xmax = 15, ymin=-20, ymax = 20, n=200, path="plots/")
    plotting.plot_determinant(planet=Earth, xmin=-10, xmax = 15, ymin=-20, ymax = 20, n=200, path="plots/")

if __name__ == "__main__":
    main()