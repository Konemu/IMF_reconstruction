import planets
import plotting

import numpy as np

def study_pos_errors(R_planet, R_bowshock, R_magnetopause, n_r, IMFs, plot_sigma, sigmas, xmin=-10, xmax = 15, ymin=-20, ymax = 20, path="plots/"):
    for IMF in IMFs:
        IMF = np.asarray(IMF) / np.linalg.norm(IMF) # !
        imfpath = path + determine_path_from_IMF(IMF) + "/"
        planet = planets.Planet(R_planet=R_planet, R_bowshock=R_bowshock, R_magnetopause=R_magnetopause, IMF=np.asarray(IMF, dtype=np.double))
        
        plotting.plot_rel_errs_pos(planet = planet, n_r = n_r, n_avg = 1, sigma = plot_sigma, xmin = 
                                    xmin, xmax = xmax, ymin = ymin, ymax = ymax, path = imfpath)
        plotting.plot_rel_errs_scaling(planet = planet, n_r = n_r, sigmas = sigmas, percentile = 99, 
                                       xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, path = imfpath)
            


def determine_path_from_IMF(IMF):
    ret = ""
    if IMF[0] > 0:
        ret += "x"
    elif IMF[0] < 0:
        ret += "mx"
    if IMF[1] > 0:
        ret += "y"
    elif IMF[1] < 0:
        ret += "my"
    if IMF[2] > 0:
        ret += "z"
    elif IMF[2] < 0:
        ret += "mz"

    return ret