import planets
import plotting
import errors

import numpy as np
import matplotlib.pyplot as plt

def study_pos_errors(R_planet, R_bowshock, R_magnetopause, n_r, IMFs, plot_sigma, sigmas, xmin=-10, xmax = 15, ymin=-20, ymax = 20, path="plots/"):
    for IMF in IMFs:
        IMF = np.asarray(IMF) / np.linalg.norm(IMF) # !
        imfpath = path + determine_path_from_IMF(IMF) + "/"
        planet = planets.Planet(R_planet=R_planet, R_bowshock=R_bowshock,
                                R_magnetopause=R_magnetopause, IMF=np.asarray(IMF, dtype=np.double))
        
        plotting.plot_rel_errs_pos(planet = planet, n_r = n_r, n_avg = 1, sigma = plot_sigma, xmin = 
                                    xmin, xmax = xmax, ymin = ymin, ymax = ymax, path = imfpath)
        plotting.plot_rel_errs_scaling(planet = planet, n_r = n_r, sigmas = sigmas, percentile = 99, 
                                       xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, path = imfpath)
            

def err_scale_orientation(R_planet, R_bowshock, R_magnetopause, n_r, IMFs, sigma, xmin=-10, xmax = 15, ymin=-20, ymax = 20, path="plots/"):
    mag_errs_yz = np.zeros(len(IMFs))
    mag_errs_xy = np.zeros(len(IMFs))
    mag_errs_xz = np.zeros(len(IMFs))
    IMF_yz = np.zeros(len(IMFs))
    IMF_zy = np.zeros(len(IMFs))
    IMF_xy = np.zeros(len(IMFs))
    IMF_yx = np.zeros(len(IMFs))
    IMF_xz = np.zeros(len(IMFs))
    IMF_zx = np.zeros(len(IMFs))
    for [IMF, i] in zip(IMFs, range(len(IMFs))):
        IMFyz = np.asarray([0, IMF[1], IMF[2]]) / np.linalg.norm(np.asarray([0, IMF[1], IMF[2]])) # !
        IMFxy = np.asarray([IMF[0], IMF[1], 0]) / np.linalg.norm(np.asarray([IMF[0], IMF[1], 0])) # !
        IMFxz = np.asarray([IMF[0], 0, IMF[2]]) / np.linalg.norm(np.asarray([IMF[0], 0, IMF[2]])) # !
        planetyz = planets.Planet(R_planet=R_planet, R_bowshock=R_bowshock,    
                                R_magnetopause=R_magnetopause, 
                                IMF=np.asarray(IMFyz, dtype=np.double))
        planetxy = planets.Planet(R_planet=R_planet, R_bowshock=R_bowshock,    
                                R_magnetopause=R_magnetopause, 
                                IMF=np.asarray(IMFxy, dtype=np.double))
        planetxz = planets.Planet(R_planet=R_planet, R_bowshock=R_bowshock,    
                                R_magnetopause=R_magnetopause, 
                                IMF=np.asarray(IMFxz, dtype=np.double))
        xs, ys, err_X, err_Y, err_Z, err_magyz = \
            errors.relative_reconstruction_errors_pos(planetyz, n_r, 1, sigma, xmin, xmax, ymin, ymax)
        xs, ys, err_X, err_Y, err_Z, err_magxy = \
            errors.relative_reconstruction_errors_pos(planetxy, n_r, 1, sigma, xmin, xmax, ymin, ymax)
        xs, ys, err_X, err_Y, err_Z, err_magxz = \
            errors.relative_reconstruction_errors_pos(planetxz, n_r, 1, sigma, xmin, xmax, ymin, ymax)
        mag_errs_yz[i] = np.percentile(err_magyz, 99, axis = None)
        mag_errs_xy[i] = np.percentile(err_magxy, 99, axis = None)
        mag_errs_xz[i] = np.percentile(err_magxz, 99, axis = None)
        IMF_yz[i] =  IMFyz[1] / IMFyz[2]
        IMF_zy[i] =  IMFyz[2] / IMFyz[1]
        IMF_xy[i] =  IMFxy[0] / IMFxy[1]
        IMF_yx[i] =  IMFxy[1] / IMFxy[0]
        IMF_xz[i] =  IMFxz[0] / IMFxz[2]
        IMF_zx[i] =  IMFxz[2] / IMFxz[0]

    fig, ax = plt.subplots()
    ax.scatter(IMF_yz, mag_errs_yz)
    ax.set_xlim([-10, 10])
    ax.set_xlabel("$B_{\\text{SW},y}/B_{\\text{SW},z}$")
    ax.set_ylabel("99th Perc. $|\\delta B|$")
    fig.tight_layout()
    fig.savefig(path+f"err_yz_{np.round(sigma*6371, 1)}.pdf")  
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.scatter(IMF_zy, mag_errs_yz)
    ax.set_xlim([-10, 10])
    ax.set_xlabel("$B_{\\text{SW},z}/B_{\\text{SW},y}$")
    ax.set_ylabel("99th Perc. $|\\delta B|$")
    fig.tight_layout()
    fig.savefig(path+f"err_zy_{np.round(sigma*6371, 1)}.pdf")  
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.scatter(IMF_xy, mag_errs_xy)
    ax.set_xlim([-10, 10])
    ax.set_xlabel("$B_{\\text{SW},x}/B_{\\text{SW},y}$")
    ax.set_ylabel("99th Perc. $|\\delta B|$")
    fig.tight_layout()
    fig.savefig(path+f"err_xy_{np.round(sigma*6371, 1)}.pdf")  
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.scatter(IMF_yx, mag_errs_xy)
    ax.set_xlim([-10, 10])
    ax.set_xlabel("$B_{\\text{SW},y}/B_{\\text{SW},x}$")
    ax.set_ylabel("99th Perc. $|\\delta B|$")
    fig.tight_layout()
    fig.savefig(path+f"err_yx_{np.round(sigma*6371, 1)}.pdf")  
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.scatter(IMF_xz, mag_errs_xz)
    ax.set_xlim([-10, 10])
    ax.set_xlabel("$B_{\\text{SW},x}/B_{\\text{SW},z}$")
    ax.set_ylabel("99th Perc. $|\\delta B|$")
    fig.tight_layout()
    fig.savefig(path+f"err_xz_{np.round(sigma*6371, 1)}.pdf")  
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.scatter(IMF_zx, mag_errs_xz)
    ax.set_xlim([-10, 10])
    ax.set_xlabel("$B_{\\text{SW},z}/B_{\\text{SW},x}$")
    ax.set_ylabel("99th Perc. $|\\delta B|$")
    fig.tight_layout()
    fig.savefig(path+f"err_zx_{np.round(sigma*6371, 1)}.pdf")  
    plt.close(fig)
    return


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