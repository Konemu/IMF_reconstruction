import planet
import numpy as np
import matplotlib.pyplot as plt

def parab(y, f, R):
    return -(y**2) / (4 * f) + R

def plot_field_lines(planet, xmin, xmax, ymin, ymax, n, path):    
    xs = np.linspace(xmin, xmax, n)
    ys = np.linspace(ymin, ymax, n)    
    xx, yy = np.meshgrid(xs, ys)
    BX, BY = np.meshgrid(np.zeros(n), np.zeros(n))

    for i in range(n):
        for j in range(n):            
            r = np.array([xs[i], ys[j], 0], dtype=np.double)
            if planet.check_vector_in_sheath(r):
                [BX[j][i], BY[j][i], disc] = planet.MS_from_IMF(r)  # !!!

    fig, ax = plt.subplots()

    ax.streamplot(xs, ys, BX, BY, density=2, linewidth=0.3)

    ax.add_artist(plt.Circle((0, 0), planet.R_planet, color="black")) # type: ignore
    f_bs = planet.R_bowshock - planet.R_magnetopause / 2
    f_mp = planet.R_magnetopause / 2
    ax.plot(parab(ys, f_bs, planet.R_bowshock), ys, color="black")
    ax.plot(parab(ys, f_mp, planet.R_magnetopause), ys, color="black")

    ax.set_aspect(1)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    fig.savefig(path+"test.pdf")

def plot_determinant(planet, xmin, xmax, ymin, ymax, n, path):    
    xs = np.linspace(xmin, xmax, n)
    ys = np.linspace(ymin, ymax, n)        
    det = np.zeros((n, n), dtype=np.double)

    for i in range(n):
        for j in range(n):            
            r = np.array([xs[i], ys[j], 0], dtype=np.double)
            if planet.check_vector_in_sheath(r):
                det[j][i] = np.linalg.det( planet.trans_mat(r) )  # !!!

    fig, ax = plt.subplots()

    cont = ax.contourf(xs, ys, det, levels=20)
    cbar = fig.colorbar(cont)

    ax.add_artist(plt.Circle((0, 0), planet.R_planet, color="black")) # type: ignore
    f_bs = planet.R_bowshock - planet.R_magnetopause / 2
    f_mp = planet.R_magnetopause / 2
    ax.plot(parab(ys, f_bs, planet.R_bowshock), ys, color="black")
    ax.plot(parab(ys, f_mp, planet.R_magnetopause), ys, color="black")

    ax.set_aspect(1)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    fig.savefig(path+"det.pdf")
