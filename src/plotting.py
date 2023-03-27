import planets
import errors
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

    ax.set_xlabel("$x$ ($R_E$)")
    ax.set_ylabel("$y$ ($R_E$)")
    ax.set_title("$\\vec{B}$ field lines for $\\vec{B}_{SW} = -B_0 \\vec{e}_x$")

    fig.tight_layout()
    fig.savefig(path+"field_x.pdf")
    fig.savefig(path+"field_x.png")
    plt.close(fig)

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

    ax.set_xlabel("$x$ ($R_E$)")
    ax.set_ylabel("$y$ ($R_E$)")
    ax.set_title("$\\det T$")

    fig.tight_layout()
    fig.savefig(path+"det.pdf")
    fig.savefig(path+"det.png")
    plt.close(fig)


def plot_rel_errs_geometry(planet, R_bs_dist, R_mp_dist, n_r, xmin, xmax, ymin, ymax, path):
    xs, ys, relative_errs_x, relative_errs_y, relative_errs_z, relative_errs_mag \
        = errors.relative_reconstruction_errors_geometry(planet, R_bs_dist, R_mp_dist, n_r, xmin, xmax, ymin, ymax)
    errs = [[relative_errs_x, relative_errs_y], [relative_errs_z, relative_errs_mag]]
    lab  = [["$\\delta B_x / B_0$", "$\\delta B_y / B_0$"], ["$\\delta B_z / B_0$", "$\\delta |\\vec{B}| / B_0$"]]

    fig, axes = plt.subplots(nrows=2, ncols=2)
    for axl, errl, labl in zip(axes, errs, lab):
        for ax, err, lab in zip(axl, errl, labl):
            ax.contourf(xs, ys, err)
            cont = ax.contourf(xs, ys, err, levels=200)
            cbar = fig.colorbar(cont)
            ax.set_title(lab)
            ax.set_aspect(1)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

            ax.add_artist(plt.Circle((0, 0), planet.R_planet, color="black")) # type: ignore
            f_bs = planet.R_bowshock - planet.R_magnetopause / 2
            f_mp = planet.R_magnetopause / 2
            ax.plot(parab(ys, f_bs, planet.R_bowshock), ys, color="black")
            ax.plot(parab(ys, f_mp, planet.R_magnetopause), ys, color="black")

            ax.set_xlabel("$x$ ($R_E$)")
            ax.set_ylabel("$y$ ($R_E$)")
    fig.tight_layout()
    fig.savefig(path+"test.pdf")
    