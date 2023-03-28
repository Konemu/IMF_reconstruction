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
    fig.savefig(path+"field_x.png", dpi=300)
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

    cont = ax.pcolormesh(xs, ys, det, rasterized=True)
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
    fig.savefig(path+"det.png", dpi=300)
    plt.close(fig)


def plot_rel_errs_geometry(planet, R_bs_dist, R_mp_dist, n_r, xmin, xmax, ymin, ymax, path):
    xs, ys, relative_errs_mag = errors.relative_reconstruction_errors_geometry(planet, R_bs_dist, R_mp_dist, n_r, xmin, xmax, ymin, ymax)

    fig, ax = plt.subplots()

    from matplotlib.cm import ScalarMappable # https://stackoverflow.com/questions/54979958/set-colorbar-range-with-contourf-in-matplotlib

    quadcontourset = ax.pcolormesh(
        xs, ys, relative_errs_mag,  # change this to `levels` to get the result that you want
        vmin=0, vmax=0.15, rasterized=True
    )
    fig.colorbar(
        ScalarMappable(norm=quadcontourset.norm, cmap=quadcontourset.cmap), # type: ignore
    )


    #ax.contourf(xs, ys, relative_errs_mag)
    #cont = ax.contourf(xs, ys, relative_errs_mag, levels=200)
    #cbar = fig.colorbar(cont)



    ax.set_title("$\\delta |\\vec{B}| / B_0$")
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
    fig.suptitle(f"{n_r}x{n_r} grid"+", $\\tilde R_\\text{MP}=9.1$, $\\tilde R_\\text{BS}=12.6,$")
    fig.tight_layout()
    fig.savefig(path+"err_geometry.pdf")
    fig.savefig(path+"err_geometry.png", dpi=300)
    plt.close(fig)


def plot_rel_errs_field(planet, n_r, n_avg, sigma, xmin, xmax, ymin, ymax, path):
    xs, ys, err_X, err_Y, err_Z, err_mag = errors.relative_reconstruction_errors_field(planet, n_r, n_avg, sigma, xmin, xmax, ymin, ymax)

    errs = [[err_X, err_Y], [err_Z, err_mag]]
    labels = [["$\\delta B_x / B_{0,x}$", "$\\delta B_y / B_{0,y}$"], ["$\\delta B_z / B_{0,z}$", "$\\delta |\\vec{B}| / B_0$"]]

    fig, axes = plt.subplots(ncols=2, nrows=2)

    for axl, errl, labl in zip(axes, errs, labels):
        for ax, err, lab in zip(axl, errl, labl):
            cont = ax.pcolormesh(xs, ys, err, vmin=-0.4, vmax=0.4, rasterized=True, cmap="bwr")
            cbar = fig.colorbar(cont)
            ax.set_title(lab)
            ax.set_aspect(1)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

            ax.add_artist(plt.Circle((0, 0), planet.R_planet, color="black")) # type: ignore
            f_bs = planet.R_bowshock - planet.R_magnetopause / 2
            f_mp = planet.R_magnetopause / 2
            ax.plot(parab(ys, f_bs, planet.R_bowshock), ys, color="black", lw=0.2)
            ax.plot(parab(ys, f_mp, planet.R_magnetopause), ys, color="black", lw=0.2)

            ax.set_xlabel("$x$ ($R_E$)")
            ax.set_ylabel("$y$ ($R_E$)")
    
    fig.suptitle(f"$\\sigma={sigma}$ $B_0$, $n={n_avg}$ averages, {n_r}x{n_r} grid")
    
    fig.tight_layout()
    fig.savefig(path+"err_field.pdf")       
    fig.savefig(path+"err_field.png", dpi=300)       
    plt.close(fig)

def plot_rel_errs_pos(planet, n_r, n_avg, sigma, xmin, xmax, ymin, ymax, path):
    xs, ys, err_X, err_Y, err_Z, err_mag = errors.relative_reconstruction_errors_pos(planet, n_r, n_avg, sigma, xmin, xmax, ymin, ymax)

    errs = [[err_X, err_Y], [err_Z, err_mag]]
    labels = [["$\\delta B_x / B_{0,x}$", "$\\delta B_y / B_{0,y}$"], ["$\\delta B_z / B_{0,z}$", "$\\delta |\\vec{B}| / B_0$"]]

    fig, axes = plt.subplots(ncols=2, nrows=2)

    for axl, errl, labl in zip(axes, errs, labels):
        for ax, err, lab in zip(axl, errl, labl):
            maxerr = np.percentile(err, 99, axis=None)
            cont = ax.pcolormesh(xs, ys, err, vmin=-maxerr, vmax=maxerr, rasterized=True, cmap="bwr") # type: ignore
            cbar = fig.colorbar(cont)
            ax.set_title(lab)
            ax.set_aspect(1)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

            ax.add_artist(plt.Circle((0, 0), planet.R_planet, color="black")) # type: ignore
            f_bs = planet.R_bowshock - planet.R_magnetopause / 2
            f_mp = planet.R_magnetopause / 2
            ax.plot(parab(ys, f_bs, planet.R_bowshock), ys, color="black", lw=0.2)
            ax.plot(parab(ys, f_mp, planet.R_magnetopause), ys, color="black", lw=0.2)

            ax.set_xlabel("$x$ ($R_E$)")
            ax.set_ylabel("$y$ ($R_E$)")
    
    fig.suptitle("Field reconstructed from erronious position information.\\\\" + f"$\\sigma={np.round(sigma * 6371, 0)}$ km, $n={n_avg}$ averages, {n_r}x{n_r} grid \\\\ Colorbar maximum = 99th percentile")
    
    fig.tight_layout()
    fig.savefig(path+f"err_pos_sig_{np.round(sigma * 6371)}.pdf")       
    fig.savefig(path+f"err_pos_sig_{np.round(sigma * 6371)}.png", dpi=300)       
    plt.close(fig)


def plot_rel_errs_scaling(planet, n_r, sigmas, percentile, xmin, xmax, ymin, ymax, path):
    n_s = len(sigmas)
    mean_err_mag = np.zeros(n_s)
    mean_err_X = np.zeros(n_s)
    mean_err_Y = np.zeros(n_s)
    mean_err_Z = np.zeros(n_s)
    percentile_err_mag = np.zeros(n_s)
    percentile_err_X = np.zeros(n_s)
    percentile_err_Y = np.zeros(n_s)
    percentile_err_Z = np.zeros(n_s)

    for sigma, i in zip(sigmas, range(n_s)):
        xs, ys, err_X, err_Y, err_Z, err_mag = errors.relative_reconstruction_errors_pos(planet, n_r, 1, sigma, xmin, xmax, ymin, ymax)
        mean_err_mag[i] = err_mag.mean()
        mean_err_X[i] = err_X.mean()
        mean_err_Y[i] = err_Y.mean()
        mean_err_Y[i] = err_Y.mean()
        percentile_err_mag[i] = np.percentile(err_mag, percentile, axis=None)
        percentile_err_X[i] = np.percentile(err_X, percentile, axis=None)
        percentile_err_Y[i] = np.percentile(err_Y, percentile, axis=None)
        percentile_err_Z[i] = np.percentile(err_Z, percentile, axis=None)

    fig, ax = plt.subplots()
    #ax.plot(sigmas, mean_err_mag, label="Mean err magnitude")
    #ax.plot(sigmas, mean_err_X, label="Mean err X")
    #ax.plot(sigmas, mean_err_Y, label="Mean err Y")
    #ax.plot(sigmas, mean_err_Z, label="Mean err Y")
    ax.plot(sigmas, percentile_err_mag, label=f"{percentile}th percentile err magnitude")
    ax.plot(sigmas, percentile_err_X, label=f"{percentile}th percentile err X")
    ax.plot(sigmas, percentile_err_Y, label=f"{percentile}th percentile err Y")
    ax.plot(sigmas, percentile_err_Z, label=f"{percentile}th percentile err Z")
    
    #ax.set_title(lab)

    ax.set_xlabel("$\\sigma$ ($R_\\text{E}$)")
    ax.set_ylabel("$\\delta B$")
    ax.legend()
    ax.loglog()
    
    #fig.suptitle("Field reconstructed from erronious position information.\\\\" + f"$\\sigma={np.round(sigma * 6371, 0)}$ km, $n={n_avg}$ averages, {n_r}x{n_r} grid \\\\ Colorbar maximum = 99th percentile")
    
    fig.tight_layout()
    fig.savefig(path+f"err_scale.pdf")       
    fig.savefig(path+f"err_scale.png", dpi=300)       
    plt.close(fig)