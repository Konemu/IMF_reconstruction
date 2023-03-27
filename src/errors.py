import planets

import numpy as np
from numba import njit
from numpy.random import default_rng

@njit
def get_exact_field(planet, n, xs, ys):
    BX = np.zeros((n, n), dtype=np.double)
    BY = np.zeros((n, n), dtype=np.double)
    BZ = np.zeros((n, n), dtype=np.double)
    for i in range(n):
        for j in range(n):            
            r = np.array([xs[i], ys[j], 0], dtype=np.double)
            if planet.check_vector_in_sheath(r):
                [BX[j][i], BY[j][i], BZ[j][i]] = planet.MS_from_IMF(r) # !!!
    return BX, BY, BZ


@njit
def get_reconstructed_field(planet, n, xs, ys, BX, BY, BZ):
    BSWX = np.zeros((n, n), dtype=np.double)
    BSWY = np.zeros((n, n), dtype=np.double)
    BSWZ = np.zeros((n, n), dtype=np.double)

    for i in range(n):
        for j in range(n):            
            r = np.array([xs[i], ys[j], 0], dtype=np.double)
            if planet.check_vector_in_sheath(r):
                [BSWX[j][i], BSWY[j][i], BSWZ[j][i]] = planet.IMF_from_MS(r, np.array([BX[j][i], BY[j][i], BZ[j][i]], dtype=np.double)) # !!!
            else:
                [BSWX[j][i], BSWY[j][i], BSWZ[j][i]] = planet.IMF
    return BSWX, BSWY, BSWZ


#@njit
def relative_reconstruction_errors_geometry(planet, R_bs_dist, R_mp_dist, n_r, xmin, xmax, ymin, ymax):
    IMF = planet.IMF
    B0  = np.linalg.norm(IMF)

    xs = np.linspace(xmin, xmax, n_r)
    ys = np.linspace(ymin, ymax, n_r)
    BX, BY, BZ = get_exact_field(planet, n_r, xs, ys)

    disturbed_planet = planets.Planet(R_planet=planet.R_planet, R_bowshock=R_bs_dist, R_magnetopause=R_mp_dist, IMF=IMF)
    BSWX_dist, BSWY_dist, BSWZ_dist = get_reconstructed_field(planet=disturbed_planet, n=n_r, xs=xs, ys=ys, BX=BX, BY=BY, BZ=BZ)

    relative_errs_x = np.abs((BSWX_dist - IMF[0]))/B0
    relative_errs_y = np.abs((BSWY_dist - IMF[1]))/B0
    relative_errs_z = np.abs((BSWZ_dist - IMF[2]))/B0
    relative_errs_mag = (np.sqrt(BSWX_dist**2 + BSWY_dist**2 + BSWZ_dist**2) - B0) / B0

    return xs, ys, relative_errs_x, relative_errs_y, relative_errs_z, relative_errs_mag



@njit # https://stackoverflow.com/questions/70613681/numba-compatible-numpy-meshgrid
def meshgrid(x, y, z):
    xx = np.empty(shape=(x.size, y.size, z.size), dtype=x.dtype)
    yy = np.empty(shape=(x.size, y.size, z.size), dtype=y.dtype)
    zz = np.empty(shape=(x.size, y.size, z.size), dtype=z.dtype)
    for i in range(z.size):
        for j in range(y.size):
            for k in range(x.size):
                xx[i,j,k] = k  # change to x[k] if indexing xy
                yy[i,j,k] = j  # change to y[j] if indexing xy
                zz[i,j,k] = i  # change to z[i] if indexing xy
    return zz, yy, xx