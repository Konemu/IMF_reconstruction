import planets

import numpy as np
from numba import njit, prange


@njit
def get_exact_field(planet, n, xs, ys):
    BX = np.zeros((n, n), dtype=np.double)
    BY = np.zeros((n, n), dtype=np.double)
    BZ = np.zeros((n, n), dtype=np.double)
    for i in range(n):
        for j in range(n):            
            r = np.array([xs[i], ys[j], 0], dtype=np.double)
            [BX[j][i], BY[j][i], BZ[j][i]] = planet.MS_from_IMF(r) # !!!
    return BX, BY, BZ


@njit
def get_reconstructed_field(planet, n, xs, ys, BX, BY, BZ):
    BSWX = np.zeros((n, n), dtype=np.double)
    BSWY = np.zeros((n, n), dtype=np.double)
    BSWZ = np.zeros((n, n), dtype=np.double)

    for i in range(n):
        for j in prange(n):            
            r = np.array([xs[i], ys[j], 0], dtype=np.double)
            [BSWX[j][i], BSWY[j][i], BSWZ[j][i]] = planet.IMF_from_MS(r, np.array([BX[j][i], BY[j][i], BZ[j][i]], dtype=np.double)) # !!!           
    return BSWX, BSWY, BSWZ


@njit
def get_reconstructed_field_pos_errs(planet, n, xs, ys, zs, BX, BY, BZ):
    BSWX = np.zeros((n, n), dtype=np.double)
    BSWY = np.zeros((n, n), dtype=np.double)
    BSWZ = np.zeros((n, n), dtype=np.double)

    for i in range(n):
        for j in range(n):            
            r = np.array([xs[j][i], ys[j][i], zs[j][i]], dtype=np.double)
            [BSWX[j][i], BSWY[j][i], BSWZ[j][i]] = planet.IMF_from_MS(r, np.array([BX[j][i], BY[j][i], BZ[j][i]], dtype=np.double)) # !!!           
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

    relative_errs_mag = np.zeros((n_r, n_r), dtype=np.double)
    for i in range(n_r):
        for j in range(n_r):            
            r = np.array([xs[i], ys[j], 0], dtype=np.double)
            if planet.check_vector_in_sheath(r):
                relative_errs_mag[j][i] = ( np.sqrt(BSWX_dist[j][i]**2 + BSWY_dist[j][i]**2 + BSWZ_dist[j][i]**2) - B0) / B0
    

    return xs, ys, relative_errs_mag


@njit
def relative_reconstruction_errors_field(planet, n_r, n_avg, sigma, xmin, xmax, ymin, ymax):
    IMF = planet.IMF    
    B0  = np.linalg.norm(IMF)

    xs = np.linspace(xmin, xmax, n_r)
    ys = np.linspace(ymin, ymax, n_r)
    BX, BY, BZ = get_exact_field(planet, n_r, xs, ys)
    err_X = np.zeros((n_r, n_r), dtype=np.double)
    err_Y = np.zeros((n_r, n_r), dtype=np.double)
    err_Z = np.zeros((n_r, n_r), dtype=np.double)
    err_mag = np.zeros((n_r, n_r), dtype=np.double)
    
    for m in range(n_avg):
        BX_dev = BX + np.random.normal(loc=0, scale=sigma, size=(n_r, n_r))
        BY_dev = BY + np.random.normal(loc=0, scale=sigma, size=(n_r, n_r))
        BZ_dev = BZ + np.random.normal(loc=0, scale=sigma, size=(n_r, n_r))
    
        BSWX_dist, BSWY_dist, BSWZ_dist = get_reconstructed_field(planet=planet, n=n_r, xs=xs, ys=ys, BX=BX_dev, BY=BY_dev, BZ=BZ_dev)
        for i in range(n_r):
            for j in range(n_r):            
                r = np.array([xs[i], ys[j], 0], dtype=np.double)
                if planet.check_vector_in_sheath(r):
                    err_X[j][i] += (BSWX_dist[j][i] - IMF[0]) / n_avg
                    err_Y[j][i] += (BSWY_dist[j][i] - IMF[1]) / n_avg
                    err_Z[j][i] += (BSWZ_dist[j][i] - IMF[2]) / n_avg
                    err_mag[j][i] += (np.sqrt(BSWX_dist[j][i]**2 + BSWY_dist[j][i]**2 + BSWZ_dist[j][i]) - B0) / n_avg

    err_X /= B0
    err_Y /= B0
    err_Z /= B0
    err_mag /= B0
    

    return xs, ys, err_X, err_Y, err_Z, err_mag


#@njit
def relative_reconstruction_errors_pos(planet, n_r, n_avg, sigma, xmin, xmax, ymin, ymax):
    IMF = planet.IMF    
    B0  = np.linalg.norm(IMF)

    xs = np.linspace(xmin, xmax, n_r)
    ys = np.linspace(ymin, ymax, n_r)
    BX, BY, BZ = get_exact_field(planet, n_r, xs, ys)
    err_X = np.zeros((n_r, n_r), dtype=np.double)
    err_Y = np.zeros((n_r, n_r), dtype=np.double)
    err_Z = np.zeros((n_r, n_r), dtype=np.double)
    err_mag = np.zeros((n_r, n_r), dtype=np.double)    

    for m in range(n_avg):        
        xx, yy = np.meshgrid(xs, ys)
        xs_dev = xx + np.random.normal(loc=0, scale=sigma, size=(n_r, n_r))
        ys_dev = yy + np.random.normal(loc=0, scale=sigma, size=(n_r, n_r))
        zs_dev = np.random.normal(loc=0, scale=sigma, size=(n_r, n_r))
        
    
        BSWX_dist, BSWY_dist, BSWZ_dist = get_reconstructed_field_pos_errs(planet=planet, n=n_r, xs=xs_dev, ys=ys_dev, zs=zs_dev, BX=BX, BY=BY, BZ=BZ)
        for i in range(n_r):
            for j in range(n_r):            
                r = np.array([xs_dev[j][i], ys_dev[j][i], zs_dev[j][i]], dtype=np.double)
                if planet.check_vector_in_sheath(r):
                    err_X[j][i] += (BSWX_dist[j][i] - IMF[0]) / n_avg
                    err_Y[j][i] += (BSWY_dist[j][i] - IMF[1]) / n_avg
                    err_Z[j][i] += (BSWZ_dist[j][i] - IMF[2]) / n_avg
                    err_mag[j][i] += (np.sqrt(BSWX_dist[j][i]**2 + BSWY_dist[j][i]**2 + BSWZ_dist[j][i]) - B0) / n_avg

    err_X /= B0
    err_Y /= B0
    err_Z /= B0
    err_mag /= B0

    return xs, ys, err_X, err_Y, err_Z, err_mag



@njit # https://stackoverflow.com/questions/70613681/numba-compatible-numpy-meshgrid
def meshgrid(x, y):
    xx = np.empty(shape=(x.size, y.size), dtype=x.dtype)
    yy = np.empty(shape=(x.size, y.size), dtype=y.dtype)
    for j in range(y.size):
        for k in range(x.size):
            xx[j,k] = k  # change to x[k] if indexing xy
            yy[j,k] = j  # change to y[j] if indexing xy
    return yy, xx