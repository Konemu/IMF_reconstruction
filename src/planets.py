import numpy as np
import numba as nb
from numba import double, boolean, njit
from numba.experimental import jitclass

# needed for numba classes as of now
planet_spec = [
    ("R_planet", double),
    ("R_bowshock", double),
    ("R_magnetopause", double),
    ("r_focus", double[:]),
    ("IMF", double[:]),
    ("IMF_known", boolean),
]


# compiled class that aggregates geometric information of a planet and its IMF (if given)
# provides routines for calculating the MS field from the IMF and vice versa
@jitclass(planet_spec)
class Planet:    
    def __init__(self, R_planet, R_bowshock, R_magnetopause, IMF):
        self.R_planet       = R_planet
        self.R_bowshock     = R_bowshock
        self.R_magnetopause = R_magnetopause

        self.r_focus = np.array([R_magnetopause / 2, 0, 0])

        self.IMF           = IMF
        if IMF is None:
            self.IMF_known = False
        else:
            self.IMF_known = True


    # r: 3D XYZ coordinates
    # return: 3x3 matrix that maps solar wind field to bow shock field
    def trans_mat(self, r):
        [x, y ,z] = r
        x_f = self.r_focus[0]
        d = self.d(r)        

        C = self.R_magnetopause * (2 * self.R_bowshock - self.R_magnetopause) / (2 * (self.R_bowshock - self.R_magnetopause))

        T =   C / self.R_magnetopause * np.diag(np.array([1, 1, 1], dtype=np.double)) \
            + C / (d * (d + x - x_f)) * np.array([  [-0.5 * (d + x - x_f)   , -y                        , -z                        ], 
                                                    [-y/2                   , d - y**2 / (d + x - x_f)  , y * z / (d + x - x_f)     ], 
                                                    [z/2                    , y * z / (d + x - x_f)     , d - z**2 / (d + x - x_f)  ]], dtype=np.double)
       
        return T
        

    # r: 3D XYZ coordinates
    # return: distance of point r to focus point
    def d(self, r):        
        return np.linalg.norm(self.r_focus - np.asarray(r, dtype=np.double))
    

    # r: 3D XYZ coordinates
    # return MS field at r if IMF known
    # does not check if r in MS!
    def MS_from_IMF(self, r):
        if not self.IMF_known:
            print("Cannot infer magnetosheath field from IMF if IMF is not given.")      
            return None                      
        else:
            return self.trans_mat(r) @ self.IMF
        
    
    # r: 3D XYZ coordinates
    # B: 3D MS field
    # return IMF from MS-field
    # does not check if r in MS!
    def IMF_from_MS(self, r, B):
        return np.linalg.inv(self.trans_mat(r)) @ B
        
    
    # equation for the boundary parabola
    # y, z: coords
    # f: focus
    # R: radius of MP or BS
    def parab(self, y, z, f, R):
        return -(y**2 + z**2) / (4 * f) + R
    
    # r: 3D XYZ coordinates
    # does what it says
    def check_vector_in_sheath(self, r):
        [x, y, z] = r
        f_bs = self.R_bowshock - self.R_magnetopause / 2
        f_mp = self.R_magnetopause / 2
        if self.parab(y, z, f_mp, self.R_magnetopause) < x < self.parab(y, z, f_bs, self.R_bowshock):
            return True
        else:
            return False
        

    