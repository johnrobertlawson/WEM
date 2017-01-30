import numpy as N

import scipy.ndimage.interpolation as INT

class fractal_generator(object):
    def __init__(self,arr):
        self.arr = arr

    def IFS_go(self,r,s,theta,phi,e,f):
        self.r = r
        self.s = s
        self.theta = theta
        self.phi = phi
        self.e = e
        self.f = f
        self.xdim, self.ydim = arr.shape

        arr_mod = N.copy(arr)
        arr_mod = self.apply_r(arr_mod)
        arr_mod = self.apply_s(arr_mod)
        arr_mod = self.apply_theta(arr_mod)
        arr_mod = self.apply_phi(arr_mod)
        arr_mod = self.apply_e(arr_mod)
        arr_mod = self.apply_f(arr_mod)

        self.fractal = arr_mod
        return self.fractal

    def apply_r(self,arr_in):
        # Scaling in the horizontal
        zoom = (self.r,0)
        arr_out = INT.zoom(arr_in,zoom)
        return arr_out

    def apply_s(self,arr_in):
        # Scaling in the horizontal
        zoom = (0,self.s)
        arr_out = INT.zoom(arr_in,zoom)
        return arr_out

    def apply_theta(self,arr_in):
        arr_out = INT.rotate(arr_in,self.theta)
        return arr_out

    def apply_phi(self,arr_in):
        arr_out = INT.rotate(arr_in,self.phi)
        return arr_out

    def apply_e(self,arr_in):
        shift = (self.e,0)
        arr_out = INT.shift(arr_in,shift)
        return arr_out

    def apply_f(self,arr_in):
        shift = (0,self.f)
        arr_out = INT.shift(arr_in,shift)
        return arr_out
