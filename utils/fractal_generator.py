import pdb

import numpy as N

import scipy.ndimage.interpolation as INT
from scipy.interpolate import RectBivariateSpline as RBS

class FracGen(object):
    def __init__(self,arr=False):
        self.arr = arr
        pass

    def IFS_go(self,r=0,s=0,theta=0,phi=0,e=0,f=0,zoom=2):
        # new_arr = INT.zoom(self.arr,3,output=int)
        self.xdim, self.ydim = self.arr.shape
        self.r = r
        self.s = s
        self.theta = theta
        # Need to make phi different from theta
        self.phi = phi
        self.e = e*self.xdim
        self.f = f*self.ydim

        self.xx_in = N.arange(self.xdim)
        self.yy_in = N.arange(self.ydim)

        arr_mod = N.copy(self.arr)
        if r != 0:
            arr_mod = self.apply_r(arr_mod)
        if s != 0:
            arr_mod = self.apply_s(arr_mod)
        if theta != 0:
            arr_mod = self.apply_theta(arr_mod)
        # if phi != 0:
            # arr_mod = self.apply_phi(arr_mod)
        if e != 0:
            arr_mod = self.apply_e(arr_mod)
        if f != 0:
            arr_mod = self.apply_f(arr_mod)

        return arr_mod

    def apply_r(self,arr_in):
        # Scaling in the horizontal
        xx_out = self.xx_in * self.r
        rbs = RBS(self.yy_in,xx_out,arr_in)
        nn = rbs(self.yy_in,self.xx_in)
        return nn.round().astype(int)

    def apply_s(self,arr_in):
        # Scaling in the vertical
        yy_out = self.yy_in * self.r
        rbs = RBS(yy_out,self.xx_in,arr_in)
        nn = rbs(self.yy_in,self.xx_in)
        return nn.round().astype(int)

    def apply_theta(self,arr_in):
        arr_out = INT.rotate(arr_in,self.theta,reshape=False)
        return arr_out

    def apply_phi(self,arr_in):
        arr_out = INT.rotate(arr_in,self.phi,reshape=False)
        return arr_out

    def apply_e(self,arr_in):
        shift = (self.e,0)
        arr_out = INT.shift(arr_in,shift)
        return arr_out

    def apply_f(self,arr_in):
        shift = (0,self.f)
        arr_out = INT.shift(arr_in,shift)
        return arr_out

