import os

import numpy as N
import scipy.ndimage.filters

from WEM.utils.exceptions import FormatError

class self.FSS():
    def __init__(self,data_fcst,data_obs,itime=False,ftime=False,
                lv=False,thresholds=(0.5,1,2,4,8),ns=None,
                ns_step=4):
        """ Fcst and ob data needs to be on the same grid.
        """
        self.data_fcst = data_fcst
        self.data_obs = data_obs
        self.thresholds = thresholds

        self.enforce_2D()
        self.do_grid_check()

        self.xdim = data

        # Neighbourhoods
        if ns is None:
            ns = N.arange(1,maxlen,ns_step)
        self.ns = ns

        # Computations
        self.compute_MSE()
        self.compute_FSS()
        return
        
    def enforce_2D(self,):
        """Both data grids need to be 2D.
        """
        for data in (self.data_obs,self.data_fcst):
            shp = data.shape
            if len(shp) == 2:
                pass 
            elif len(shp) == 3:
                if shp[0] == 0:
                    data = data[0,:,:]
            elif len(shp) == 4:
                if (shp[0] == 0) and (shp[1] == 0):
                    data = data[0,0,:,:]
            else:
                raise FormatError("Data needs to be 2D.")
        return

    def do_grid_check(self,):
        """ Make sure grids are identical size.
        """
        self.ydim, self.xdim = self.data_fcst.shape 
        if self.data_obs.shape != (self.ydim,self.xdim):
            raise FormatError("Obs and forecast data not same size.")
        return
    
    def compute_FSS(self):
        maxlen = max(self.ydim,self.xdim)

        for th in threshs:
            self.MSE[th] = {}
            self.FSS[th] = {}
            # Convert to binary using thresholds
            fc = N.copy(self.data_fcst)
            ob = N.copy(self.data_obs)
            fc[fc < th] = False
            fc[fc >= th] = True
            ob[ob < th] = False
            ob[ob >= th] = True
            for n in ns:
                self.MSE[th][n] = {}
                self.FSS[th][n] = {}
                # print("self.FSS for threshold {0} mm and n={1}.".format(th,n))
                # self.FSS computation w/ fractions

                pad = int((n-1)/2)
                On = scipy.ndimage.filters.uniform_filter(ob,size=n,
                                mode='constant',cval=0)
                Mn = scipy.ndimage.filters.uniform_filter(fc,size=n,
                                mode='constant',cval=0)

                # Delete meaningless smoothed data
                cutrangex = list(range(0,pad)) + list(range(self.xdim-pad,self.xdim))
                cutrangey = list(range(0,pad)) + list(range(self.ydim-pad,self.xdim))
                On = N.delete(On,cutrangey,axis=0)
                Mn = N.delete(Mn,cutrangey,axis=0)
                On = N.delete(On,cutrangex,axis=1)
                Mn = N.delete(Mn,cutrangex,axis=1)
                cutlenx = On.shape[1]
                cutleny = On.shape[0]

                # self.MSE
                sqdif = (On-Mn)**2
                self.MSE[th][n]['score'] = (1/(cutlenx*cutleny))*N.sum(sqdif)

                # Reference self.MSE
                self.MSE[th][n]['ref'] = (1/(cutlenx*cutleny))*(N.sum(On**2)+N.sum(Mn**2))
            
                # self.FSS
                self.FSS[th][n] = 1 - (self.MSE[th][n]['score'] / self.MSE[th][n]['ref'])
        return

