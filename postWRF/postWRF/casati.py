import os
import pdb

import numpy as N
from scipy.ndimage.filters import uniform_filter

from WEM.utils.exceptions import QCError, FormatError

class Casati:
    def __init__(self,fcstdata,verifdata,thresholds=(0.5,1,4,16,64),
                    store_all_arrays=True,calibration=True):
        """Casati et al. 2004, Meteorol. Appl.

        All data in SI units (mm). 

        Optional:
        store_all_arrays (bool) -   if False, delete arrays to conserve memory.

        Attribute dictionaries (all accessed by a threshold then spatial key):
        self.father, self.mother    -   wavelet decompositions (in do_BED)
        self.MSE, self.MSE_total    -   Mean Square Error (by scale and total)
        self.SS                     -   Skill score

        e.g. 
        self.SS[4][3] will return the skill score for 4 mm/hr threshold 
            and the L=3 spatial scale (= 2**3 grid points)
        """
        self.thresholds = thresholds
        self.raw_fcstdata = fcstdata
        self.raw_verifdata = verifdata

        # Format checks
        self.enforce_2D()
        self.do_square_check()
        self.do_QC()

        # Pre-processing
        self.do_norain_pixels()
        self.do_dithering()
        self.do_normalize()
        self.do_recalibration()

        # Set length scales
        self.Ls = [2**L for L in range(1,self.L+1)]

        # Verification
        self.do_BED()
        self.do_MSE()
        self.do_SS()


    def enforce_2D(self,):
        """Make sure the grid is 2-D.
        """
        msg = 'Pass a 2D array.'
        for data in (self.raw_fcstdata,self.raw_verifdata):
            if len(data.shape) == 1:
                raise FormatError(msg)
            elif len(data.shape) > 2:
                for dim in N.arange(len(data.shape)-2):
                    if data.shape[0] !=1:
                        raise FormatError(msg)
            else:
                print("Passed 2D grid check.")
        return

    
    def do_square_check(self):
        """Assert that the grid is a power of 2.
        """
        for data in (self.raw_fcstdata,self.raw_verifdata):
            for dim in range(len(data.shape)):
                if not N.log2(data.shape[dim]).is_integer():
                    msg = "Data dimension not square/power of 2, but of shape {}.".format(
                                    data.shape)
                    raise FormatError(msg)
                elif data.shape[0] != data.shape[1]:
                    raise FormatError(msg)
                else:
                    print("Passed square check.")
                    self.L = int(N.log2(data.shape[dim]))
        return

    def do_QC(self,print_location=True,min_thresh=0.0,max_thresh=500):
        """Check for stupid values in data.
        
        Optional:
        print_location (bool) - if True, print the array location where
                                the nonsensical values exist
        min_thresh, max_thresh (float,int) - min and max values of allowable
                                                range, to raise Exception
        """
        for data in (self.raw_fcstdata,self.raw_verifdata):
            where_under = N.where(data < min_thresh)
            where_over = N.where(data > max_thresh)

            if print_location:
                pass_idx = (where_under,where_over)
            else:
                pass_idx = False

            for d in range(len(where_over)):
                if (where_over[d].size > 0) or (where_under[d].size > 0):
                    raise QCExrror("Unacceptable data found.",pass_idx=pass_idx)
        return

    def do_dithering(self,dithrange=(1/64)):
        """Apply dithering to non-zero precipitation values.
        """
        for data in (self.raw_fcstdata,self.raw_verifdata):
            where_zero = N.where(data > 0.0)
            #shp = [where_zero[n].shape for n in range(len(where_zero))]
            sze = sum([where_zero[n].size for n in range(len(where_zero))])
            noise = N.random.uniform(low=-1*dithrange,high=dithrange,size=sze)
            data[where_zero] += noise
        print("Dithering complete.")
        return

    def do_norain_pixels(self,val=-6):
        """Assign value to no-rain pixels.
        TODO: Should be range around 0, not exactly, due to round-off?
        """
        for data in (self.raw_fcstdata,self.raw_verifdata):
            norain_idx = N.where(data == 0.0)
            data[norain_idx] = val
        print("No-rain pixels now set to {}.".format(val))


    def do_normalize(self,):
        """Normalise data by base 2.
        """
        self.norm_fcstdata = N.log2(self.raw_fcstdata)
        self.norm_verifdata = N.log2(self.raw_verifdata)
        print("Normalization complete.")
        return

    def do_recalibration(self):
        """
        Yr = (1/Fx) * Fy

        Yr = recalibrated forecast
        Fx = Empirical cum. dist. of verif
        Fy = Empirical cum. dist. of fcst
        """
        # Convert calc_ecdf to a vectorized function
        vf = N.vectorize(self.calc_ecdf)

        Fx = vf(self.norm_verifdata,self.norm_fcstdata)
        Fy = vf(self.norm_fcstdata,self.norm_verifdata)
        # for fcstval in self.norm_fcstdata:
            # Fx = calc_ecdf(self.norm_verifdata,fcstval)
        self.recalib_fcstdata = (1/Fx) * Fy
        return

    def calc_ecdf(self,arr,val):
        """ Do empirical cumulative distribution of an array.
        
        arr -   N.ndarray
        val -   threshold
        """
        x = N.sort(arr)
        ecd = N.searchsorted(x,val,side='right') / x.size
        return ecd
    
    def do_BED(self,):
        """Binary error decomposition.

        Yr = Recalibrated forecast
        X = Normalized analysis/verif
        Z3 = Binary error (Eq. 3 in paper)
        IYr = Binary image for recal. fcst (Eq.2 in paper)
        IX = Binary image for norm. verif (Eq.2 in paper)
        Zf, Zm = Father and mother wavelets from Eq. 4 / Appendix in paper
        Z = binary error image, post-Haar
        """
        X = self.self.norm_verifdata
        Yr = self.recalib_fcstdata
        vBED = N.vectorize(self.BED)
        self.IYr = {}
        self.IX = {}
        self.Zf = {}
        self.Zm = {}

        for th in self.thresholds:
            self.IYr[th] = vBED(Yr,th)
            self.IX[th] = vBED(X,th)
            self.Z3[th] = self.IYr[th] - self.IX[th]
            self.Zl[th],self.Zf[th], self.Zm[th], self.Ztotal[th] = self.Haar(self.Z[th])
        return

    def BED(self,pixel,thresh):
        return True if pixel > thresh else False

    def Haar(self,Z):
        """ Z14 is Eq 14 from paper.
        """
        father = {}
        mother = {}
        father[2] = Z
        Zl = {}

        print("Beginning Haar decomposition.")
        for Lidx,L in enumerate(self.Ls):
            L0 = L/2
            father[L] = uniform_filter(Z,size=L,mode='constant',cval=0)
            mother[L] = father[L0]-father[L]
            Zl[L] = father[L] + mother[L] # is this the same as old Z?

        assert father[-1] == 0.0
        Ztotal = father[self.Ls[-1]] + N.sum(N.dstack([mother[l] for l in self.Ls]),axis=2)
        print("Finished Haar decomposition.")
        return Zl, father, mother, Ztotal

    def do_MSE(self,):
        # Eq. 8 etc are ambiguous.
        MSE = {}
        for th in thresholds:
            self.MSE[th] = {}
            for l in self.Ls:
                self.MSE[th][l] = N.mean(self.Zl[th][l]**2)
        self.MSE_total = N.sum([self.MSE[th][l] for l in self.Ls])
        # MSE(l=1) should equal MSE_total
        return

    def do_SS(self,):
        self.SS = {}
        for th in self.thresholds:
            self.SS[th] = {}
            for l in self.Ls:
                # self.SS[th][l] = ((self.MSE[th] - MSErandom[th])/
                               # (MSEbest[th] - MSErandom[th]))
                # Epsilon is base rate - fraction of rain vs no rain pixels
                ep = N.sum(self.Zl[th][l])/self.Zl[th][l].size
                self.SS[th][l] = 1 - (self.MSE[th][l]/(2*ep*(1-ep)*(1/l)))

