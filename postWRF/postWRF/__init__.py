"""Take config settings and run plotting scripts.

Classes:
C = Config = configuration settings set by user, passed from user script
D = Defaults = used when user does not specify a non-essential item
W = Wrfout = wrfout file
F = Figure = a superclass of figures
    mp = Birdseye = a lat--lon slice through data with basemap
    xs = CrossSection = distance--height slice through data with terrain 

"""

import os
import matplotlib as M
M.use('Agg')
import matplotlib.pyplot as plt
import collections
import fnmatch
import calendar
import pdb
import itertools
import numpy as N

from wrfout import WRFOut
from axes import Axes
from figure import Figure
from birdseye import BirdsEye
#import scales
from defaults import Defaults
import utils

class WRFEnviron:
    def __init__(self,config):
        self.C = config

        # Set defaults if they don't appear in user's settings
        self.D = Defaults()
        
        self.font_prop = getattr(self.C,'font_prop',self.D.font_prop)
        self.usetex = getattr(self.C,'usetex',self.D.usetex)
        self.dpi = getattr(self.C,'dpi',self.D.dpi)
        self.plot_titles = getattr(self.C,'plot_titles',self.D.plot_titles) 

        # Set some general settings
        M.rc('text',usetex=self.usetex)
        M.rc('font',**self.font_prop)
        M.rcParams['savefig.dpi'] = self.dpi

    def wrfout_files_in(self,folder,dom='notset',init_time='notset'):
        w = 'wrfout' # Assume the prefix
        if init_time=='notset':
            suffix = '*0'
            # We assume the user has wrfout files in different folders for different times
        else:
            try:
                it = self.string_from_time('wrfout',init_time)
            except:
                print("Not a valid wrfout initialisation time; try again.")
                raise Error
            suffix = '*' + it

        if dom=='notset':
            # Presume all domains are desired.
            prefix = w + '_d'
        elif (dom == 0) | (dom > 8):
            print("Domain is out of range. Choose number between 1 and 8 inclusive.")
            raise IndexError
        else:
            dom = 'd{0:02d}'.format(dom)
            prefix = w + '_' + dom

        wrfouts = []
        for root,dirs,files in os.walk(folder):
            for fname in fnmatch.filter(files,prefix+suffix):
                wrfouts.append(os.path.join(root, fname))
        return wrfouts

    def dir_from_naming(self,*args):
        l = [str(a) for a in args]
        path = os.path.join(self.C.output_root,*l)
        return path

    def string_from_time(self,usage,t,dom=0,strlen=0,conven=0):
        str = utils.string_from_time(usage=usage,t=t,dom=dom,strlen=strlen,conven=conven)
        return str

    def plot_variable2D(self,va,pt,en,lv,p2p,na=0,da=0):
        """Plot a longitude--latitude cross-section (bird's-eye-view).
        Use Basemap to create geographical data
        
        ========
        REQUIRED
        ========

        va = variable(s)
        pt = plot time(s)
        nc = ensemble member(s)
        lv = level(s) 
        p2p = path to plots

        ========
        OPTIONAL
        ========

        da = smaller domain area(s), needs dictionary || DEFAULT = 0
        na = naming scheme for plot files || DEFAULT = get what you're given

        """
        va = self.get_sequence(va)
        pt = self.get_sequence(pt,SoS=1)
        en = self.get_sequence(en)
        lv = self.get_sequence(lv)
        da = self.get_sequence(da)

        #perms = self.make_iterator(va,pt,en,lv,da)
        #perm2 = self.make_iterator2(va,pt,en,lv,da)
        #pdb.set_trace()

        # Find some way of looping over wrfout files first, avoiding need
        # to create new W instances
        # print("Beginning plotting of {0} figures.".format(len(list(perms))))
        pdb.set_trace() 

        #for x in perms:
        for x in itertools.product(va,pt,en,lv,da):
            va,pt,en,lv,da = x
            pdb.set_trace()
            W = WRFOut(en)    # wrfout file class using path
            F = BirdsEye(self.C,W,p2p)    # 2D figure class
            F.plot2D(va,pt,en,lv,da,na)  # Plot/save figure
            pt_s = utils.string_from_time('title',pt)
            print("Plotting from file {0}: \n variable = {1}" 
                  " time = {2}, level = {3}, area = {4}.".format(en,va,pt_s,lv,da))
    
    def make_iterator2(self,*args):
        for arg in args:
            for a in arg:
                yield a
 
    def make_iterator(self,va,pt,en,lv,da):   
        for v in va:
            for p in pt:
                for e in en:
                    for l in lv:
                        for d in da:
                            yield v,p,e,l,d
    
    def get_sequence(self,x,SoS=0):
        """ Returns a sequence (tuple or list) for iteration.

        SoS = 1 enables the check for a sequence of sequences (list of dates)
        """


        if SoS:
            y = x[0]
        else:
            y = x

        if isinstance(y, collections.Sequence) and not isinstance(y, basestring):
            return x
        else:
            return [x]
        
   
    def plot_cross_section(self,var,latA,lonA,latB,lonB):
        xs = CrossSection()
        xs.plot(var,latA,lonA,latB,lonB)

    def save_data(self,data,folder,fname):
        # Ensure file suffix will be .npy
        fname2 = os.path.splitext(fname)[0]
        # Check for folder and create if necessary
        utils.trycreate(folder)
        # Save in binary
        fpath = os.path.join(folder,fname2)
        N.save(fpath,data)
        
        print("Save file {0}.npy to {1}.".format(fname,folder))

    def load_data(self,folder,fname):
        # Ensure file suffix will be .npy
        fname2 = os.path.splitext(fname)[0]
        fpath = os.path.join(folder,fname2)
        data = N.load(fpath)       

        print("Loaded file {0}.npy from {1}.".format(fname,folder))
        return data

    def compute_diff_energy(
            self,ptype,energy,files,times,upper=None,lower=None,
            d_save=1,d_return=1,d_fname='diff_energy_data'):
        """
        This method computes difference kinetic energy (DKE)
        or different total energy (DTE, including temp)
        between WRFout files for a given depth of the 
        atmosphere, at given time intervals
        
        Inputs:
        
        ptype   :   'sum_z' or 'sum_xyz'
        energy  :   'kinetic' or 'total'
        upper   :   upper limit of vertical integration
        lower   :   lower limit of vertical integration
        files   :   abs paths to all wrfout files
        times   :   times for computations - tuple format
        d_save  :   save dictionary to folder (path to folder)
        d_return:   return dictionary (True or False)
        d_fname :   custom filename
        
        Outputs:
        
        data    :   time series or list of 2D arrays
        
        ptype 'sum_z' integrates vertically between lower and
        upper hPa and creates a time series.
        
        ptype 'sum_xyz' integrates over the 3D space (again between
        the upper and lower bounds) and creates 2D arrays.
        """
        if d_save and not isinstance(d_save,basestring):
            d_save = os.environ['HOME']
        
        # First, save or output? Can't be neither!
        if not d_save and not d_return:
            print("Pick save or output, otherwise it's a waste of computer"
                    "power")
            raise Exception
        
        # Look up the method to use depending on type of plot
        PLOTS = {'sum_z':self.DE_z, 'sum_xyz':self.DE_xyz}
        
        # Creates sequence of times
        ts = self.get_sequence(times)

        # Dictionary of data
        DATA = {}
        
        # Get all permutations of files
        for perm in itertools.combinations(files,2):
            print("Next permutation...")
            DATA[perm] = {}
            f1, f2 = perm
            W1 = WRFOut(f1)
            W2 = WRFOut(f2)
            #pdb.set_trace() 
            # Make sure times are the same in both files
            if not N.all(N.array(W1.wrf_times) == N.array(W2.wrf_times)):
                print("Times are not identical between input files.")
                raise Exception
            else:
                print("Passed check for identical timestamps between"
                      "NetCDF files")
        
            # Find indices of each time
            t_idx = []
            for t in ts:
                t_idx.append(W1.get_time_idx(t))
        
            print("Calculating values now...")
            DATA[perm]['times'] = ts
            DATA[perm]['values'] = PLOTS[ptype](W1.nc,W2.nc,t_idx,
                                                energy,lower,upper)
        
        if d_return and not d_save:
            return DATA
        elif d_save and not d_return:
            self.save_data(DATA,d_save,d_fname)
        elif d_return and d_save:
            self.save_data(DATA,d_save,d_fname)
            return DATA

    def DE_xyz(self,nc0,nc1,t_idx,energy,*args):
        """
        Computation for difference kinetic energy (DKE).
        Sums DKE over the 3D space, returns a time series.
        
        Destaggering is not enabled as it introduces
        computational cost that is of miniscule value considering
        the magnitudes of output values.

        Inputs:
        
        nc0     :   netCDF file
        nc1     :   netCDF file
        t_idx   :   times indices to difference
        energy  :   kinetic or total
        *args   :   to catch lower/upper boundary which isn't relevant here
        
        Outputs:
        
        data    :   time series.
        """
        # Wind data
        U0 = nc0.variables['U']
        V0 = nc0.variables['V']
        U1 = nc1.variables['U']
        V1 = nc1.variables['V']

        if energy=='total':
            T0 = nc0.variables['T']
            T1 = nc1.variables['T']
            R = 287.0 # Universal gas constant (J / deg K * kg)
            Cp = 1004.0 # Specific heat of dry air at constant pressure (J / deg K * kg)
            kappa = (R/Cp)

        xlen = U0.shape[2]

        DKE = []
        for n,t in enumerate(t_idx):
            print("Finding DKE at time {0} of {1}.".format(n,len(t)))
            DKE_hr = 0   # Sum up all DKE for the 3D space
            for i in range(xlen):
                if energy=='kinetic':
                    DKE_hr += N.sum(0.5*((U0[t,:,:,i]-U1[t,:,:,i])**2 +
                                (V0[t,:,:-1,i]-V1[t,:,:-1,i])**2))
                elif energy=='total':
                    DKE_hr += N.sum(0.5*((U0[t,:,:,i]-U1[t,:,:,i])**2 +
                                (V0[t,:,:-1,i]-V1[t,:,:-1,i])**2 +
                                kappa*(T0[t,:,:,i]-T1[t,:,:,i])**2))
            print("DTE at this time: {0}".format(DKE_hr))
            DKE.append(DKE_hr)
        return DKE 

    def DE_z(self,nc0,nc1,t_idx,energy,lower,upper):
        """
        Computation for difference kinetic energy (DKE).
        Sums DKE over all levels between lower and upper,
        for each grid point, and returns a 2D array.
        
        Destaggering is not enabled as it introduces
        computational cost that is of miniscule value considering
        the magnitudes of output values.
        
        Method finds levels nearest lower/upper hPa and sums between
        them inclusively.
                
        Inputs:
        
        nc0     :   netCDF file
        nc1     :   netCDF file
        t_idx   :   times indices to difference
        energy  :   kinetic or total
        lower   :   lowest level, hPa
        upper   :   highest level, hPa
        
        Outputs:
        
        data    :   2D array.
        """
        
        # Speed up script by only referencing data, not 
        # loading it to a variable yet
        
        # WIND
        U0 = nc0.variables['U']
        V0 = nc0.variables['V']
        U1 = nc1.variables['U']
        V1 = nc1.variables['V']

        # PERT and BASE PRESSURE
        P0 = nc0.variables['P']
        PB0 = nc0.variables['PB']
        P1 = nc1.variables['P']
        PB1 = nc1.variables['PB']

        if energy=='total':
            T0 = nc0.variables['T']
            T1 = nc1.variables['T']
            R = 287.0 # Universal gas constant (J / deg K * kg)
            Cp = 1004.0 # Specific heat of dry air at constant pressure (J / deg K * kg)
            kappa = (R/Cp)
            
        xlen = U0.shape[2] # 1 less than in V
        ylen = V0.shape[3] # 1 less than in U
        zlen = U0.shape[1] # identical in U & V
        
        # Generator for lat/lon points
        def latlon(nlats,nlons):
            for i in range(nlats):
                for j in range(nlons):
                    yield i,j
                
        gridpts = latlon(xlen,ylen)
        
        DKE = []
        for n,t in enumerate(t_idx): 
            DKE2D = N.zeros((xlen,ylen))
            print("Calculating 2D grid for time index {0}...".format(t))
            for gridpt in gridpts:
                i,j = gridpt
                # print("Calculating for gridpoints {0} & {1}.".format(i,j))
                # Find closest level to 'lower', 'upper'
                P_col = P0[t,:,i,j] + PB0[t,:,i,j]
                if lower:
                    low_idx = utils.closest(P_col,lower*100.0)
                else:
                    low_idx = None
                if upper:
                    upp_idx = utils.closest(P_col,upper*100.0)
                else:
                    upp_idx = None
                
                zidx = slice(low_idx,upp_idx+1) 
                if energy=='kinetic':
                    DKE2D[i,j] = N.sum(0.5*((U0[t,zidx,i,j]-U1[t,zidx,i,j])**2 +
                                        (V0[t,zidx,i-1,j]-V1[t,zidx,i-1,j])**2))
                elif energy=='total':
                    DKE2D[i,j] = N.sum(0.5*((U0[t,zidx,i,j]-U1[t,zidx,i,j])**2 +
                                        (V0[t,zidx,i-1,j]-V1[t,zidx,i-1,j])**2 +
                                        kappa*(T0[t,zidx,i,j]-T1[t,zidx,i,j])**2))
                                
            DKE.append(DKE2D)
        
        return DKE
        
    def plot_diff_energy(self,ptype,times,datafolder,p2p):
        DATA = N.load_data(datafolder)
        # dstack for each permutation and average for each grid point?
        # Plot  

    def generate_times(self,idate,fdate,interval):
        """
        Interval in seconds
        """
        i = calendar.timegm(idate)
        f = calendar.timegm(fdate)
        times = range(i,f,interval)
        return times
