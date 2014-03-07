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
import time
import json
import cPickle as pickle
import copy

from wrfout import WRFOut
from axes import Axes
from figure import Figure
from birdseye import BirdsEye
#import scales
from defaults import Defaults
from lookuptable import LookUpTable
import WEM.utils.utils as utils

class WRFEnviron:
    def __init__(self,config):
        self.C = config

        # Set defaults if they don't appear in user's settings
        self.D = Defaults()

        self.font_prop = getattr(self.C,'font_prop',self.D.font_prop)
        self.usetex = getattr(self.C,'usetex',self.D.usetex)
        self.dpi = getattr(self.C,'DPI',self.D.dpi)
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

    def plot_2D(self,dic):
        """
        Currently main plotting script: Feb 25 2014

        Path to wrfout file is in config file.
        Path to plot output is also in config

        This script is top-most and decides if the variables is
        built into WRF default output or needs computing. It unstaggers
        and slices data from the wrfout file appropriately.


        Inputs:
        dic     :   nested dictionary with:

            KEY
            ===
            va      :   variable to plot

            nested KEY/VALUE PAIRS
            ======================
            (MANDATORY FOR SOME VARIABLES)
            lv      :   level to plot
            pt      :   plot times
            (OPTIONAL)
            tla     :   top limit of latitude
            bla     :   bottom limit of latitude
            llo     :   left limit of longitude
            rlo     :   right limit of longitude
            ---> if these are missing, default to 'all points'


        """

        #self.en = self.get_sequence(wrfout)
        #self.pt = self.get_sequence(times) # List of plot times
        Dic = copy.deepcopy(dic)
        wrfpath = self.wrfout_files_in(self.C.wrfout_root)[0]
        self.W = WRFOut(wrfpath) # Only load netCDF file once!
        for va in Dic:
            if not 'lv' in Dic[va]: # For things like CAPE, shear.
                Dic[va]['lv'] = 'all'

            vc = utils.level_type(Dic[va]['lv']) # vertical coordinate

            # Check for pressure levels
            if vc == 'isobaric':
                nc_path = self.W.path
                p_interp_fpath = self.W.interp_to_p(self.C,nc_path,va,lv)
                # Edit p_interp namelist
                #Execute p_interp here and reassign self.W to new file
                self.W = WRFOut(p_interp_fpath)
            else: #
                # print("Non-pressure levels not supported yet.")
                # raise Exception
                pass

            F = BirdsEye(self.C,self.W)
            lv = Dic[va]['lv']
            for t in Dic[va]['pt']:
                # print('Passing point. Dic: \n', Dic)
                # pdb.set_trace()
                disp_t = utils.string_from_time('title',t)
                print("Plotting {0} at lv {1} for time {2}.".format(
                                    va,lv,disp_t))
                Dic[va]['pt'] = t
                Dic[va]['vc'] = vc
                F.plot2D(va, Dic[va])

    """
    def plot_variable2D(self,varlist,timelist):
        self.va = self.get_sequence(varlist) # List of variables
        self.pt = self.get_sequence(timelist) # List of plot times

        # Where is logic to

        for x in itertools.product(self.va,self.pt):
            va, pt = x
            W = WRFOut(self.C.wrfout_root)
            F = BirdsEye(self.C,W)
            F.plot2D(va,pt,lv=2000)


    def plot_variable2D(self,va,pt,en,lv,p2p,na=0,da=0):
        ###Plot a longitude--latitude cross-section (bird's-eye-view).
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

        ###
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
    """

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

    def save_data(self,data,folder,fname,format='pickle'):
        # Strip file extension given
        fname_base = os.path.splitext(fname)[0]
        # Check for folder, create if necessary
        utils.trycreate(folder)
        # Create absolute path
        fpath = os.path.join(folder,fname_base)

        if format=='pickle':
            with open(fpath+'.pickle','wb') as f:
                pickle.dump(data,f)
        elif format=='numpy':
            N.save(fpath,data)
        elif format=='json':
            j = json.dumps(data)
            with open(fpath+'.json','w') as f:
                print >> f,j
        else:
            print("Give suitable saving format.")
            raise Exception

        print("Saved file {0} to {1}.".format(fname,folder))

    def load_data(self,folder,fname,format='pickle'):
        fname2 = os.path.splitext(fname)[0]
        fpath = os.path.join(folder,fname2)
        if format=='pickle':
            with open(fpath+'.pickle','rb') as f:
                data = pickle.load(f)
        elif format=='numpy':
            data = N.load(fpath+'.npy')
        elif format=='json':
            print("JSON stuff not coded yet.")
            raise Exception
        else:
            print("Give suitable loading format.")
            raise Exception

        print("Loaded file {0} from {1}.".format(fname,folder))
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
        nperm = itertools.combinations(files,2).__sizeof__()
        for n, perm in enumerate(itertools.combinations(files,2)):
            if n>9999:
                print("Skipping #{0} for debugging.".format(n))
            else:
                print("No. {0} from {1} permutations".format(n,nperm))
                perm_start = time.time()
                DATA[str(n)] = {}
                f1, f2 = perm
                W1 = WRFOut(f1)
                W2 = WRFOut(f2)
                #pdb.set_trace()
                # Make sure times are the same in both files
                if not N.all(N.array(W1.wrf_times) == N.array(W2.wrf_times)):
                    print("Times are not identical between input files.")
                    raise Exception
                else:
                    print("Passed check for identical timestamps between "
                          "NetCDF files")

                # Find indices of each time
                t_idx = []
                for t in ts:
                    t_idx.append(W1.get_time_idx(t))

                print("Calculating values now...")
                DATA[str(n)]['times'] = ts
                DATA[str(n)]['values'] = []
                for t in t_idx:
                    DATA[str(n)]['values'].append(PLOTS[ptype](W1.nc,W2.nc,t,
                                                    energy,lower,upper))
                DATA[str(n)]['file1'] = f1
                DATA[str(n)]['file2'] = f2

                print "Calculation #{0} took {1:2.2f} seconds.".format(n,time.time()-perm_start)

        if d_return and not d_save:
            return DATA
        elif d_save and not d_return:
            #self.save_data(DATA,d_save,d_fname)
            self.save_data(DATA,d_save,d_fname)
            #self.json_data(DATA,d_save,d_fname)
            return
        elif d_return and d_save:
            #self.save_data(DATA,d_save,d_fname)
            self.save_data(DATA,d_save,d_fname)
            #self.json_data(DATA,d_save,d_fname)
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

    def DE_z(self,nc0,nc1,t,energy,lower,upper):
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
        t       :   times index to difference
        energy  :   kinetic or total
        lower   :   lowest level, hPa
        upper   :   highest level, hPa

        Outputs:

        data    :   2D array.
        """

        # Speed up script by only referencing data, not
        # loading it to a variable yet

        # WIND
        U0 = nc0.variables['U'][:]
        V0 = nc0.variables['V'][:]
        U1 = nc1.variables['U'][:]
        V1 = nc1.variables['V'][:]

        # PERT and BASE PRESSURE
        P0 = nc0.variables['P'][:]
        PB0 = nc0.variables['PB'][:]
        P1 = nc1.variables['P'][:]
        PB1 = nc1.variables['PB'][:]

        if energy=='total':
            T0 = nc0.variables['T'][:]
            T1 = nc1.variables['T'][:]
            R = 287.0 # Universal gas constant (J / deg K * kg)
            Cp = 1004.0 # Specific heat of dry air at constant pressure (J / deg K * kg)
            kappa = R/Cp

        xlen = U0.shape[2] # 1 less than in V
        ylen = V0.shape[3] # 1 less than in U
        zlen = U0.shape[1] # identical in U & V

        # Generator for lat/lon points
        def latlon(nlats,nlons):
            for i in range(nlats): # y-axis
                for j in range(nlons): # x-axis
                    yield i,j

        DKE = []
        DKE2D = N.zeros((xlen,ylen))
        print_time = ''.join((nc0.variables['Times'][t]))
        print("Calculating 2D grid for time {0}...".format(print_time))
        gridpts = latlon(xlen,ylen)
        for gridpt in gridpts:
            i,j = gridpt
            # print("Calculating for gridpoints {0} & {1}.".format(i,j))
            # Find closest level to 'lower', 'upper'
            P_col = P0[t,:,j,i] + PB0[t,:,j,i]
            if lower:
                low_idx = utils.closest(P_col,lower*100.0)
            else:
                low_idx = None
            if upper:
                upp_idx = utils.closest(P_col,upper*100.0)
            else:
                upp_idx = None

            zidx = slice(low_idx,upp_idx+1)
            # This needs to be a 2D array?

            if energy=='kinetic':
                DKE2D[j,i] = N.sum(0.5*((U0[t,zidx,j,i]-U1[t,zidx,j,i])**2 +
                                    (V0[t,zidx,j,i]-V1[t,zidx,j,i])**2))
            elif energy=='total':
                DKE2D[j,i] = N.sum(0.5*((U0[t,zidx,j,i]-U1[t,zidx,j,i])**2 +
                                    (V0[t,zidx,j,i]-V1[t,zidx,j,i])**2 +
                                    kappa*(T0[t,zidx,j,i]-T1[t,zidx,j,i])**2))

        DKE.append(DKE2D)

        return DKE

    def plot_diff_energy(self,ptype,energy,time,folder,fname,p2p,V):
        """
        folder  :   directory holding computed data
        fname   :   naming scheme of required files
        V       :   constant values to contour at
        """
        sw = 0

        DATA = self.load_data(folder,fname,format='pickle')
        times = self.get_sequence(time)

        for n,t in enumerate(times):
            for pn,perm in enumerate(DATA):
                if sw==0:
                    # Get times and info about nc files
                    W1 = WRFOut(DATA[perm]['file1'])
                    permtimes = DATA[perm]['times']
                    sw = 1

                # Find array for required time
                x = N.where(N.array(permtimes)==t)[0][0]
                data = DATA[perm]['values'][x][0]
                if not pn:
                    stack = data
                else:
                    stack = N.dstack((data,stack))
            stack_average = N.average(stack,axis=2)

            #birdseye plot with basemap of DKE/DTE
            F = BirdsEye(self.C,W1,p2p)    # 2D figure class
            #F.plot2D(va,t,en,lv,da,na)  # Plot/save figure
            fname_t = ''.join((fname,'_p{0:02d}'.format(n)))
            F.plot_data(stack_average,'contour',fname_t,t,V)
            print("Plotting time {0} from {1}.".format(n,len(times)))
            del data, stack

