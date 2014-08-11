"""Take config settings and run plotting scripts.

Classes:
C = Config = configuration settings set by user, passed from user script
D = Defaults = used when user does not specify a non-essential item
W = Wrfout = wrfout file
F = Figure = a superclass of figures
    mp = Birdseye = a lat--lon slice through data with basemap
    xs = CrossSection = distance--height slice through data with terrain

This script is API and should not be doing any hard work of
importing matplotlib etc!

Useful/utility scripts have been moved to WEM.utils.
"""

import calendar
import collections
import copy
import cPickle as pickle
import fnmatch
import glob
import itertools
import json
import numpy as N
import os
import pdb
import time

from wrfout import WRFOut
from axes import Axes
from figure import Figure
from birdseye import BirdsEye
from skewt import SkewT
from skewt import Profile
#import scales
from defaults import Defaults
from lookuptable import LookUpTable
import WEM.utils as utils
from xsection import CrossSection

# TODO: Make this awesome

class WRFEnviron(object):
    def __init__(self,config):
        # User's settings
        self.C = config

        # Set defaults if they don't appear in user's settings
        self.D = Defaults()

        # This stuff should be elsewhere.

        #self.font_prop = getattr(self.C,'font_prop',self.D.font_prop)
        #self.usetex = getattr(self.C,'usetex',self.D.usetex)
        #self.plot_titles = getattr(self.C,'plot_titles',self.D.plot_titles)
        #M.rc('text',usetex=self.usetex)
        #M.rc('font',**self.font_prop)
        #M.rcParams['savefig.dpi'] = self.dpi

    def plot2D(self,vrbl,times,levels,wrf_sd=0,wrf_nc=0,out_sd=0,f_prefix=0,f_suffix=0,
                bounding=0,dom=0):
        """
        Path to wrfout file is in config file.
        Path to plot output is also in config

        This script is top-most and decides if the variables is
        built into WRF default output or needs computing. It unstaggers
        and slices data from the wrfout file appropriately.


        Inputs:
        vrbl        :   string of variable name
        times       :   one or more date/times.
                        Can be tuple format (YYYY,MM,DD,HH,MM,SS - calendar.timegm)
                        Can be integer of datenum. (time.gmtime)
                        Can be a tuple or list of either.
        levels      :   one or more levels.
                        Lowest model level is integer 2000.
                        Pressure level is integer in hPa, e.g. 850
                        Isentropic surface is a string + K, e.g. '320K'
                        Geometric height is a string + m, e.g. '4000m'
        wrf_sd      :   string - subdirectory of wrfout file
        wrf_nc      :   filename of wrf file requested.
                            If no wrfout file is explicitly specified, the
                            netCDF file in that folder is chosen if unambiguous.
        out_sd      :   subdirectory of output .png.
        f_prefix    :   custom filename prefix
        f_suffix    :   custom filename suffix
        bounding    :   list of four floats (Nlim, Elim, Slim, Wlim):
            Nlim    :   northern limit
            Elim    :   eastern limit
            Slim    :   southern limit
            Wlim    :   western limit
        smooth      :   smoothing. 0 is off. non-zero integer is the degree
                        of smoothing, to be specified.
        dom         :   domain for plotting. If zero, the only netCDF file present
                        will be plotted. If list of integers, the script will loop over domains.
                        
                        
        """
        self.W = self.get_wrfout(wrf_sd,wrf_nc,dom=dom)

        outpath = self.get_outpath(out_sd)

            
        # Make sure times are in datenum format and sequence.
        t_list = utils.ensure_sequence_datenum(times)
        
        d_list = utils.get_sequence(dom)
        lv_list = utils.get_sequence(levels)
        for t, l, d in itertools.product(t_list,lv_list,d_list):
            F = BirdsEye(self.C,self.W)
            F.plot2D(vrbl,t,l,d,outpath,bounding=bounding)
            
        # LEVELS
        # Levels may not exist for CAPE, shear etc.
        # Use the '1' code in this case.
        #if not 'lv' in rq[va]:
        #    rq[va]['lv'] = 'all'
        #lvs = getattr(request[va],'lv',1)
        #lvs = self.get_list(request[va],'lv',(1,))
        #vc = utils.level_type(lv) # vertical coordinate

        # TIMES
        # if not 'pt' in rq[va]: # For averages and all times
            # if not 'itime' in rq[va]: # For all times
                # rq[va]['pt'] = ['all',]
            # else: # For specific range
                # rq[va]['pt'] = ['range',]

        # Check for pressure levels
        #if vc == 'isobaric':
        #    nc_path = self.W.path
        #    p_interp_fpath = self.W.interp_to_p(self.C,nc_path,va,lv)
        #    # Edit p_interp namelist
        #    #Execute p_interp here and reassign self.W to new file
        #    self.W = WRFOut(p_interp_fpath)
        #else: #
        #    # print("Non-pressure levels not supported yet.")
        #    # raise Exception
        #    pass

        # for t in ts:
            # for lv in lvs:
                # For each time, variable, and level:
                # Create figure
                # F = BirdsEye(self.C,self.W)
                #disp_t = utils.string_from_time('title',t,**rq[va])
                #print("Plotting {0} at lv {1} for time {2}.".format(va,lv,disp_t))
                #rq[va]['pt'] = t # Need this?
                #rq[va]['vc'] = vc # Need this?
                # F.plot2D(va, t, lv, )

    def get_wrfout(self,wrf_sd=0,wrf_nc=0,dom=0):
        """Returns the WRFOut instance, given arguments:
        
        Optional inputs:
        wrf_sd      :   subdirectory for wrf file
        wrf_nc      :   filename for wrf file
        dom         :   domain for wrf file
        """
        
        if wrf_sd and wrf_nc:
            wrfpath = os.path.join(self.C.wrfout_root,wrf_sd,wrf_nc)
        elif wrf_sd:
            wrfdir = os.path.join(self.C.wrfout_root,wrf_sd)
            wrfpath = utils.wrfout_files_in(wrfdir,dom=dom,unambiguous=1)
        else:
            wrfdir = os.path.join(self.C.wrfout_root)
            wrfpath = utils.wrfout_files_in(wrfdir,dom=dom,unambiguous=1)
            
        return WRFOut(wrfpath)

    def generate_times(self,itime,ftime,x):
        """
        Wrapper for utility method.
        """
        y = utils.generate_times(itime,ftime,x)
        return y
            
    def plot_cross_section(self,var,latA,lonA,latB,lonB):
        xs = CrossSection()
        xs.plot(var,latA,lonA,latB,lonB)

    def save_data(self,data,folder,fname,format='pickle'):
        """
        Save array to file.
        Needed by subclasses?
        """

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
        """
        Load array from file.
        Needed by subclasses?
        """
        
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

        print("Saving pickle file to {0}".format(d_save))
        # Look up the method to use depending on type of plot
        PLOTS = {'sum_z':self.DE_z, 'sum_xyz':self.DE_xyz}

        print('Get sequence of time')
        # Creates sequence of times
        ts = self.get_sequence(times)

        # Dictionary of data
        DATA = {}

        print('Get permutations')
        # Get all permutations of files
        nperm = len(list(itertools.combinations(files,2)))
        print('Start loop')
        # pdb.set_trace()
        for n, perm in enumerate(itertools.combinations(files,2)):
            print("No. {0} from {1} permutations".format(n,nperm))
            perm_start = time.time()
            DATA[str(n)] = {}
            f1, f2 = perm
            W1 = WRFOut(f1)
            W2 = WRFOut(f2)
            print('WRFOuts loaded.')
            #pdb.set_trace()
            # Make sure times are the same in both files
            if not N.all(N.array(W1.wrf_times) == N.array(W2.wrf_times)):
                print("Times are not identical between input files.")
                raise Exception
            else:
                print("Passed check for identical timestamps between "
                      "NetCDF files")

            # Find indices of each time
            print('Finding time indices')
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
        U0 = nc0.variables['U'][t,...]
        U1 = nc1.variables['U'][t,...]
        Ud = U0 - U1
        #del U0, U1

        V0 = nc0.variables['V'][t,...]
        V1 = nc1.variables['V'][t,...]
        Vd = V0 - V1
        #del V0, V1

        # PERT and BASE PRESSURE
        if lower or upper:
            P0 = nc0.variables['P'][t,...]
            PB0 = nc0.variables['PB'][t,...]
            Pr = P0 + PB0
            #del P0, PB1
            # Here we assume pressure columns are
            # roughly the same between the two...

        if energy=='total':
            T0 = nc0.variables['T'][t,...]
            T1 = nc1.variables['T'][t,...]
            Td = T0 - T1
            #del T0, T1

            R = 287.0 # Universal gas constant (J / deg K * kg)
            Cp = 1004.0 # Specific heat of dry air at constant pressure (J / deg K * kg)
            kappa = R/Cp

        xlen = Ud.shape[1] # 1 less than in V
        ylen = Vd.shape[2] # 1 less than in U
        zlen = Ud.shape[0] # identical in U & V

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
            # Find closest level to 'lower', 'upper'
            if lower or upper:
                P_col = Pr[:,j,i]
            if lower:
                low_idx = utils.closest(P_col,lower*100.0)
            else:
                low_idx = None
            if upper:
                upp_idx = utils.closest(P_col,upper*100.0)+1
            else:
                upp_idx = None

            zidx = slice(low_idx,upp_idx)

            if energy=='kinetic':
                DKE2D[j,i] = N.sum(0.5*((Ud[zidx,j,i])**2 +
                                    (Vd[zidx,j,i])**2))
            elif energy=='total':
                DKE2D[j,i] = N.sum(0.5*((Ud[zidx,j,i])**2 +
                                    (Vd[zidx,j,i])**2 +
                                    kappa*(Td[zidx,j,i])**2))

        DKE.append(DKE2D)

        return DKE

    def plot_diff_energy(self,ptype,energy,time,folder,fname,V):
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
                f1 = DATA[perm]['file1']
                f2 = DATA[perm]['file2']
                if sw==0:
                    # Get times and info about nc files
                    # First time to save power
                    W1 = WRFOut(f1)
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
            F = BirdsEye(self.C,W1)    # 2D figure class
            #F.plot2D(va,t,en,lv,da,na)  # Plot/save figure
            fname_t = ''.join((fname,'_p{0:02d}'.format(n)))
            F.plot_data(stack_average,'contourf',fname_t,t,V)
            print("Plotting time {0} from {1}.".format(n,len(times)))
            del data, stack

    def plot_error_growth(self,ofname,folder,pfname,sensitivity=0,ylimits=0,**kwargs):
        """Plots line graphs of DKE/DTE error growth
        varying by a sensitivity - e.g. error growth involving
        all members that use a certain parameterisation.

        ofname          :   output filename prefix
        pfname          :   pickle filename
        plotlist        :   list of folder names to loop over
        ylim            :   tuple of min/max for y axis range
        """
        DATA = self.load_data(folder,pfname,format='pickle')

        for perm in DATA:
            times = DATA[perm]['times']
            break

        times_tup = [time.gmtime(t) for t in times]
        time_str = ["{2:02d}/{3:02d}".format(*t) for t in times_tup]

        if sensitivity:
            # Plot multiple line charts for each sensitivity
            # Then a final chart with all the averages
            # If data is 2D, sum over x/y to get one number
            
            # Dictionary with average
            AVE = {}
            
            for sens in sensitivity:
                ave_stack = 0
                n_sens = len(sensitivity)-1
                colourlist = utils.generate_colours(M,n_sens)
                M.rcParams['axes.color_cycle'] = colourlist
                fig = plt.figure()
                labels = []
                #SENS['sens'] = {}
                for perm in DATA:
                    f1 = DATA[perm]['file1']
                    f2 = DATA[perm]['file2']
    
                    if sens in f1:
                        f = f2
                    elif sens in f2:
                        f = f1
                    else:
                        f = 0
    
                    if f:
                        subdirs = f.split('/')
                        labels.append(subdirs[-2])
                        data = self.make_1D(DATA[perm]['values'])
    
                        plt.plot(times,data)
                        # pdb.set_trace()
                        ave_stack = utils.vstack_loop(N.asarray(data),ave_stack)
                    else:
                        pass
    
                # pdb.set_trace()
                n_sens += 1
                colourlist = utils.generate_colours(M,n_sens)
                M.rcParams['axes.color_cycle'] = colourlist
                AVE[sens] = N.average(ave_stack,axis=0)
                labels.append('Average')
                plt.plot(times,AVE[sens],'k')
    
                plt.legend(labels,loc=2,fontsize=9)
                if ylimits:
                    plt.ylim(ylimits)
                plt.gca().set_xticks(times[::2])
                plt.gca().set_xticklabels(time_str[::2])
                outdir = self.C.output_root
                fname = '{0}_Growth_{1}.png'.format(ofname,sens)
                fpath = os.path.join(outdir,fname)
                fig.savefig(fpath)
    
                plt.close()
                print("Saved {0}.".format(fpath))
                
            # Averages for each sensitivity
            labels = []
            fig = plt.figure()
            ave_of_ave_stack = 0
            for sens in AVE.keys():
                plt.plot(times,AVE[sens])
                labels.append(sens)
                ave_of_ave_stack = utils.vstack_loop(AVE[sens],ave_of_ave_stack)
                
            labels.append('Average')
            ave_of_ave = N.average(ave_of_ave_stack,axis=0)
            plt.plot(times,ave_of_ave,'k')
            
            plt.legend(labels,loc=2,fontsize=9)
            
            if ylimits:
                plt.ylim(ylimits)
            plt.gca().set_xticks(times[::2])
            plt.gca().set_xticklabels(time_str[::2])
            outdir = self.C.output_root
            fname = '{0}_Growth_Averages.png'.format(ofname)
            fpath = os.path.join(outdir,fname)
            fig.savefig(fpath)

            plt.close()
            print("Saved {0}.".format(fpath))
            #pdb.set_trace()
                            
            

        else:
            fig = plt.figure()
            ave_stack = 0
            for perm in DATA:
                data = self.make_1D(DATA[perm]['values'])
                plt.plot(times,data,'blue')
                ave_stack = utils.vstack_loop(N.asarray(data),ave_stack)

            total_ave = N.average(ave_stack,axis=0)
            plt.plot(times,total_ave,'black')
            
            if ylimits:
                plt.ylim(ylimits)
            plt.gca().set_xticks(times[::2])
            plt.gca().set_xticklabels(time_str[::2])
            outdir = self.C.output_root
            fname = '{0}_Growth_allmembers.png'.format(ofname)
            fpath = os.path.join(outdir,fname)
            fig.savefig(fpath)

            plt.close()
            print("Saved {0}.".format(fpath))
            
    def composite_profile(self,va,skewT_time,skewT_latlon,enspaths,dom=1,mean=0,std=0,xlim=0,ylim=0):
        P = Profile(self.C)
        P.composite_profile(va,skewT_time,skewT_latlon,enspaths,dom,mean,std,xlim,ylim)

    def plot_skewT(self,plot_time,plot_latlon,dom=1,save_output=0,composite=0):
        wrfouts = self.wrfout_files_in(self.C.wrfout_root)
        for wrfout in wrfouts:
            if not composite:
                W = WRFOut(wrfout)
                ST = SkewT(self.C,W)
                ST.plot_skewT(plot_time,plot_latlon,dom,save_output)
                nice_time = utils.string_from_time('title',plot_time)
                print("Plotted Skew-T for time {0} at {1}".format(
                            nice_time,plot_latlon))
            else:
                #ST = SkewT(self.C)
                pass
                
    def plot_streamlines(self,lv,times):
        wrfpath = self.wrfout_files_in(self.C.wrfout_root)[0]
        self.W = WRFOut(wrfpath)
        self.F = BirdsEye(self.C,self.W)
        for pt in times:
            disp_t = utils.string_from_time('title',pt)
            print("Plotting {0} at lv {1} for time {2}.".format(
                    'streamlines',lv,disp_t))
            self.F.plot_streamlines(lv,pt)

    def plot_strongest_wind(self,itime,ftime,levels,wrf_sd=0,wrf_nc=0,out_sd=0,f_prefix=0,f_suffix=0,
                bounding=0,dom=0):
        """
        Plot strongest wind at level lv between itime and ftime.
        
        Path to wrfout file is in config file.
        Path to plot output is also in config


        Inputs:
        levels      :   level(s) for wind
        wrf_sd      :   string - subdirectory of wrfout file
        wrf_nc      :   filename of wrf file requested.
                            If no wrfout file is explicitly specified, the
                            netCDF file in that folder is chosen if unambiguous.
        out_sd      :   subdirectory of output .png.
        f_prefix    :   custom filename prefix
        f_suffix    :   custom filename suffix
        bounding    :   list of four floats (Nlim, Elim, Slim, Wlim):
            Nlim    :   northern limit
            Elim    :   eastern limit
            Slim    :   southern limit
            Wlim    :   western limit
        smooth      :   smoothing. 0 is off. non-zero integer is the degree
                        of smoothing, to be specified.
        dom         :   domain for plotting. If zero, the only netCDF file present
                        will be plotted. If list of integers, the script will loop over domains.
                        
                        
        """
        self.W = self.get_wrfout(wrf_sd,wrf_nc,dom=dom)

        outpath = self.get_outpath(out_sd)
            
        # Make sure times are in datenum format and sequence.
        it = utils.ensure_sequence_datenum(itime)
        ft = utils.ensure_sequence_datenum(ftime)
        
        d_list = utils.get_sequence(dom)
        lv_list = utils.get_sequence(levels)
        
        for l, d in itertools.product(lv_list,d_list):
            F = BirdsEye(self.C,self.W)
            F.plot2D('strongestwind',it+ft,l,d,outpath,bounding=bounding)
 
    def make_1D(self,data,output='list'):
        """ Make sure data is a time series
        of 1D values, and numpy array.

        List of arrays -> Numpy array or list
        """
        if isinstance(data,list):

            data_list = []
            for time in data:
                data_list.append(N.sum(time[0]))

            if output == 'array':
                data_out = N.array(data_list)
            else:
                data_out = data_list


        #elif isinstance(data,N.array):
        #    shape = data.shape
        #    if len(shape) == 1:
        #        data_out = data
        #    elif len(shape) == 2:
        #        data_out = N.sum(data)
        return data_out


    def get_list(self,dic,key,default):
        """Fetch value from dictionary.
        
        If it doesn't exist, use default.
        If the value is an integer, make a list of one.
        """

        val = getattr(dic,key,default)
        if isinstance(val,'int'):
            lst = (val,)
        else:
            lst = val
        return lst

    def get_outpath(self,out_sd):
        if out_sd:
            outpath = os.path.join(self.C.output_root,out_sd)
        else:
            outpath = self.C.output_root
        return outpath

    def plot_xs(self,vrbl,times,latA=0,lonA=0,latB=0,lonB=0,
                wrf_sd=0,wrf_nc=0,out_sd=0,f_prefix=0,f_suffix=0,dom=0,
                clvs=0,ztop=0):
        """
        Plot cross-section.
        
        If no lat/lon transect is indicated, a popup appears for the user
        to pick points. The popup can have an overlaid field such as reflectivity
        to help with the process.
        
        Inputs:
        vrbl        :   variable to be plotted
        times       :   times to be plotted
        latA        :   start latitude of transect
        lonA        :   start longitude of transect
        latB        :   end lat...
        lonB        :   end lon...
        wrf_sd      :   string - subdirectory of wrfout file
        wrf_nc      :   filename of wrf file requested.
                            If no wrfout file is explicitly specified, the
                            netCDF file in that folder is chosen if unambiguous.
        out_sd      :   subdirectory of output .png.
        f_prefix    :   custom filename prefix
        f_suffix    :   custom filename suffix
        clvs        :   custom contour levels
        ztop        :   highest km to plot.
        
        """
        self.W = self.get_wrfout(wrf_sd,wrf_nc,dom=dom)
        
        outpath = self.get_outpath(out_sd)
     
        XS = CrossSection(self.C,self.W,latA,lonA,latB,lonB)
        
        t_list = utils.ensure_sequence_datenum(times)
        for t in t_list:
            XS.plot_xs(vrbl,t,outpath,clvs=clvs,ztop=ztop)
