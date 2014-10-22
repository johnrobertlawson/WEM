"""This handles user requests and controls computations and plotting.

This script is API and should not be doing any hard work of
importing matplotlib etc!

Useful/utility scripts are in WEM.utils.

TODO: move all DKE stuff to stats and figure/birdseye.
TODO: move more utilities to utils.
"""
from netCDF4 import Dataset
import calendar
import collections
import copy
import cPickle as pickle
import fnmatch
import glob
import itertools
import numpy as N
import os
import pdb
import time
import matplotlib as M
M.use('gtkagg')
import matplotlib.pyplot as plt

from wrfout import WRFOut
from figure import Figure
from birdseye import BirdsEye
from ruc import RUC
from skewt import SkewT
from skewt import Profile
#import scales
from defaults import Defaults
import WEM.utils as utils
from xsection import CrossSection
from clicker import Clicker
import maps
import stats

# TODO: Make this awesome

class WRFEnviron(object):
    def __init__(self):
        # Set defaults
        self.D = Defaults()

        #self.font_prop = getattr(self.C,'font_prop',self.D.font_prop)
        #self.usetex = getattr(self.C,'usetex',self.D.usetex)
        #self.plot_titles = getattr(self.C,'plot_titles',self.D.plot_titles)
        #M.rc('text',usetex=self.usetex)
        #M.rc('font',**self.font_prop)
        #M.rcParams['savefig.dpi'] = self.dpi

    def plot2D(self,vrbl,utc,level,ncdir,outdir,ncf=False,nct=False,
                f_prefix=0,f_suffix=False,bounding=False,dom=1,
                plottype='contourf',smooth=1,fig=False,ax=False):
        """
        Basic birds-eye-view plotting.

        This script is top-most and decides if the variables is
        built into WRF default output or needs computing. It unstaggers
        and slices data from the wrfout file appropriately.


        Inputs:
        vrbl        :   string of variable name as found in WRF, or one of
                        the computed fields available in WEM
        utc    :        one date/time.
                        Can be tuple (YYYY,MM,DD,HH,MM,SS - calendar.timegm)
                        Can be integer of datenum. (time.gmtime)
        level       :   one level.
                        Lowest model level is integer 2000.
                        Pressure level is integer in hPa, e.g. 850
                        Isentropic surface is a string + K, e.g. '320K'
                        Geometric height is a string + m, e.g. '4000m'
        ncdir       :   directory of netcdf data file
        outdir      :   directory to save output figures

        Optional
        ncf         :   filename of netcdf data file if ambiguous within ncdir.
                        If no wrfout file is explicitly specified, the
                        netCDF file in that folder is chosen if unambiguous.
        nct         :   initialisation time of netcdf data file, if
                        ambiguous within ncdir.
        f_prefix    :   custom filename prefix for output
        f_suffix    :   custom filename suffix for output
        bounding    :   dictionary of four floats (Nlime, Elim, Slim, Wlim):
            Nlim    :   northern limit
            Elim    :   eastern limit
            Slim    :   southern limit
            Wlim    :   western limit
        smooth      :   smoothing. 0 is off. non-zero integer is the degree
                        of smoothing, to be specified.
        dom         :   domain for plotting (for WRF data). If zero, the only netCDF file
                        present will be plotted.
        plottype    :   matplotlib command for plotting data.
                        contour or contourf.
        fig         :   matplotlib.figure object to plot onto
        ax          :   matplotlib.axis object to plot onto


        """
        if isinstance(level,int):
            # Level is in pressure
            level = '{0}hPa'.format(level)

        # Data
        self.W = self.get_netcdf(ncdir,ncf=ncf,nct=nct,dom=dom)
        lats, lons = self.W.get_limited_domain(bounding)
        data = self.W.get(vrbl,utc,level,lons,lats)

        # Figure
        fname = self.create_fname(vrbl,utc,level)
        F = BirdsEye(self.W,fig=fig,ax=ax)
        F.plot2D(data,fname,outdir,lats=lats,lons=lons,
                    plottype=plottype,smooth=smooth)

    def create_fname(self,vrbl,utc,level,f_prefix=False,f_suffix=False):
        """
        Differentiate between e.g. different domains by using the suffix/prefix
        options.
        """
        time_str = utils.string_from_time('output',utc)

        fname = '_'.join((vrbl,time_str,level))

        if isinstance(f_prefix,basestring):
            fname = f_prefix + fname
        if isinstance(f_suffix,basestring):
            fname = fname + f_suffix
        return fname

    def get_netcdf(self,ncdir,ncf=False,nct=False,dom=0,path_only=False):
        """Returns the WRFOut or RUC instance, given arguments:

        ncdir       :   absolute path to subdirectory

        Optional inputs:
        ncf         :   file name
        nct         :   initialisation time (tuple)
        dom         :   domain (for WRF files only)
        path_only   :   if True, return only absolute path+fname
        """
        if ncf:
            fpath = os.path.join(ncdir,ncf)
            model = utils.determine_model(ncf)
        else:
            fpath, model = utils.netcdf_files_in(ncdir,init_time=nct,
                                                    dom=dom,return_model=True)

        # import pdb; pdb.set_trace()
        if path_only:
            return fpath
        else:
            # Check for WRF or RUC
            # nc = Dataset(wrfpath)
            # if 'ruc' in nc.grib_source[:3]:
            if model=='ruc':
                return RUC(fpath)
            elif model=='wrfout':
                return WRFOut(fpath)
            else:
                print("Unrecognised netCDF4 file type at {0}".format(fpath))

    def get_wrfout(self,wrf_sd=0,wrf_nc=0,dom=0,path_only=0):
        """Returns the WRFOut or RUC instance, given arguments:

        Optional inputs:
        wrf_sd      :   subdirectory for wrf file
        wrf_nc      :   filename for wrf file
        dom         :   domain for wrf file
        path_only   :   only return absolute path
        """
        # Check configuration to see if wrfout files should be
        # sought inside subdirectories.
        descend = getattr(self.C,'wrf_folders_descend',1)
        import pdb; pdb.set_trace()

        if wrf_sd and wrf_nc:
            wrfpath = os.path.join(self.C.wrfout_root,wrf_sd,wrf_nc)
        elif wrf_sd:
            wrfdir = os.path.join(self.C.wrfout_root,wrf_sd)
            # print wrfdir
            wrfpath = utils.wrfout_files_in(wrfdir,dom=dom,unambiguous=1,descend=descend)
        else:
            wrfdir = os.path.join(self.C.wrfout_root)
            wrfpath = utils.wrfout_files_in(wrfdir,dom=dom,unambiguous=1,descend=descend)

        if path_only:
            return wrfpath
        else:
            # Check for WRF or RUC
            # nc = Dataset(wrfpath)
            # if 'ruc' in nc.grib_source[:3]:
            if 'ruc' in wrfpath[:5]:
                return RUC(wrfpath)
            elif 'wrfout' in wrfpath[:6]:
                return WRFOut(wrfpath)
            else:
                print("Unrecognised netCDF file type at {0}".format(wrfpath))

    def generate_times(self,itime,ftime,interval):
        """
        Wrapper for utility method.

        itime   :   Time tuple of start time
        ftime   :   Time tuple of end time
        interval:   interval
        """
        listoftimes = utils.generate_times(itime,ftime,interval)
        return listoftimes

    def plot_diff_energy(self,vrbl,utc,energy,datadir,outdir,dataf=False,
                            outprefix=False,outsuffix=False,clvs=0,
                            title=False,fig=False,ax=False):
        """
        vrbl        :   'sum_z' or 'sum_xyz'
        utc         :   date/time for plot
        energy      :   'kinetic' or 'total'
        datadir     :   directory holding computed data
        outdir      :   root directory for plots

        Optional
        dataf       :   file name of data file, if ambiguous
        outprefix   :   prefix for output files
        outsuffix   :   suffix for output files
        clvs        :   contour levels
        title       :   title for output
        fig         :   matplotlib.figure object to plot on
        ax          :   matplotib.axis object to plot on

        TODO: find data file automatically, given folder
        """
        DATA = utils.load_data(folder,fname,format='pickle')

        if isinstance(utc,collections.Sequence):
            utc = calendar.timegm(utc)

        for pn,perm in enumerate(DATA):
            f1 = DATA[perm]['file1']
            f2 = DATA[perm]['file2']
            # Get times and info about nc files
            # First time to save power
            W1 = WRFOut(f1)
            permtimes = DATA[perm]['times']

            # Find array for required time
            x = N.where(N.array(permtimes)==utc)[0][0]
            data = DATA[perm]['values'][x][0]
            if not pn:
                stack = data
            else:
                stack = N.dstack((data,stack))
                stack_average = N.average(stack,axis=2)

        kwargs1 = {}
        kwargs2 = {}
        if ax:
            kwargs1['ax'] = ax
            kwargs2['save'] = 0
        if title:
            kwargs2['title'] = 1

        #birdseye plot with basemap of DKE/DTE
        F = BirdsEye(self.C,W1,**kwargs1)    # 2D figure class
        tstr = utils.string_from_time('output',utc)
        fname_t = ''.join((plotname,'_{0}'.format(tstr)))
        # fpath = os.path.join(p2p,fname_t)
        import pdb; pdb.set_trace()

        fig_obj = F.plot_data(stack_average,'contourf',p2p,fname_t,utc,V,
                                **kwargs2)


    def delta_diff_energy(self,vrbl,utc0,utc1,energy,datadir,outdir,
                            meanvrbl='Z',meanlevel=500,
                            dataf=False,outprefix=False,outsuffix=False,
                            clvs=0,title=False,fig=False,ax=False,ncdata=False):
        """
        Plot DKE/DTE growth with time, DDKE/DDTE (contours) over (optional)
        ensemble mean of a variable.

        Will calculate DDKE/DDTE for halfway between time0 and time1.

        vrbl        :   'sum_z' or 'sum_xyz'
        utc0       :   first time, must exist in pickle file
        utc1       :   second time, ditto
        energy      :   'kinetic' or 'total'
        datadir     :   directory holding computed data
        outdir      :   root directory for plots

        Optional
        meanvrbl    :   variable of ensemble mean
        meanlevel   :   level for ensemble mean variable
        ncdata      :   if meanvrbl is not False, link to all netcdf files of
                        ensemble members for ensemble
        dataf       :   file name of data file, if ambiguous
        outprefix   :   prefix for output files
        outsuffix   :   suffix for output files
        clvs        :   contour levels
        title       :   title for output
        fig         :   matplotlib.figure object to plot on
        ax          :   matplotib.axis object to plot on

        TODO: Interpolate geopotential height to pressure level.
        """

        data = self.load_data(folder,fname,format='pickle')

        for n, perm in enumerate(data):
            if n==0:
                permtimes = data[perm]['times']
                deltatimes = [(t0+t1)/2.0 for t0,t1 in zip(permtimes[:-2],permtimes[1:])]
                W1 = WRFOut(data[perm]['file1'])
                break

        for t, delt in enumerate(deltatimes):
            print('Computing for time {0}'.format(time.gmtime(delt)))
            for n, perm in enumerate(data):
                diff0 = data[perm]['values'][t][0]
                diff1 = data[perm]['values'][t+1][0]
                if n==0:
                    stack0 = diff0
                    stack1 = diff1
                else:
                    stack0 = N.dstack((diff0,stack0))
                    stack1 = N.dstack((diff1,stack1))

            for wnum,wrf in enumerate(wrfouts):
                W2 = Dataset(wrf)
                ##### NEEDS REFACTORING #######
                ght = TimeLevelLatLonWRF(W1.nc,W2,times=delt)
                ##### NEEDS REFACTORING #######
                if wnum==0:
                    ghtstack = ght
                else:
                    ghtstack = N.dstack((ght,ghtstack))

            heightmean = N.average(ghtstack,axis=2)

            delta = N.average(stack1,axis=2) - N.average(stack0,axis=2)

            F = BirdsEye(self.C, W1)
            tstr = utils.string_from_time('output',int(delt))
            fname_t = ''.join((plotname,'_{0}'.format(tstr)))
            F.plot_data(delta,'contourf',p2p,fname_t,delt,
                        save=0,levels=N.arange(-2000,2200,200))
            F.plot_data(heightmean,'contour',p2p,fname_t,delt,
                       colors='k',levels=N.arange(2700,3930,30))

    def plot_error_growth(self,datadir,outprefix=False,dataf=False,
                            sensitivity=0,ylim=0,outsuffix=False):
        """Plots line graphs of DKE/DTE error growth
        varying by a sensitivity - e.g. error growth involving
        all members that use a certain parameterisation.

        datadir         :   folder with pickle data

        Optional
        outprefix       :   output filename prefix
        dataf           :   pickle filename if ambiguous
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

    def composite_profile(self,va,utc,latlon,enspaths,
                            dom=0,mean=0,std=0,xlim=0,ylim=0):
        P = Profile(self.C)
        P.composite_profile(va,utc,latlon,enspaths,dom,mean,std,xlim,ylim)

    def twopanel_profile(self,va,utc,wrf_sds,out_sd,two_panel=1,dom=1,mean=1,std=1,
                         xlim=0,ylim=0,latlon=0,locname=0,overlay=0,ml=-2):
        """
        Create two-panel figure with profile location on map,
        with profile of all ensemble members in comparison.

        Inputs:
        va          :   variable for profile
        utc        :   time of plot
        wrf_sds     :   subdirs containing wrf file
        out_d       :   out directory for plots

        Optional:
        two_panel   :   add inset for plot location
        dom         :   WRF domain to use
        mean        :   overlay mean on profile
        std         :   overlay +/- std dev on profile
        xlim        :   three-item list/tuple with limits, spacing interval
                        for xaxis, in whatever default units
        ylim        :   similarly for yaxis but in hPa
                        or dictionary with locations (METAR etc) and two-item tuple
        latlon      :   two-item list/tuple with lat/lon.
                        If not specified, use pop-ups to select.
        locname     :   pass this to the filename of output for saving
        overlay     :   data from the same time to overlay on inset
        ml          :   member level. negative number that corresponds to the
                        folder in absolute string for naming purposes.


        """
        # Initialise with first wrfout file
        self.W = self.get_netcdf(wrf_sds[0],dom=dom)
        outpath = self.get_outpath(out_sd)

        # Get list of all wrfout files
        enspaths = self.list_ncfiles(wrf_sds)

        self.data = 0
        if two_panel:
            P2 = Figure(self.C,self.W,layout='inseth')
            if overlay:
                F = BirdsEye(self.C, self.W)
                self.data = F.plot2D('cref',utc,2000,dom,outpath,save=0,return_data=1)

        # TODO: Not sure basemap inset works for lat/lon specified
        if isinstance(latlon,collections.Sequence):
            if not len(latlon) == 2:
                print("Latitude and longitude needs to be two-item list/tuple.")
                raise Exception
            lat0,lon0 = latlon
            C = Clicker(self.C,self.W,fig=P2.fig,ax=P2.ax0,data=self.data)
            x0, y0 = C.bmap(lon0,lat0)
            C.ax.scatter(x0,y0,marker='x')
        else:
            t_long = utils.string_from_time('output',utc)
            print("Pick location for {0}".format(t_long))
            C = Clicker(self.C,self.W,fig=P2.fig,ax=P2.ax0,data=self.data)
            # fig should be P2.fig.
            # C.fig.tight_layout()

            # Pick location for profile
            C.click_x_y(plotpoint=1)
            lon0, lat0 = C.bmap(C.x0,C.y0,inverse=True)


        # Compute profile
        P = Profile(self.C)
        P.composite_profile(va,utc,(lat0,lon0),enspaths,outpath,dom=dom,mean=mean,
                            std=std,xlim=xlim,ylim=ylim,fig=P2.fig,ax=P2.ax1,
                            locname=locname,ml=ml)


    def plot_skewT(self,utc,latlon,out_sd=0,wrf_sd=0,dom=1,save_output=0,composite=0):

        outpath = self.get_outpath(out_sd)
        W = self.get_netcdf(wrf_sd,dom=dom)

        if not composite:
            ST = SkewT(self.C,W)
            ST.plot_skewT(utc,plot_latlon,dom,outpath,save_output=save_output)
            nice_time = utils.string_from_time('title',utc)
            print("Plotted Skew-T for time {0} at {1}".format(
                        nice_time,plot_latlon))
        else:
            #ST = SkewT(self.C)
            pass

    def plot_streamlines(self,utc,lv,wrf_sd=0,wrf_nc=0,out_sd=0,dom=1):
        self.W = self.get_netcdf(wrf_sd,wrf_nc,dom=dom)
        outpath = self.get_outpath(out_sd)

        self.F = BirdsEye(self.C,self.W)
        disp_t = utils.string_from_time('title',utc)
        print("Plotting {0} at lv {1} for time {2}.".format(
                'streamlines',lv,disp_t))
        self.F.plot_streamlines(lv,utc,outpath)

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
        self.W = self.get_netcdf(wrf_sd,wrf_nc,dom=dom)

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
            for utc in data:
                data_list.append(N.sum(utc[0]))

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


    def plot_xs(self,vrbl,utc,latA=0,lonA=0,latB=0,lonB=0,
                wrf_sd=0,wrf_nc=0,out_sd=0,f_prefix=0,f_suffix=0,dom=0,
                clvs=0,ztop=0):
        """
        Plot cross-section.

        If no lat/lon transect is indicated, a popup appears for the user
        to pick points. The popup can have an overlaid field such as reflectivity
        to help with the process.

        Inputs:
        vrbl        :   variable to be plotted
        utc       :   times to be plotted
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
        self.W = self.get_netcdf(wrf_sd,wrf_nc,dom=dom)

        outpath = self.get_outpath(out_sd)

        XS = CrossSection(self.C,self.W,latA,lonA,latB,lonB)

        XS.plot_xs(vrbl,utc,outpath,clvs=clvs,ztop=ztop)


    def cold_pool_strength(self,utc,wrf_sd=0,wrf_nc=0,out_sd=0,
                            swath_width=100,dom=1,twoplot=0,fig=0,
                            axes=0,dz=0):
        """
        Pick A, B points on sim ref overlay
        This sets the angle between north and line AB
        Also sets the length in along-line direction
        For every gridpt along line AB:
            Locate gust front via shear
            Starting at front, do 3-grid-pt-average in line-normal
            direction

        utc    :   time (tuple or datenum) to plot
        wrf_sd  :   string - subdirectory of wrfout file
        wrf_nc  :   filename of wrf file requested.
                            If no wrfout file is explicitly specified, the
                            netCDF file in that folder is chosen if unambiguous.
        out_sd      :   subdirectory of output .png.
        swath_width :   length in gridpoints in cross-section-normal direction
        dom     :   domain number
        return2 :   return two figures. cold pool strength and cref/cross-section.
        axes    :   if two-length tuple, this is the first and second axes for
                    cross-section/cref and cold pool strength, respectively
        dz      :   plot height of cold pool only.

        """
        # Initialise
        self.W = self.get_netcdf(wrf_sd,wrf_nc,dom=dom)
        outpath = self.get_outpath(out_sd)

        # keyword arguments for plots
        line_kwargs = {}
        cps_kwargs = {}
        # Create two-panel figure
        if twoplot:
            P2 = Figure(self.C,self.W,plotn=(1,2))
            line_kwargs['ax'] = P2.ax.flat[0]
            line_kwargs['fig'] = P2.fig
            P2.ax.flat[0].set_size_inches(3,3)

            cps_kwargs['ax'] = P2.ax.flat[1]
            cps_kwargs['fig'] = P2.fig
            P2.ax.flat[1].set_size_inches(6,6)

        elif isinstance(axes,tuple) and len(axes)==2:
            line_kwargs['ax'] = axes[0]
            line_kwargs['fig'] = fig

            cps_kwargs['ax'] = axes[1]
            cps_kwargs['fig'] = fig

            return_ax = 1

        # Plot sim ref, send basemap axis to clicker function
        F = BirdsEye(self.C,self.W)
        self.data = F.plot2D('cref',utc,2000,dom,outpath,save=0,return_data=1)

        C = Clicker(self.C,self.W,data=self.data,**line_kwargs)
        # C.fig.tight_layout()

        # Line from front to back of system
        C.draw_line()
        # C.draw_box()
        lon0, lat0 = C.bmap(C.x0,C.y0,inverse=True)
        lon1, lat1 = C.bmap(C.x1,C.y1,inverse=True)

        # Pick location for environmental dpt
        # C.click_x_y()
        # Here, it is the end of the cross-section
        lon_env, lat_env = C.bmap(C.x1, C.y1, inverse=True)
        y_env,x_env,exactlat,exactlon = utils.getXY(self.W.lats1D,self.W.lons1D,lat_env,lon_env)
        # Create the cross-section object
        X = CrossSection(self.C,self.W,lat0,lon0,lat1,lon1)

        # Ask user the line-normal box width (self.km)
        #C.set_box_width(X)

        # Compute the grid (DX x DY)
        cps = self.W.cold_pool_strength(X,utc,swath_width=swath_width,env=(x_env,y_env),dz=dz)
        # import pdb; pdb.set_trace()

        # Plot this array
        CPfig = BirdsEye(self.C,self.W,**cps_kwargs)
        tstr = utils.string_from_time('output',utc)
        if dz:
            fprefix = 'ColdPoolDepth_'
        else:
            fprefix = 'ColdPoolStrength_'
        fname = fprefix + tstr

        pdb.set_trace()
        # imfig,imax = plt.subplots(1)
        # imax.imshow(cps)
        # plt.show(imfig)
        # CPfig.plot_data(cps,'contourf',outpath,fname,time,V=N.arange(5,105,5))
        mplcommand = 'contour'
        plotkwargs = {}
        if dz:
            clvs = N.arange(100,5100,100)
        else:
            clvs = N.arange(10,85,2.5)
        if mplcommand[:7] == 'contour':
            plotkwargs['levels'] = clvs
            plotkwargs['cmap'] = plt.cm.ocean_r
        cf2 = CPfig.plot_data(cps,mplcommand,outpath,fname,utc,**plotkwargs)
        # CPfig.fig.tight_layout()

        plt.close(fig)

        if twoplot:
            P2.save(outpath,fname+"_twopanel")

        if return_ax:
            return C.cf, cf2

    def spaghetti(self,utc,lv,va,contour,wrf_sds,out_sd,dom=1):
        """
        Do a multi-member spaghetti plot.

        utc       :   time for plot
        va      :   variable in question
        contour :   value to contour for each member
        wrf_sds :   list of wrf subdirs to loop over
        out_sd  :   directory to save image
        """
        outpath = self.get_outpath(out_sd)

        # Use first wrfout to initialise grid etc
        self.W = self.get_netcdf(wrf_sds[0],dom=dom)
        F = BirdsEye(self.C, self.W)

        ncfiles = []
        for wrf_sd in wrf_sds:
            ncfile = self.get_netcdf(wrf_sd,dom=dom,path_only=1)
            ncfiles.append(ncfile)

        F.spaghetti(utc,lv,va,contour,ncfiles,outpath)

    def std(self,vrbl,utc,level,outdir,wrfdirs=False,wrfpaths=False,dom=1,
                clvs=False):
        """Compute standard deviation of all members
        for given variable.

        Inputs:
        utc     :   time
        level      :   level
        va      :   variable

        Must have one of these two:
        wrfdirs :   list of wrf dirs to loop over

        Optional
        out_sd  :   directory in which to save image
        clvs    :   user-set contour levels
        """
        if wrfdirs is False and wrfpaths is False:
            print("Must have either wrfdirs or wrfpaths.")
            raise Exception
        elif isinstance(wrfdirs,(tuple,list)):
            ncfiles = self.list_ncfiles(wrf_sds)
        elif isinstance()

        # Use first wrfout to initialise grid, get indices
        self.W = self.get_netcdf(wrf_sds[0],dom=dom)

        tidx = self.W.get_time_idx(t)

        if lv==2000:
            # lvidx = None
            lvidx = 0
        else:
            print("Only support surface right now")
            raise Exception

        std_data = stats.std(ncfiles,va,tidx,lvidx)

        F = BirdsEye(self.C, self.W)
        t_name = utils.string_from_time('output',t)
        fname_t = 'std_{0}_{1}'.format(va,t_name)

        # pdb.set_trace()
        plotkwargs = {}
        plotkwargs['no_title'] = 1
        if isinstance(clvs,N.ndarray):
            plotkwargs['clvs'] = clvs
        F.plot_data(std_data,'contourf',outpath,fname_t,t,**plotkwargs)
        print("Plotting std dev for {0} at time {1}".format(va,t_name))

    def list_ncfiles(self,wrf_sds,dom=1,path_only=1):
        ncfiles = []
        for wrf_sd in wrf_sds:
            ncfile = self.get_netcdf(wrf_sd,dom=dom,path_only=path_only)
            ncfiles.append(ncfile)
        return ncfiles

    def plot_domains(self,wrfouts,labels,latlons,out_sd=0,colour=0):
        outpath = self.get_outpath(out_sd)
        maps.plot_domains(wrfouts,labels,latlons,outpath,colour)


    def frontogenesis(self,utc,level,ncdir,outdir,ncf=False,nct=False,
                        dom=1,smooth=0,clvs=0,title=0):
        """
        Compute and plot (Miller?) frontogenesis as d/dt of theta gradient.

        Use a centred-in-time derivative; hence, if
        time index is start or end of wrfout file, skip the plot.

        smooth      :   gaussian smooth by this many grid points
        """
        self.W = self.get_netcdf(ncdir,ncf=ncf,nct=nct,dom=dom)
        fname = self.create_fname(vrbl,utc,level)

        Front = self.W.compute_frontogenesis(utc,level)
        if isinstance(Front,N.ndarray):

            if smooth:
                Front = stats.gauss_smooth(Front,smooth)

            if level==2000:
                lv_str = 'sfc'
            else:
                lv_str = str(level)

                F = BirdsEye(self.C,self.W)
                fname = 'frontogen_{0}_{1}.png'.format(lv_str,tstr)
                F.plot_data(Front,'contourf',outpath,fname,time,clvs=clvs,
                            no_title=no_title,**kwargs)
        else:
            print("Skipping this time; at start or end of run.")
