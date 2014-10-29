""" This handles user requests and controls computations and plotting.

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
from scales import Scales

# TODO: Make this awesome

class WRFEnviron(object):
    """Main environment API.
    """
    def __init__(self):
        """ This currently only loads default settings.
        """
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
                plottype='contourf',smooth=1,fig=False,ax=False,
                clvs=False,cmap=False):
        """Basic birds-eye-view plotting.

        This script is top-most and decides if the variables is
        built into WRF default output or needs computing. It unstaggers
        and slices data from the wrfout file appropriately.

        :param vrbl:        variable name as found in WRF, or one of
                            the computed fields available in WEM
        :type vrbl:         str
        :param utc:         one date/time. The tuple/list format is
                            YYYY,MM,DD,HH,MM,SS (ready for calendar.timegm).
                            Integer format is epoch/datenum (ready for
                            time.gmtime).
        :type utc:          tuple,list,int
        :param level:       required level. 
                            Lowest model level is integer 2000.
                            Pressure level is integer in hPa, e.g. 850.
                            Isentropic surface is a string + K, e.g. '320K'.
                            Geometric height is a string + m, e.g. '4000m'.
        :type level:        int,str
        :param ncdir:       directory of netcdf data file
        :type ncdir:        str
        :param outdir:      directory to save output figures
        :type outdir:       str
        :param ncf:         filename of netcdf data file if ambiguous within ncdir.
                            If no wrfout file is explicitly specified, the
                            netCDF file in that folder is chosen if unambiguous.
        :type ncf:          bool,str
        :param nct:         initialisation time of netcdf data file, if
                            ambiguous within ncdir.
        :type nct:          bool,str
        :param f_prefix:    custom filename prefix for output. Ignore if False.
        :type f_prefix:     bool,str
        :param f_suffix     custom filename suffix for output. Ignore if False.
        :type f_suffix      bool,str
        :param bounding:    bounding box for domain.
                            Dictionary contains four keys (Nlim, Elim, Slim, Wlim)
                            with float values (northern latitude limit, eastern
                            longitude limit, southern latitude limit, western
                            latitude limit, respectively).
        :type bounding:     dict
        :param smooth:      pass data through a Gaussian filter. Value of 1 is
                            essentially `off'.
                            Integer greater than zero is the degree of smoothing,
                            in grid spacing.
        :type smooth:       int
        :param dom:         domain for plotting (for WRF data).
                            If zero, the only netCDF file present will be
                            plotted.
        :type dom:          int
        :params plottype:   matplotlib command for plotting data
                            (contour or contourf).
        :type plottype:     str
        :param fig:         value of False will create new figure. A value of
                            matplotlib.figure object will plot data onto
                            this figure (similarly for axis, below).
        :type fig:          bool,matplotlib.figure
        :param ax:          matplotlib.axis object to plot onto
        :type ax:           bool,matplotlib.axis
        :param clvs:        contour levels for plotting.
                            Generate using numpy.arange.
                            False is automatic.
        :type clvs:         bool,numpy.ndarray
        :param cmap:        matplotlib.cmap name. Pick a nice one from
                            http://matplotlib.org/examples/color/
                            colormaps_reference.html
        :type cmap:         str
        :returns:           None.

        """
        # TODO: lats/lons False when no bounding, and selected with limited
        # domain.

        level = self.get_level_string(level)

        # Data
        self.W = self.get_netcdf(ncdir,ncf=ncf,nct=nct,dom=dom)
        # lats, lons = self.W.get_limited_domain(bounding)
        data = self.W.get(vrbl,utc,level,lons=False,lats=False)
        if smooth>1:
            data = stats.gauss_smooth(data,smooth)

        # Scales
        if clvs is False and cmap is False:
            S = Scales(vrbl,level)
            clvs = S.clvs
            cmap = S.cm
        elif clvs is False:
            S = Scales(vrbl,level)
            clvs = S.clvs
        elif cmap is False:
            S = Scales(vrbl,level)
            cmap = S.cm

        # Figure
        fname = self.create_fname(vrbl,utc,level)
        F = BirdsEye(self.W,fig=fig,ax=ax)
        # import pdb; pdb.set_trace()
        F.plot2D(data,fname,outdir,lats=False,lons=False,
                    plottype=plottype,smooth=smooth,
                    clvs=clvs,cmap=cmap)

    def create_fname(self,vrbl,utc,level,f_prefix=False,f_suffix=False):
        """
        Generate a filename (without extension) for saving a figure.
        Differentiate between similar plots for e.g. different domains by
        using the f_suffix/f_prefix options.

        :param vrbl:        variable name as found in WRF, or one of
                            the computed fields available in WEM.
        :type vrbl:         str
        :param utc:         one date/time. The tuple/list format is
                            YYYY,MM,DD,HH,MM,SS (ready for calendar.timegm).
                            Integer format is epoch/datenum (time.gmtime).
        :type utc:          tuple,list,int
        :param level:       required level.
        :type level:        str
        :param f_prefix:    custom filename prefix for output. Ignore if False.
        :type f_prefix:     bool,str
        :param f_suffix     custom filename suffix for output. Ignore if False.
        :type f_suffix      bool,str

        :returns:           str -- filename with extension.
        """
        time_str = utils.string_from_time('output',utc)

        fname = '_'.join((vrbl,time_str,level))

        if isinstance(f_prefix,basestring):
            fname = f_prefix + fname
        if isinstance(f_suffix,basestring):
            fname = fname + f_suffix
        return fname

    def get_netcdf(self,ncdir,ncf=False,nct=False,dom=1,path_only=False):
        """
        Returns the WRFOut, ECMWF, or RUC instance.

        :param ncdir:       absolute path to directory that contains the
                            netCDF file.

        :param ncf:         filename of netcdf data file if ambiguous within ncdir.
                            If no wrfout file is explicitly specified, the
                            netCDF file in that folder is chosen if unambiguous.
        :type ncf:          bool,str
        :param nct:         initialisation time of netcdf data file, if
                            ambiguous within ncdir.
        :type nct:          bool,str
        :param dom:         domain for plotting (for WRF data).
                            If zero, the only netCDF file present will be
                            plotted.
        :type dom:          int
        :param path_only:   if True, return only absolute path to file.
                            This is useful to loop over ensemble members and
                            generate a list of files.
        :type path_only:    bool
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

    def generate_times(self,itime,ftime,interval):
        """
        Wrapper for utility method
        :func:`WEM.utils.GIS_tools.generate_times`, so user can access
        this at the top level to loop over times.

        """
        listoftimes = utils.generate_times(itime,ftime,interval)
        return listoftimes

    def plot_diff_energy(self,vrbl,utc,energy,datadir,outdir,dataf=False,
                            outprefix=False,outsuffix=False,clvs=0,
                            title=False,fig=False,ax=False):
        """
        This function requires data already generated by
        :func:`WEM.postWRF.postWRF.stats.compute_diff_energy`.

        :param vrbl:    Vertically integrated ('sum_z') or summated over all
                        dimensions ('sum_xyz').
        :type vrbl:     str
        :param utc:     one date/time. The tuple/list format is
                        YYYY,MM,DD,HH,MM,SS (ready for calendar.timegm).
                        Integer format is epoch/datenum (ready for
                        time.gmtime).
        :type utc:      tuple,list,int
        :param energy:  DKE ('kinetic') or DTE ('total').
        :type energy:   str
        :param datadir: directory holding computed data
        :type datadir:  str
        :param outdir:  root directory for plots
        :type outdir:   str
        :param dataf:   file name of data file, if ambiguous
        :type dataf:    str
        :param f_prefix: custom filename prefix for output. Ignore if False.
        :type f_prefix: bool,str
        :param f_suffix: custom filename suffix for output. Ignore if False.
        :type f_suffix: bool,str
        :param clvs:    contour levels for plot. Generate with
                        :func:`numpy.arange`.
        :type clvs:     bool,N.ndarray
        :param title:   title for output
        :type title:    bool,str
        :param fig:         value of False will create new figure. A value of
                            matplotlib.figure object will plot data onto
                            this figure (similarly for axis, below).
        :type fig:          bool,matplotlib.figure
        :param ax:          matplotlib.axis object to plot onto
        :type ax:           bool,matplotlib.axis

        """
        DATA = utils.load_data(folder,fname,format='pickle')

        if isinstance(utc,(list,tuple)):
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

        fig_obj = F.plot_data(stack_average,'contourf',p2p,fname_t,utc,V,
                                **kwargs2)


    def delta_diff_energy(self,vrbl,utc0,utc1,energy,datadir,outdir,
                            meanvrbl='Z',meanlevel=500,
                            dataf=False,outprefix=False,outsuffix=False,
                            clvs=0,title=False,fig=False,ax=False,ncdata=False):
        """
        Plot DKE/DTE growth with time: delta DKE/DTE (DDKE/DDTE).
        Filled contours of DDKE or DDTE is optionally plotted over
        the ensemble mean of a variable (contours). DDKE/DDTE is valid
        halfway between the first and second times specified.

        :param vrbl:        Vertically integrated ('sum_z') or summated over all
                            dimensions ('sum_xyz').
        :type vrbl:         str
        :param utc0:        First date/time. The tuple/list format is
                            YYYY,MM,DD,HH,MM,SS (ready for calendar.timegm).
                            Integer format is epoch/datenum (ready for
                            time.gmtime).
        :type utc0:         tuple,list,int
        :param utc1:        as for `utc0`, but for the second time.
        :type utc1:         tuple,list,int
        :param energy:      DKE ('kinetic') or DTE ('total').
        :type energy:       str
        :param datadir:     directory holding computed data
        :type datadir:      str
        :param outdir:      root directory for plots
        :type outdir:       str
        :param dataf:       file name of data file, if ambiguous
        :type dataf:        str
        :param f_prefix:    custom filename prefix for output. Ignore if False.
        :type f_prefix:     bool,str
        :param f_suffix:    custom filename suffix for output. Ignore if False.
        :type f_suffix:     bool,str
        :param clvs:        contour levels for DDKE/DDTE. Generate with
                            :func:`numpy.arange`.
        :type clvs:         bool,N.ndarray
        :param meanclvs:    contour levels for ensemble mean. Generate with
                            :func:`numpy.arange`.
        :type meanclvs:     bool,N.ndarray
        :param title:       title for output
        :type title:        bool,str
        :param fig:         value of False will create new figure. A value of
                            matplotlib.figure object will plot data onto
                            this figure (similarly for axis, below).
        :type fig:          bool,matplotlib.figure
        :param ax:          matplotlib.axis object to plot onto
        :type ax:           bool,matplotlib.axis
        :param meanvrbl:    variable of ensemble mean (WRF key or computed vrbl)
        :type meanvrbl:     str
        :param meanlevel:   level for the ensemble mean variable. 
                            Lowest model level is integer 2000.
                            Pressure level is integer in hPa, e.g. 850.
                            Isentropic surface is a string + K, e.g. '320K'.
                            Geometric height is a string + m, e.g. '4000m'.
        :type meanlevel:    int,str
        :param ncdata:      if meanvrbl is not False, list of absolute
                            paths to all netcdf files of ensemble members

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

    def plot_error_growth(self,datadir,dataf=False,ensnames=False,
                            ylim=0,f_prefix=False,f_suffix=False):
        """Plots line graphs of DKE/DTE error growth
        varying by a sensitivity - e.g. error growth involving
        all members that use a certain parameterisation.

        Requires data file in pickle format already produced by
        [method here].

        :param datadir:     folder with pickle data
        :type datadir:      str
        :param dataf:       data (pickle) filename if ambiguous
        :type dataf:        str
        :param ensnames:    names of each ensemble member, e.g. the
                            parameterisation scheme, the initial
                            conditions used. This is used as the
                            label for the plot legend.
        :type ensnames:     list,tuple 
        :param ylim:        [min,max] for y axis range
        :type ylim:         list,tuple
        :param f_prefix:    custom filename prefix for output. Ignore if False.
        :type f_prefix:     bool,str
        :param f_suffix     custom filename suffix for output. Ignore if False.
        :type f_suffix      bool,str

        TODO: is sensitivity/ensnames variable OK to be optional?
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

    def composite_profile(self,vrbl,utc,enspaths,latlon=False,
                            dom=1,mean=True,std=True,xlim=False,
                            ylim=False):
        """
        Plot multiple vertical profiles of atmospheric variables 
        including optional mean and standard deviation.

        Superceded by :func:`twopanel_profile`?

        :param vrbl:        WRF variable or computed quantity
        :type vrbl:         str
        :param utc:         date/time
        :type utc:          int,list,tuple
        :param enspaths:    absolute paths to all netCDF files for each
                            ensemble member.
        :type enspaths:     list,tuple
        :param latlon:      (lat,lon) for plotting. If this is False,
                            a pop-up window will allow user to choose
                            location.
        :type latlon:       bool,tuple,list
        :param dom:         WRF domain to use
        :type dom:          int
        :param mean:        plot ensemble mean of variable
        :type mean:         bool
        :param std:         plot ensemble standard deviate of variable
                            (+/- 1 sigma)
        :type std:          bool
        :param xlim:        x-axis limit. False is automatic.
        :type xlim:         bool,tuple,list
        :param ylim:        y-axis limit, False is automatic.
        :type ylim:         bool,tuple,list

        """

        P = Profile(self.C)
        P.composite_profile(vrbl,utc,latlon,enspaths,dom,mean,std,xlim,ylim)

    def twopanel_profile(self,vrbl,utc,enspaths,outdir,two_panel=1,dom=1,
                            mean=1,std=1,xlim=False,ylim=False,latlon=False,
                            locname=False,overlay=False,ml=-2):
        """
        Create two-panel figure with profile location on map,
        with profile of all ensemble members in comparison.

        :param vrbl:        WRF variable or computed quantity
        :type vrbl:         str
        :param utc:         date/time
        :type utc:          int,list,tuple
        :param enspaths:    absolute paths to all netCDF files for each
                            ensemble member.
        :type enspaths:     list,tuple
        :param outdir:      directory for plot output
        :type outdir:       str
        :param two_panel:   Add extra panel of location if True.
        :type two_panel:    bool
        :param dom:         WRF domain to use
        :type dom:          int
        :param mean:        plot ensemble mean of variable
        :type mean:         bool
        :param std:         plot ensemble standard deviate of variable
                            (+/- 1 sigma)
        :type std:          bool
        :param xlim:        x-axis limit. False is automatic.
        :type xlim:         bool,tuple,list
        :param ylim:        y-axis limit, False is automatic.
        :type ylim:         bool,tuple,list
        :param latlon:      (lat,lon) for plotting. If this is False,
                            a pop-up window will allow user to choose
                            location.
        :type latlon:       bool,tuple,list
        :param locname:     this is passed to the filename of output
                            figure when saved, to differentiate similar
                            plots of different locations.
        :type locname:      str
        :param overlay:     data from the same time to overlay on inset
                            basemap. E.g. radar reflectivity.
                            TODO: this is only cref right now (bool)
        :type overlay:      str
        :param ml:          member level. Negative number that corresponds
                            to the folder in the absolute path string,
                            for naming purposes. Useful for file naming, 
                            labelling.
        :type ml:           int

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


    def plot_skewT(self,utc,ncdir,outdir,ncf=False,nct=False,f_prefix=False,
                    f_suffix=False, latlon=False,dom=1,save_output=0,
                    composite=0):
        """
        TODO: use Clicker instance if latlon is False.

        :param utc:         one date/time. The tuple/list format is
                            YYYY,MM,DD,HH,MM,SS (ready for calendar.timegm).
                            Integer format is epoch/datenum (ready for
                            time.gmtime).
        :type utc:          tuple,list,int
        :param ncdir:       directory of netcdf data file
        :type ncdir:        str
        :outdir:            directory to save output figures
        :type outdir:       str
        :param ncf:         filename of netcdf data file if ambiguous within ncdir.
                            If no wrfout file is explicitly specified, the
                            netCDF file in that folder is chosen if unambiguous.
        :type ncf:          bool,str
        :param nct:         initialisation time of netcdf data file, if
                            ambiguous within ncdir.
        :type nct:          bool,str
        :param f_prefix:    custom filename prefix for output. Ignore if False.
        :type f_prefix:     bool,str
        :param f_suffix     custom filename suffix for output. Ignore if False.
        :type f_suffix      bool,str
        :param latlon:      (lat,lon) for plotting. If False, choose from
                            pop-up.
        :type latlon:       bool,list,tuple
        :param dom:         WRF domain to plot from.
        :type dom:          int
        :param save_output: not sure why this here? TODO
        :type save_output:  bool
        :param composite:   If not False, plot numerous Skew-Ts on the same graph.
                            List is absolute paths to netCDF files.
        :type composite:    list,tuple,bool

        """
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

    def plot_streamlines(self,utc,level,ncdir,outdir,ncf=False,nct=False,
                            f_prefix=False,f_suffix=False,dom=1,smooth=1,
                            fig=False,ax=False,bounding=False):
        """
        Plot streamlines of wind at a level.

        :param utc:         one date/time. The tuple/list format is
                            YYYY,MM,DD,HH,MM,SS (ready for calendar.timegm).
                            Integer format is epoch/datenum (ready for
                            time.gmtime).
        :type utc:          tuple,list,int
        :param level:       required level. 
                            Lowest model level is integer 2000.
                            Pressure level is integer in hPa, e.g. 850.
                            Isentropic surface is a string + K, e.g. '320K'.
                            Geometric height is a string + m, e.g. '4000m'.
        :type level:        int,str
        :param ncdir:       directory of netcdf data file
        :type ncdir:        str
        :outdir:            directory to save output figures
        :type outdir:       str
        :param ncf:         filename of netcdf data file if ambiguous within ncdir.
                            If no wrfout file is explicitly specified, the
                            netCDF file in that folder is chosen if unambiguous.
        :type ncf:          bool,str
        :param nct:         initialisation time of netcdf data file, if
                            ambiguous within ncdir.
        :type nct:          bool,str
        :param f_prefix:    custom filename prefix for output. Ignore if False.
        :type f_prefix:     bool,str
        :param f_suffix     custom filename suffix for output. Ignore if False.
        :type f_suffix      bool,str
        :param bounding:    bounding box for domain.
                            Dictionary contains four keys (Nlim, Elim, Slim, Wlim)
                            with float values (northern latitude limit, eastern
                            longitude limit, southern latitude limit, western
                            latitude limit, respectively).
        :type bounding:     dict
        :param smooth:      pass data through a Gaussian filter. Value of 1 is
                            essentially `off'.
                            Integer greater than zero is the degree of smoothing,
                            in grid spacing.
        :type smooth:       int
        :param dom:         domain for plotting (for WRF data).
                            If zero, the only netCDF file present will be
                            plotted.
        :type dom:          int
        :param fig:         value of False will create new figure. A value of
                            matplotlib.figure object will plot data onto
                            this figure (similarly for axis, below).
        :type fig:          bool,matplotlib.figure
        :param ax:          matplotlib.axis object to plot onto
        :type ax:           bool,matplotlib.axis

        TODO: extra kwargs to account for arrow size, density, etc.
        """
        level = self.get_level_string(level)
        # Data
        self.W = self.get_netcdf(ncdir,ncf=ncf,nct=nct,dom=dom)
        lats, lons = self.W.get_limited_domain(bounding)

        if level=='2000hPa':
            U = self.W.get('U10',utc,level,lons,lats)[0,:,:]
            V = self.W.get('V10',utc,level,lons,lats)[0,:,:]
        else:
            U = self.W.get('U',utc,level,lons,lats)[0,0,:,:]
            V = self.W.get('V',utc,level,lons,lats)[0,0,:,:]

        self.F = BirdsEye(self.W,fig=fig,ax=ax)
        # disp_t = utils.string_from_time('title',utc)
        # print("Plotting {0} at lv {1} for time {2}.".format(
                # 'streamlines',lv,disp_t))
        fname = self.create_fname('streamlines',utc,level)
        self.F.plot_streamlines(U,V,outdir,fname)

    def plot_strongest_wind(self,itime,ftime,level,ncdir,outdir,
                            ncf=False,nct=False,
                            f_prefix=False,f_suffix=False,bounding=False,
                            dom=1):
        """
        Plot strongest wind at level between itime and ftime.

        :param itime:       initial time. The tuple/list format is
                            YYYY,MM,DD,HH,MM,SS (ready for calendar.timegm).
                            Integer format is epoch/datenum (ready for
                            time.gmtime).
        :type itime:        tuple,list,int
        :param ftime:       final time. Same format as `itime`.
        :type ftime:        tuple,list,int
        :param level:       required level. 
                            Lowest model level is integer 2000.
                            Pressure level is integer in hPa, e.g. 850.
                            Isentropic surface is a string + K, e.g. '320K'.
                            Geometric height is a string + m, e.g. '4000m'.
        :type level:        int,str
        :param ncdir:       directory of netcdf data file
        :type ncdir:        str
        :outdir:            directory to save output figures
        :type outdir:       str
        :param ncf:         filename of netcdf data file if ambiguous within ncdir.
                            If no wrfout file is explicitly specified, the
                            netCDF file in that folder is chosen if unambiguous.
        :type ncf:          bool,str
        :param nct:         initialisation time of netcdf data file, if
                            ambiguous within ncdir.
        :type nct:          bool,str
        :param f_prefix:    custom filename prefix for output. Ignore if False.
        :type f_prefix:     bool,str
        :param f_suffix     custom filename suffix for output. Ignore if False.
        :type f_suffix      bool,str
        :param bounding:    bounding box for domain.
                            Dictionary contains four keys (Nlim, Elim, Slim, Wlim)
                            with float values (northern latitude limit, eastern
                            longitude limit, southern latitude limit, western
                            latitude limit, respectively).
        :type bounding:     dict
        :param dom:         domain for plotting (for WRF data).
                            If zero, the only netCDF file present will be
                            plotted.
        :type dom:          int


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
        """ Ensure input data is a time series
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


    def plot_xs(self,vrbl,utc,ncdir,outdir,latA=0,lonA=0,latB=0,lonB=0,
                ncf=False,nct=False,f_prefix=0,f_suffix=0,dom=1,
                clvs=False,ylim=False):
        """
        Plot cross-section.

        If no lat/lon transect is indicated, a popup appears for the user
        to pick points. The popup can have an overlaid field such as reflectivity
        to help with the process.

        :param vrbl:        variable name as found in WRF, or one of
                            the computed fields available in WEM
        :type vrbl:         str
        :param utc:         one date/time. The tuple/list format is
                            YYYY,MM,DD,HH,MM,SS (ready for calendar.timegm).
                            Integer format is epoch/datenum (ready for
                            time.gmtime).
        :type utc:          tuple,list,int
        :param ncdir:       directory of netcdf data file
        :type ncdir:        str
        :param outdir:      directory to save output figures
        :type outdir:       str
        :param latA:        latitude of transect start point. 
                            False triggers a pop-up box.
        :type latA:         bool,float
        :param lonA:        longitude of transect start point.
                            False triggers a pop-up box.
        :type lonB:         bool,float                    
        :param latB:        latitude of transect end point. 
                            False triggers a pop-up box.
        :type latB:         bool,float
        :param lonB:        longitude of transect end point.
                            False triggers a pop-up box.
        :type lonB:         bool,float                    
        :param ncf:         filename of netcdf data file if ambiguous within ncdir.
                            If no wrfout file is explicitly specified, the
                            netCDF file in that folder is chosen if unambiguous.
        :type ncf:          bool,str
        :param nct:         initialisation time of netcdf data file, if
                            ambiguous within ncdir.
        :type nct:          bool,str
        :param f_prefix:    custom filename prefix for output. Ignore if False.
        :type f_prefix:     bool,str
        :param f_suffix     custom filename suffix for output. Ignore if False.
        :type f_suffix      bool,str
        :param dom:         domain for plotting (for WRF data).
                            If zero, the only netCDF file present will be
                            plotted.
        :type dom:          int
        :param clvs:        contouring for the variable plotted.
                            False is automatic.
                            Generate the `numpy.ndarray` with 
                            :func:`numpy.arange`.
        :type clvs:         numpy.ndarray,bool
        :param ylim:        [min,max] (in km) altitude to plot.
                            False is automatic (all model levels)
        :type ylim:         list,tuple,bool

        """
        self.W = self.get_netcdf(wrf_sd,wrf_nc,dom=dom)

        outpath = self.get_outpath(out_sd)

        XS = CrossSection(self.C,self.W,latA,lonA,latB,lonB)

        XS.plot_xs(vrbl,utc,outpath,clvs=clvs,ztop=ztop)


    def cold_pool_strength(self,utc,ncdir,outdir,ncf=False,nct=False,
                            f_prefix=False,f_prefix=False,
                            swath_width=100,bounding=False,dom=1,
                            twoplot=0,fig=0,ax=0,dz=0):
        """
        Pick A, B points on sim ref overlay
        This sets the angle between north and line AB
        Also sets the length in along-line direction
        For every gridpt along line AB:
            * Locate gust front via shear
            * Starting at front, do 3-grid-pt-average in line-normal
              direction

        :param utc:         one date/time. The tuple/list format is
                            YYYY,MM,DD,HH,MM,SS (ready for calendar.timegm).
                            Integer format is epoch/datenum (ready for
                            time.gmtime).
        :type utc:          tuple,list,int
        :param ncdir:       directory of netcdf data file
        :type ncdir:        str
        :param outdir:      directory to save output figures
        :type outdir:       str
        :param ncf:         filename of netcdf data file if ambiguous within ncdir.
                            If no wrfout file is explicitly specified, the
                            netCDF file in that folder is chosen if unambiguous.
        :type ncf:          bool,str
        :param nct:         initialisation time of netcdf data file, if
                            ambiguous within ncdir.
        :type nct:          bool,str
        :param f_prefix:    custom filename prefix for output. Ignore if False.
        :type f_prefix:     bool,str
        :param f_suffix:    custom filename suffix for output. Ignore if False.
        :type f_suffix:     bool,str
        :param swath_width: length in gridpoint in cross-section-normal
                            direction.
        :type swath_width:  int
        :param bounding:    bounding box for domain.
                            Dictionary contains four keys (Nlim, Elim, Slim, Wlim)
                            with float values (northern latitude limit, eastern
                            longitude limit, southern latitude limit, western
                            latitude limit, respectively).
        :type bounding:     dict
        :param dom:         domain for plotting (for WRF data).
                            If zero, the only netCDF file present will be
                            plotted.
        :type dom:          int
        :param twoplot:     If true, return two figures: cold pool strength
                            and the cref/cross-section
        :type twoplot:      bool
        :param dz:          plot height of cold pool only.
        :type dz:           bool
        :param fig:         value of False will create new figure. A value of
                            matplotlib.figure object will plot data onto
                            this figure (similarly for axis, below).
        :type fig:          bool,matplotlib.figure
        :param ax:          matplotlib.axis object to plot onto.
                            If tuple/list of length two, this is the
                            first and second axis, if twoplot is True.
        :type ax:           bool,matplotlib.axis

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

    def spaghetti(self,vrbl,utc,level,contour,ncdirs,outdir,
                    bounding=False,dom=1):
        """
        Do a multi-member spaghetti plot, contouring a value of
        a given variable.

        :param vrbl:        variable name as found in WRF, or one of
                            the computed fields available in WEM
        :type vrbl:         str
        :param utc:         one date/time. The tuple/list format is
                            YYYY,MM,DD,HH,MM,SS (ready for calendar.timegm).
                            Integer format is epoch/datenum (ready for
                            time.gmtime).
        :type utc:          tuple,list,int
        :param level:       required level. 
                            Lowest model level is integer 2000.
                            Pressure level is integer in hPa, e.g. 850.
                            Isentropic surface is a string + K, e.g. '320K'.
                            Geometric height is a string + m, e.g. '4000m'.
        :type level:        int,str
        :param contour:     contour to draw for data
        :type contour:      float,int
        :param ncdirs:      directories of netcdf data file
        :type ncdirs:       list,tuple
        :param outdir:      directory to save output figures
        :type outdir:       str
        :param bounding:    bounding box for domain.
                            Dictionary contains four keys (Nlim, Elim, Slim, Wlim)
                            with float values (northern latitude limit, eastern
                            longitude limit, southern latitude limit, western
                            latitude limit, respectively).
        :type bounding:     dict
        :param dom:         domain for plotting (for WRF data).
                            If zero, the only netCDF file present will be
                            plotted.
        :type dom:          int

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

    def std(self,vrbl,utc,level,ncdirs,outdir,ncf=False,nct=False,
            f_prefix=False,f_suffix=False,bounding=False,
            smooth=1,dom=1,plottype='contourf',fig=False,ax=False,
            clvs=False,cmap='jet'):
        """
        Plot standard deviation of all members for given variable.

        :param vrbl:        variable name as found in WRF, or one of
                            the computed fields available in WEM
        :type vrbl:         str
        :param utc:         one date/time. The tuple/list format is
                            YYYY,MM,DD,HH,MM,SS (ready for calendar.timegm).
                            Integer format is epoch/datenum (ready for
                            time.gmtime).
        :type utc:          tuple,list,int
        :param level:       required level. 
                            Lowest model level is integer 2000.
                            Pressure level is integer in hPa, e.g. 850.
                            Isentropic surface is a string + K, e.g. '320K'.
                            Geometric height is a string + m, e.g. '4000m'.
        :type level:        int,str
        :param ncdirs:      directories of netcdf data files for all
                            ensemble members. If not ambiguous, user needs
                            to specify either ncf (if all data
                            files have the same name) or nct (they need
                            to have the same start time).
        :type ncdir:        list,tuple
        :param outdir:      directory to save output figures
        :type outdir:       str
        :param ncf:         filename of netcdf data file if ambiguous within ncdir.
                            If no wrfout file is explicitly specified, the
                            netCDF file in that folder is chosen if unambiguous.
        :type ncf:          bool,str
        :param nct:         initialisation time of netcdf data file, if
                            ambiguous within ncdir.
        :type nct:          bool,str
        :param f_prefix:    custom filename prefix for output. Ignore if False.
        :type f_prefix:     bool,str
        :param f_suffix     custom filename suffix for output. Ignore if False.
        :type f_suffix      bool,str
        :param bounding:    bounding box for domain.
                            Dictionary contains four keys (Nlim, Elim, Slim, Wlim)
                            with float values (northern latitude limit, eastern
                            longitude limit, southern latitude limit, western
                            latitude limit, respectively).
        :type bounding:     dict
        :param smooth:      pass data through a Gaussian filter. Value of 1 is
                            essentially `off'.
                            Integer greater than zero is the degree of smoothing,
                            in grid spacing.
        :type smooth:       int
        :param dom:         domain for plotting (for WRF data).
                            If zero, the only netCDF file present will be
                            plotted.
        :type dom:          int
        :params plottype:   matplotlib command for plotting data
                            (contour or contourf).
        :type plottype:     str
        :param fig:         value of False will create new figure. A value of
                            matplotlib.figure object will plot data onto
                            this figure (similarly for axis, below).
        :type fig:          bool,matplotlib.figure
        :param ax:          matplotlib.axis object to plot onto
        :type ax:           bool,matplotlib.axis
        :param clvs:        contour levels for plotting.
                            Generate using numpy.arange.
                            False is automatic.
        :type clvs:         bool,numpy.ndarray
        :param cmap:        matplotlib.cmap name. Pick a nice one from
                            http://matplotlib.org/examples/color/
                            colormaps_reference.html
        :type cmap:         str

        """
        if wrfdirs is False and wrfpaths is False:
            print("Must have either wrfdirs or wrfpaths.")
            raise Exception
        elif isinstance(wrfdirs,(tuple,list)):
            ncfiles = self.list_ncfiles(wrf_sds)
        elif isinstance(wrfpaths,(tuple,list)):
            ncfiles = wrfpaths

        # Use first wrfout to initialise grid, get indices
        self.W = self.get_netcdf(ncfiles[0],dom=dom)

        tidx = self.W.get_time_idx(utc)

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

    def list_ncfiles(self,ncdirs,nct=False,ncf=False,dom=0,path_only=1):
        """
        Create list of absolute paths or objects* to netCDF files, given the
        list/tuple of directories they are in. If ambiguous, user
        needs to specify domain, and either the filename (if
        all identical) or initialisation time of the run (this
        needs to be the same for all files).

        Note * that the object will be a WRFOut instance if netCDF is
        a wrfout file, RUC instance for RUC, or EC instance for ECMWF.

        :param ncdirs:      absolute paths to directories containing
                            netCDF files.
        :type ncdirs:       tuple,list
        :param ncf:         filename of netcdf data file if ambiguous within ncdir.
                            If no wrfout file is explicitly specified, the
                            netCDF file in that folder is chosen if unambiguous.
        :type ncf:          bool,str
        :param nct:         initialisation time of netcdf data file, if
                            ambiguous within ncdir.
        :type nct:          bool,str
        :param dom:         domain for plotting (for WRF data).
                            If zero, the only netCDF file present will be
                            plotted.
        :type dom:          int
        :param path_only:   if True, return only strings to the files.
                            if False, return the instances (RUC,EC,WRFOut).
        :returns:           either a list of absolute path strings, or 
                            a list of instances (RUC,EC,WRFOut).

        TODO: Deal with ambiguous selections.    
        """
        ncfiles = []
        for wrf_sd in wrf_sds:
            ncfile = self.get_netcdf(wrf_sd,dom=dom,path_only=path_only)
            ncfiles.append(ncfile)
        return ncfiles

    def plot_domains(self,ncdirs,labels,outdir,colours='black',
                        nct=False,ncf=False,latlons=False):
        """
        Plot only the domains for each netCDF file specified.

        :param ncdirs:      Absolute paths to all netCDF directories, 
                            or one single absolute path if only 
                            one domain is to be plotted.
        :type ncdirs:       str,tuple,list
        :param labels:      labels for each domain. 
        :type labels:       str,tuple,list
        :param outdir:      directory to save output figures
        :type outdir:       str
        :param colours:     colours for each domain box, in the same order as 
                            the ncdirs sequence (if more than one)
        :type colours:      str,list,tuple 
        :param ncf:         filename of netcdf data file if ambiguous within ncdir.
                            If no wrfout file is explicitly specified, the
                            netCDF file in that folder is chosen if unambiguous.
        :type ncf:          bool,str
        :param nct:         initialisation time of netcdf data file, if
                            ambiguous within ncdir.
        :type nct:          bool,str
        :param latlons:     (lat,lon) of each label for each domain, in the same
                            order as the ncdirs sequence (if more than one).
        :type latlons:      tuple,list

        """
        outpath = self.get_outpath(out_sd)
        maps.plot_domains(wrfouts,labels,latlons,outpath,colour)


    def frontogenesis(self,utc,level,ncdir,outdir,ncf=False,nct=False,
                        dom=1,smooth=0,clvs=0,title=0):
        """
        Compute and plot Miller frontogenesis as d/dt of theta gradient.

        Use a centred-in-time derivative; hence, if
        time index is start or end of wrfout file, skip the plot.

        :param utc:         one date/time. The tuple/list format is
                            YYYY,MM,DD,HH,MM,SS (ready for calendar.timegm).
                            Integer format is epoch/datenum (ready for
                            time.gmtime).
        :type utc:          tuple,list,int
        :param level:       required level. 
                            Lowest model level is integer 2000.
                            Pressure level is integer in hPa, e.g. 850.
                            Isentropic surface is a string + K, e.g. '320K'.
                            Geometric height is a string + m, e.g. '4000m'.
        :type level:        int,str
        :param ncdir:       directory of netcdf data file
        :type ncdir:        str
        :param outdir:      directory to save output figures
        :type outdir:       str
        :param ncf:         filename of netcdf data file if ambiguous within ncdir.
                            If no wrfout file is explicitly specified, the
                            netCDF file in that folder is chosen if unambiguous.
        :type ncf:          bool,str
        :param nct:         initialisation time of netcdf data file, if
                            ambiguous within ncdir.
        :type nct:          bool,str
        :param f_prefix:    custom filename prefix for output. Ignore if False.
        :type f_prefix:     bool,str
        :param f_suffix     custom filename suffix for output. Ignore if False.
        :type f_suffix      bool,str
        :param bounding:    bounding box for domain.
                            Dictionary contains four keys (Nlim, Elim, Slim, Wlim)
                            with float values (northern latitude limit, eastern
                            longitude limit, southern latitude limit, western
                            latitude limit, respectively).
        :type bounding:     dict
        :param smooth:      pass data through a Gaussian filter. Value of 1 is
                            essentially `off'.
                            Integer greater than zero is the degree of smoothing,
                            in grid spacing.
        :type smooth:       int
        :param dom:         domain for plotting (for WRF data).
                            If zero, the only netCDF file present will be
                            plotted.
        :type dom:          int
        :param fig:         value of False will create new figure. A value of
                            matplotlib.figure object will plot data onto
                            this figure (similarly for axis, below).
        :type fig:          bool,matplotlib.figure
        :param ax:          matplotlib.axis object to plot onto
        :type ax:           bool,matplotlib.axis
        :param clvs:        contour levels for plotting.
                            Generate using numpy.arange.
                            False is automatic.
        :type clvs:         bool,numpy.ndarray
        :param cmap:        matplotlib.cmap name. Pick a nice one from
                            http://matplotlib.org/examples/color/
                            colormaps_reference.html
        :type cmap:         str

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

    def get_level_string(self,level):
        """
        Makes sure user's level input is a string.
        Saves typing hPa for the common usage of pressure levels.

        :param level:   desired level.
        :type level:    str,int
        :returns:       str
        """
        if isinstance(level,int):
            # Level is in pressure
            level = '{0}hPa'.format(level)
        return level
