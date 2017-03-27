""" This handles user requests and controls computations and plotting.

This script is API and should not be doing any hard work of
importing matplotlib etc!

Useful/utility scripts are in WEM.utils.

TODO: move all DKE stuff to stats and figure/birdseye.

TODO: move more utilities to utils.

TODO: generalise plotting so the mountain of arguments can be easily changed
and documented. Right now a small change to general plotting API takes a lot
of time, and the documentation is a bit crazy.

TODO: make boring methods private, e.g. _create_fname(). Then hide it from
documentation if it is revealed to the user at the top level.
"""
from netCDF4 import Dataset
import calendar
import collections
import copy
import pickle as pickle
import fnmatch
import glob
import itertools
import numpy as N
import os
import pdb
import time
import matplotlib as M
# M.use('gtkagg')
M.use('agg')
import matplotlib.pyplot as plt

from .wrfout import WRFOut
from .figure import Figure
from .birdseye import BirdsEye
from .ruc import RUC
from .skewt import SkewT
from .skewt import Profile
#import scales
from .defaults import Defaults
import WEM.utils as utils
from .xsection import CrossSection
from .clicker import Clicker
from . import maps
from . import stats
from .scales import Scales
from .obs import Obs
from .obs import Radar
from .ts import TimeSeries
from .ensemble import Ensemble

# TODO: Make this awesome

class WRFEnviron(object):
    def __init__(self,rootdir=None,initutc=None,doms=1,ctrl='ctrl',aux=False,
                    model='wrf',fmt='em_real',f_prefix=None,
                            output_t=False,history_sec=None,
                            ncf=False):
        """ Sets up the environment for main methods and scripts.

        Args:
            rootdir (str):  Directory at root of datafiles.
            initutc (datetime.datetime): Initialization time.

            doms (int, optional): Number of domains
            ctrl (bool, optional): Whether ensemble has control member
            aux (bool, dict, optional): Dictionary lists, per domain, data
                files that contain additional variables and each
                file's prefix. Default is False (no auxiliary files).
                Not implemented yet.
            enstype (str, optional): Type of ensemble. Default is for
                Weather Research and Forecast (WRF) model.
            fmt (str, optional): The type of simulations. Default is
                real-world simulations (em_real from WRF).
            f_prefix (tuple, optional): Tuple of prefixes for each
                ensemble member's main data files. Must be length /doms/.
                Default is None, which then uses a method to determine
                the file name using default outputs from e.g. WRF.
            output_t (list, optional): Future extension that allows multiple data
                files for each time.
            history_sec (int, optional): Difference between history output
                intervals. If all data is contained in one file, use None
                (default). Methods will keep looking for history output
                files until a file error is raised.
        """
        self.fmt = fmt # Used to be self.em

        if rootdir is not None:
            self.ensemble = Ensemble(rootdir=rootdir,initutc=initutc,doms=doms,
                                    ctrl=ctrl,aux=aux,model=model,fmt=fmt,
                                    f_prefix=f_prefix,loadobj=False,
                                    ncf=ncf)

            # If true, WRFOuts exist in ensemble.
            # If not, they need loading when needed
            self.loadobj = self.ensemble.loadobj

    def plot2D(self,vrbl,utc=0,member='ctrl',level=None,outdir=False,
                f_prefix=False,f_suffix=False,dom=1,plottype='contourf',
                smooth=1,fig=False,ax=False,clvs=False,cmap=False,
                locations=False,cb=True,match_nc=False,Nlim=False,Elim=False,
                Slim=False,Wlim=False,color='k',inline=False,lw=False,
                extend=False,save=True,accum_hr=False,cblabel=False,
                data=None,fname=False,ideal=False, return_figax=False,
                cont2_data=False,cont2_clvs=False,
                cont2_lats=False,cont2_lons=False,
                drawcounties=False,other=False,return_basemap=False):
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
                            Can be False if variable has no level.
                            Lowest model level is integer 2000.
                            Pressure level is integer in hPa, e.g. 850.
                            Isentropic surface is a string + K, e.g. '320K'.
                            Geometric height is a string + m, e.g. '4000m'.
        :type level:        int,str,bool
        :param ncdir:       directory of netcdf data file.
                            False uses home directory.
        :type ncdir:        str,bool
        :param outdir:      directory to save output figures
                            False uses home directory.
        :type outdir:       str,bool
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
        :param locations:   Locations to plot on the basemap.
                            Format: locations = {'label':(lat,lon),etc}
        :type locations:    dict
        :param cb:          plot a colorbar.
        :type cb:           bool
        :param match_nc:    Use domain from other netCDF file.
                            Absolute path to this netCDF file.
        :type match_nc:     str
        :param Nlim:        north limit (latitude) for plot
        :type Nlim:         float
        :param Elim:        east limit (longitude) for plot
        :type Elim:         float
        :param Slim:        south limit (latitude) for plot
        :type Slim:         float
        :param Wlim:        west limit (longitude) for plot
        :type Wlim:         float
        :returns:           None.

        """
        # TODO: lats/lons False when no bounding, and selected with limited
        # domain.
        # TODO: return_basemap and return_cb merged into same argument (and
        # do this in BirdsEye()

        if self.fmt is not 'em_real':
            ideal = True
        if outdir is False:
            outdir = os.path.expanduser("~")

        if level:
            level = self.get_level_string(level)

        # Match domain
        if isinstance(match_nc,str):
            MATCH = WRFOut(match_nc,fmt=self.fmt)
            if not Nlim:
                Nlim, Elim, Slim, Wlim = MATCH.get_limits()

        if data is None:
            # Data
            W = self.get_dataobj(dom=dom,utc=utc,member=member)
            # lats, lons = self.W.get_limited_domain(bounding)
            # import pdb; pdb.set_trace()
            if vrbl == 'accum_precip':
                if not accum_hr:
                    raise Exception("Set accumulation period")
                data = W.compute_accum_rain(utc,accum_hr)[0,0,:,:]
            else:
                data = W.get(vrbl,utc=utc,level=level,lons=None,lats=None,
                                other=other)[0,0,:,:]
        else:
            W = MATCH

        # Needs to be shape [1,1,nlats,nlons].
        if smooth>1:
            data = stats.gauss_smooth(data,smooth)

        if isinstance(Nlim,float):
            data,lats,lons = utils.return_subdomain(data,W.lats1D,W.lons1D,
                                Nlim,Elim,Slim,Wlim,fmt='latlon')
        else:
            lats = False
            lons = False

        # Scales for plotting
        cmap, clvs = self.get_cmap_clvs(vrbl,level,cmap=cmap,clvs=clvs)

        # Figure
        if fname is False:
            fname = self.create_fname(vrbl,utc,level,f_suffix=f_suffix,
                                    f_prefix=f_prefix,other=other,)
        F = BirdsEye(W,fig=fig,ax=ax)
        if cont2_data is not False:
            save = False
        if (ax is not False) and (fig is not False):
            save = False
        rets = F.plot2D(data,fname,outdir,lats=lats,lons=lons,
                    plottype=plottype,smooth=smooth,
                    clvs=clvs,cmap=cmap,locations=locations,
                    cb=cb,color=color,inline=inline,lw=lw,
                    extend=extend,save=save,cblabel=cblabel,
                    ideal=ideal,alpha=0.8,return_basemap=return_basemap,
                    drawcounties=drawcounties,)
        if cont2_data is not False:
            F.plot2D(cont2_data,fname,outdir,lats=cont2_lats,
                    lons=cont2_lons,cb=False,
                    plottype='contour',smooth=smooth,
                    clvs=cont2_clvs,locations=locations,
                    color=color,inline=inline,lw=lw,
                    save=True,ideal=ideal,m=m,)

        if return_figax:
            return (F.fig, F.ax)
        else:
            return rets

    def get_dataobj(self,utc=0,dom=1,member='ctrl'):
        # t = utc
        # dataobj = self.ensemble.members[member][dom][t]['dataobj']
        # if not dataobj:
            # fpath = self.ensemble.members[member][dom][t]['fpath']
        # dataobj = self.ensemble.datafile_object(self,fpath,loadobj=True)
        dataobj = self.ensemble.return_DF_for_t(utc,member,dom=1)
            # dataobj = WRFOut(fpath)
        return dataobj

    def get_cmap_clvs(self,vrbl,level,clvs=False,cmap=False):
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
        return cmap,clvs

    def create_fname(self,vrbl,utc=None,level=False,other=False,
                        f_prefix=False,f_suffix=False,
                        extension='png'):
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
        strs = [vrbl,]

        if level:
            strs.append('{0}'.format(level))

        if utc is not None:
            if isinstance(utc,int) and utc<500:
                strs.append('t{0:03d}'.format(utc))
            else:
                time_str = utils.string_from_time('output',utc)
                strs.append(time_str)

        if isinstance(other,dict):
            for k,v in other.items():
                strs.append(str(v))

        # import pdb; pdb.set_trace()

        fname = '_'.join(strs)

        if isinstance(f_prefix,str):
            fname = '_'.join((f_prefix,fname))
        if isinstance(f_suffix,str):
            fname = '_'.join((fname,f_suffix))

        if isinstance(extension,str):
            fname = '.'.join((fname,extension))

        return fname

    def generate_times(self,itime,ftime,interval):
        """
        Wrapper for utility method
        :func:`WEM.utils.GIS_tools.generate_times`, so user can access
        this at the top level to loop over times.

        """
        listoftimes = utils.generate_times(itime,ftime,interval)
        return listoftimes

    def compute_diff_energy(self,*args,**kwargs):
        stats.compute_diff_energy(*args,**kwargs)
        return

    def plot_diff_energy(self,vrbl,energy,datadir,outdir,utc=False,dataf=False,
                            outprefix=False,outsuffix=False,clvs=False,
                            title=False,fig=False,ax=False,cb=False):
        """
        This function requires data already generated by
        :func:`WEM.postWRF.postWRF.stats.compute_diff_energy`.

        :param vrbl:    Vertically integrated ('2D') or summated over all
                        dimensions ('3D').
        :type vrbl:     str
        :param utc:     one date/time. The tuple/list format is
                        YYYY,MM,DD,HH,MM,SS (ready for calendar.timegm).
                        Integer format is epoch/datenum (ready for
                        time.gmtime).
                        If false, loop over all times
        :type utc:      tuple,list,int
        :param energy:  DKE or DTE.
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
        DATA = utils.load_data(datadir,dataf,format='pickle')

        if isinstance(utc,(list,tuple)):
            utc = calendar.timegm(utc)

        if utc==False:
            for perm in DATA:
                looptimes = DATA[perm]['times']
                f1 = DATA[perm]['file1']
                try:
                    W1 = WRFOut(f1)
                except:
                    # From a bug that added an erroneous dir
                    # for STCH members in 2013, maybe fixed?
                    ff = f1.split('/')
                    del ff[-2]
                    W1 = WRFOut('/'.join(ff))
                break
        else:
            looptimes = (utc,)

        for t in looptimes:
            for pn,perm in enumerate(DATA):
                # f2 = DATA[perm]['file2']
                # Get times and info about nc files
                # First time to save power

                permtimes = DATA[perm]['times']

                # Find array for required time
                x = N.where(N.array(permtimes)==t)[0][0]
                data = DATA[perm]['values'][x][0]
                if not pn:
                    stack = data
                else:
                    stack = N.dstack((data,stack))
                    stack_average = N.average(stack,axis=2)

            #birdseye plot with basemap of DKE/DTE
            F = BirdsEye(W1,ax=ax,fig=fig)
            # tstr = utils.string_from_time('output',t)
            # fname = ''.join((plotname,'_{0}'.format(tstr)) )
            vrbl_long = "{0}_{1}".format(energy,vrbl)
            fname = self.create_fname(vrbl_long,utc=t,f_prefix=outprefix,f_suffix=outsuffix)
            F.plot2D(stack_average,fname,outdir,clvs=clvs,title=title,cb=cb)

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
            print(('Computing for time {0}'.format(time.gmtime(delt))))
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

    def plot_error_growth(self,outdir,datadir,dataf=False,sensitivity=False,
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
        ofname = '2D'
        DATA = utils.load_data(datadir,dataf,format='pickle')

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
                if ylim:
                    plt.ylim(ylim)
                plt.gca().set_xticks(times[::2])
                plt.gca().set_xticklabels(time_str[::2])

                vrbl_long = '{0}_{1}'.format(ofname,sens)
                fname = self.create_fname(vrbl_long,f_prefix=f_prefix,f_suffix=f_suffix)
                fpath = os.path.join(outdir,fname)
                fig.savefig(fpath)

                plt.close()
                print(("Saved type #1 {0}.".format(fpath)))

            # Averages for each sensitivity
            labels = []
            fig = plt.figure()
            ave_of_ave_stack = 0
            for sens in list(AVE.keys()):
                plt.plot(times,AVE[sens])
                labels.append(sens)
                ave_of_ave_stack = utils.vstack_loop(AVE[sens],ave_of_ave_stack)

            labels.append('Average')
            ave_of_ave = N.average(ave_of_ave_stack,axis=0)
            plt.plot(times,ave_of_ave,'k')

            plt.legend(labels,loc=2,fontsize=9)

            if ylim:
                plt.ylim(ylim)
            plt.gca().set_xticks(times[::2])
            plt.gca().set_xticklabels(time_str[::2])
            vrbl_long = '{0}_Averages'.format(ofname,)
            fname = self.create_fname(vrbl_long,f_prefix=f_prefix,f_suffix=f_suffix)
            fpath = os.path.join(outdir,fname)
            fig.savefig(fpath)

            plt.close()
            print(("Saved type #2 {0}.".format(fpath)))
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

            if ylim:
                plt.ylim(ylim)
            plt.gca().set_xticks(times[::2])
            plt.gca().set_xticklabels(time_str[::2])
            allstr = 'allmembers'
            vrbl_long = '{0}_{1}'.format(ofname,allstr)
            fname = self.create_fname(vrbl_long,f_prefix=f_prefix,f_suffix=f_suffix)
            fpath = os.path.join(outdir,fname)
            fig.savefig(fpath)

            plt.close()
            print(("Saved type #3 {0}.".format(fpath)))

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
            print(("Pick location for {0}".format(t_long)))
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


    def plot_skewT(self,utc,ncdir=False,outdir=False,ncf=False,nct=False,f_prefix=False,
                    f_suffix=False, latlon=False,dom=1,save_output=False,
                    composite=0,ax=False,fig=False):
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
        W = self.get_netcdf(ncdir,dom=dom)

        if not composite:
            ST = SkewT(W)
            ST.plot_skewT(utc,latlon,dom,outdir,save_output=save_output)
            nice_time = utils.string_from_time('title',utc)
            print(("Plotted Skew-T for time {0} at {1}".format(
                        nice_time,latlon)))
        else:
            #ST = SkewT(self.C)
            pass

    def plot_streamlines(self,utc,level,ncdir,outdir,ncf=False,nct=False,
                            f_prefix=False,f_suffix=False,dom=1,smooth=1,
                            fig=False,ax=False,bounding=False,density=1.8,
                            ideal=False):
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
        # lats, lons = self.W.get_limited_domain(bounding)
        lons = False
        lats = False

        if level=='2000hPa':
            U = self.W.get('U10',utc,level,lons,lats)[0,0,:,:]
            V = self.W.get('V10',utc,level,lons,lats)[0,0,:,:]
        else:
            U = self.W.get('U',utc,level,lons,lats)[0,0,:,:]
            V = self.W.get('V',utc,level,lons,lats)[0,0,:,:]

        if isinstance(bounding,dict):
            U,lats,lons = utils.return_subdomain(U,self.W.lats1D,self.W.lons1D,
                                bounding['Nlim'],bounding['Elim'],
                                bounding['Slim'],bounding['Wlim'],fmt='latlon')
            V,lats,lons = utils.return_subdomain(V,self.W.lats1D,self.W.lons1D,
                                bounding['Nlim'],bounding['Elim'],
                                bounding['Slim'],bounding['Wlim'],fmt='latlon')
        # else:
            # lats = False
            # lons = False
        self.F = BirdsEye(self.W,fig=fig,ax=ax)
        # disp_t = utils.string_from_time('title',utc)
        # print("Plotting {0} at lv {1} for time {2}.".format(
                # 'streamlines',lv,disp_t))
        fname = self.create_fname('streamlines',utc,level)
        # import pdb; pdb.set_trace()
        self.F.plot_streamlines(U,V,outdir,fname,density=density,lats=lats,lons=lons,
                                ideal=ideal)

    def plot_strongest_wind(self,itime,ftime,level=2000,ncdir=False,
                            outdir=False,ncf=False,nct=False,
                            f_prefix=False,f_suffix=False,bounding=False,
                            dom=1,clvs=False,fig=False,ax=False,cb=True,
                            extend='max',cmap='jet',save=True):
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
        if ncdir is False:
            ncdir = os.path.expanduser("~")
        if outdir is False:
            outdir = os.path.expanduser("~")

        if level:
            level = self.get_level_string(level)

        # Data
        self.W = self.get_netcdf(ncdir,ncf=ncf,nct=nct,dom=dom)

        # Make sure times are in datenum format and sequence.
        it = utils.ensure_datenum(itime)
        ft = utils.ensure_datenum(ftime)
        trange = self.W.return_tidx_range(it,ft)
        deltahr = str(int((ft-it)/3600.0))

        # import pdb; pdb.set_trace()
        data = self.W.get('strongestwind',utc=trange,level=level)[0,0,:,:]

        F = BirdsEye(self.W,fig=fig,ax=ax)
        fname = self.create_fname('strongestwind',ftime,level=level,f_suffix='_'+deltahr)
        xx = F.plot2D(data,fname,outdir,clvs=clvs,cb=cb,cmap=cmap,
                extend=extend,save=save)
        return xx

    def plot_probs(self,vrbl,overunder,threshold,itime,ftime=None,smooth=False,
                    level=None,outdir=False,fname=False,dom=1,
                    clvs=False,fig=False,ax=False,cb=True,accum_hr=False,
                    Nlim=False,Elim=False,Slim=False,Wlim=False,
                    return_figax=False,verif=False):
        """Docs.
        """
        pc_arr = self.ensemble.get_prob_threshold(vrbl,overunder,threshold,
                    itime=itime,level=level,Nlim=Nlim,Elim=Elim,
                    Slim=Slim,Wlim=Wlim,dom=dom,ftime=ftime)
        if verif is not False:
            obsdata = verif.return_array(ftime,accum_hr=accum_hr)

        ff = self.plot2D('probs',data=pc_arr,outdir=outdir,fname=fname,
                        match_nc=self.ensemble.arbitrary_pick(give_path=True),
                        return_figax=return_figax,cont2_data=obsdata,
                        cont2_clvs=[threshold,],cont2_lats=verif.lats,
                        cont2_lons=verif.lons)

        return ff

    def probability_threshold(self,vrbl,overunder,threshold,itime,ftime,smooth=False,
                            level=2000, outdir=False,f_prefix=False,f_suffix=False,bounding=False,
                            dom=1,clvs=False,fig=False,ax=False,cb=True,accum_hr=False):
        """
        Create threshold contour plots.
        """
        enssize = len(ensemble)
        self.ensemble = ensemble # Dictionary
        nens = 0
        for ens in self.ensemble:
            print(("Computing for ensemble member {0}.".format(ens)))
            # pdb.set_trace()
            if self.ensemble[ens]['control']:
                continue

            nens += 1.0

            self.ensemble[ens]['data'] = WRFOut(self.ensemble[ens]['path'])
            if nens==1:
                examplewrf = self.ensemble[ens]['data']
            tidx = self.ensemble[ens]['data'].return_tidx_range(itime,ftime)
            ens_data = self.ensemble[ens]['data'].get(vrbl,utc=tidx,level=level,
                                                lons=False,lats=False)
            for n in range(ens_data.shape[0]):
                if isinstance(smooth,str):
                    if smooth == 'maxfilter':
                        ens_data[n,0,:,:] = stats.max_filter(ens_data[n,0,:,:],size=11)
            if nens == 1:
                w,x,y,z = ens_data.shape
                all_ens_data = N.zeros((enssize,w,x,y,z))
                del w,x,y,z
            all_ens_data[nens-1,...] = ens_data

        if overunder == 'over':
            # True/False if member meets condition
            bool_arr = N.where(all_ens_data > threshold,1,0)
            # Find maximum for all times
            max_arr = N.amax(bool_arr,axis=1)
            # Count members that exceed the threshold
            # And convert to percentage
            count_arr = N.sum(max_arr,axis=0)
            percent_arr = 100*(count_arr/nens)

        elif overunder == 'under':
            # True/False if member meets condition
            bool_arr = N.where(all_ens_data < threshold,1,0)
            # Find maximum for all times
            max_arr = N.amin(bool_arr,axis=1)
            # Count members that exceed the threshold
            # And convert to percentage
            count_arr = N.sum(max_arr,axis=0)
            percent_arr = 100*(count_arr/nens)

        else:
            raise Exception("Pick over or under for threshold comparison.")

        output = percent_arr[0,:,:]

        # fname = self.create_fname(vrbl,utc,level,f_suffix=f_suffix, f_prefix=f_prefix,other=other)
        F = BirdsEye(examplewrf,fig=fig,ax=ax)
        plottype = 'contourf'
        clvs = N.arange(10,110,10)
        color = 'k'
        inline = False
        fname = 'probability_{0}_{1}.png'.format(vrbl,threshold)
        F.plot2D(output,fname,outdir,plottype=plottype,smooth=smooth,
                    clvs=clvs,color=color,inline=inline)
        # import pdb; pdb.set_trace()

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
                clvs=False,ylim=False,ztop=8,cmap='jet',
                contour_vrbl='skip',contour_clvs=False,
                avepts=False,shiftpts=False,
                cftix=False,cflabel=False):
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
        self.W = self.get_netcdf(ncdir,ncf=ncf,nct=nct,dom=dom)

        XS = CrossSection(self.W,latA,lonA,latB,lonB)
        if avepts:
            XS.plot_average(vrbl,avepts,utc,outdir,clvs=clvs,ztop=ztop,
                    f_suffix=f_suffix,cmap=cmap,contour_vrbl=contour_vrbl,
                    contour_clvs=contour_clvs,cflabel=cflabel,cftix=cftix)
        else:
            XS.plot_xs(vrbl,utc,outdir,clvs=clvs,ztop=ztop,f_suffix=f_suffix,cmap=cmap,
                contour_vrbl=contour_vrbl,contour_clvs=contour_clvs)

        if shiftpts is not False:
            if f_suffix is False:
                f_suffix = ''
            for shn in range(-shiftpts,shiftpts+1):
                if shn:
                    XS.translate_xs(shn)
                    super(CrossSection,XS).__init__(self.W)
                    fsnew = f_suffix + '_SHIFT{0}'.format(shn)
                    XS.plot_xs(vrbl,utc,outdir,clvs=clvs,ztop=ztop,
                            f_suffix=fsnew,cmap=cmap,contour_vrbl=contour_vrbl,
                            contour_clvs=contour_clvs)
                else:
                    XS.translate_xs(1)


    def cold_pool_strength(self,utc,ncdir=False,outdir=False,ncf=False,nct=False,
                            f_prefix=False,f_suffix=False,
                            swath_width=100,bounding=False,dom=1,
                            twoplot=False,fig=0,axes=0,dz=0):
        """
        Pick A, B points on sim ref overlay
        This sets the angle between north and line AB
        Also sets the length in along-line direction
        For every gridpt along line AB:
            * Locate gust front via shear
            * Starting at front, do 3-grid-pt-average in line-normal
              direction

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
        self.W = self.get_netcdf(ncdir,ncf=ncf,nct=nct,dom=dom)

        # keyword arguments for plots
        line_kwargs = {}
        cps_kwargs = {}

        # Create two-panel figure
        if twoplot:
            P2 = Figure(self.W,plotn=(1,2))
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
        # F = BirdsEye(self.W)

        cref_data = self.W.get('cref',utc=utc,level=False,lons=False,lats=False)[0,0,:,:]
        # self.data = F.plot2D('cref',utc,2000,dom,outpath,save=False,return_data=1)
        cmap, clvs = self.get_cmap_clvs('cref',level=False)
        # import pdb; pdb.set_trace()
        C = Clicker(self.W,data=cref_data,cmap=cmap,clvs=clvs,**line_kwargs)
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
        X = CrossSection(self.W,lat0,lon0,lat1,lon1)

        # Ask user the line-normal box width (self.km)
        #C.set_box_width(X)

        # Compute the grid (DX x DY)
        cps = self.W.cold_pool_strength(X,utc,swath_width=swath_width,env=(x_env,y_env),dz=dz)
        # import pdb; pdb.set_trace()

        # Plot this array
        CPfig = BirdsEye(self.W,**cps_kwargs)
        tstr = utils.string_from_time('output',utc)
        if dz:
            fprefix = 'ColdPoolDepth_'
        else:
            fprefix = 'ColdPoolStrength_'
        fname = fprefix + tstr

        # pdb.set_trace()
        # imfig,imax = plt.subplots(1)
        # imax.imshow(cps)
        # plt.show(imfig)
        # CPfig.plot_data(cps,'contourf',outpath,fname,time,V=N.arange(5,105,5))
        mplcommand = 'contourf'
        plotkwargs = {'cb':False}
        if dz:
            clvs = N.arange(100,5100,100)
        else:
            clvs = N.arange(10,85,2.5)
        if mplcommand[:7] == 'contour':
            plotkwargs['clvs'] = clvs
            # plotkwargs['cmap'] = 'ocean_r'
            plotkwargs['cmap'] = 'jet'
            plotkwargs['color'] = None

        cf2 = CPfig.plot2D(cps,fname,outdir,plottype=mplcommand,**plotkwargs)
        # CPfig.fig.tight_layout()

        plt.close(fig)

        if twoplot:
            P2.save(outdir,fname+"_twopanel")

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
        print(("Plotting std dev for {0} at time {1}".format(va,t_name)))

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

    def plot_domains(self,ncdirs,labels,outdir,fname,Nlim,Elim,
                        Slim,Wlim,colours='black',
                        nct=False,ncf=False,fill_land=False,
                        labpos=False,fill_water=False,):
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
        :param Nlim:        north limit (latitude) for plot
        :type Nlim:         float
        :param Elim:        east limit (longitude) for plot
        :type Elim:         float
        :param Slim:        south limit (latitude) for plot
        :type Slim:         float
        :param Wlim:        west limit (longitude) for plot
        :type Wlim:         float
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

        """
        maps.plot_domains(ncdirs,labels,outdir,Nlim,Elim,
                            Slim,Wlim,colours=colours,fname=fname,
                            fill_land=fill_land,fill_water=fill_water,
                            labpos=labpos)

    def frontogenesis(self,utc,level,ncdir,outdir,ncf=False,nct=False,
                        dom=1,smooth=0,clvs=0,title=0,cmap='bwr',
                        fig=False,ax=False,cb=True,match_nc=False,
                        Nlim=False,Elim=False,Slim=False,Wlim=False):
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
        :param f_suffix:    custom filename suffix for output. Ignore if False.
        :type f_suffix:     bool,str
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

        # Match domain
        if not Nlim and isinstance(match_nc,str):
            MATCH = WRFOut(match_nc)
            Nlim, Elim, Slim, Wlim = MATCH.get_limits()

        Front = self.W.compute_frontogenesis(utc,level)
        if isinstance(Front,N.ndarray):
            if Nlim:
                data,lats,lons = utils.return_subdomain(Front,self.W.lats1D,self.W.lons1D,
                                    Nlim,Elim,Slim,Wlim,fmt='latlon')
            else:
                lats= False
                lons = False

            if smooth:
                Front = stats.gauss_smooth(Front,smooth)

            if level==2000:
                lv_str = 'sfc'
            else:
                lv_str = str(level)

            F = BirdsEye(self.W,fig=fig,ax=ax)
            fname = self.create_fname('frontogen',utc,lv_str)
            # fname = 'frontogen_{0}_{1}.png'.format(lv_str,tstr)
            F.plot2D(Front,fname,outdir,clvs=clvs,lons=lons,lats=lats,
                        cmap=cmap,cb=cb)
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
            if level > 99:
                # Level is in pressure
                level = '{0}hPa'.format(level)
        return level

    def plot_radar(self,utc,datadir,outdir=False,Nlim=False,Elim=False,
                    Slim=False,Wlim=False,ncdir=False,nct=False,
                    ncf=False,dom=1,composite=False,locations=False,
                    fig=False,ax=False,cb=True,compthresh=False,
                    drawcounties=False):
        """
        Plot verification radar.

        composite allows plotting max reflectivity for a number of times
        over a given domain.
        This can show the evolution of a system.

        Need to rewrite so plotting is done in birdseye.
        """
        # Get limits of domain
        # import pdb; pdb.set_trace()
        if not Nlim and isinstance(ncdir,str):
            self.W = self.get_netcdf(ncdir,ncf=ncf,nct=nct,dom=dom)
            Nlim, Elim, Slim, Wlim = self.W.get_limits()

        if composite:
            radars = {}
            for n,t in enumerate(utc):
                radars[n] = Radar(t,datadir)
                if compthresh:
                    dBZ = radars[n].get_dBZ(radars[n].data)
                    radars[n].data[dBZ<compthresh] = 0
                if n == 0:
                    stack = radars[0].data
                else:
                    stack = N.dstack((stack,radars[n].data))
                # import pdb; pdb.set_trace()

            max_pixel = N.max(stack,axis=2)
            # Create new instance for the methods
            # Overwrite the data to become composite
            R = Radar(utc[-1],datadir)
            R.data = max_pixel

        else:
            R = Radar(utc,datadir)

        R.plot_radar(outdir,Nlim=Nlim,Elim=Elim,Slim=Slim,Wlim=Wlim,
                fig=fig,ax=ax,cb=cb,drawcounties=drawcounties)
        # print("Plotting radar for {0}".format(utc))

    def plot_accum_rain(self,utc,accum_hr,ncdir,outdir,ncf=False,nct=False,
                            f_prefix=0,f_suffix=False,dom=1,
                            plottype='contourf',smooth=1,fig=False,ax=False,
                            clvs=False,cmap=False,locations=False,
                            Nlim=False,Elim=False,Slim=False,Wlim=False):
        """
        Needs to be expanded to include other forms of precip.
        Plot accumulated precip (RAIN!) valid at time utc for accum_hr hours.

        """
        self.W = self.get_netcdf(ncdir,ncf=ncf,nct=nct,dom=dom)
        data = self.W.compute_accum_rain(utc,accum_hr)[0,:,:]
        fname = self.create_fname('accum_precip',utc)
        F = BirdsEye(self.W)
        F.plot2D(data,fname,outdir,lats=False,lons=False,
                    plottype=plottype,smooth=smooth,
                    clvs=clvs,cmap=cmap,locations=locations,
                    Nlim=Nlim,Elim=Elim,Slim=Slim,Wlim=Wlim)

    def all_error_growth(self,outdir,infodict,ylim=False,f_prefix=False,
                            f_suffix=False,energy='total'):
        """
        Compare many ensembles' DKE and DTE spreads on one plot.
        The times don't need to be identical.
        """
        fig = plt.figure()
        n_ens = len(infodict)
        cols = utils.generate_colours(M,n_ens)
        M.rcParams['axes.color_cycle'] = cols
        labels = []

        for ex in infodict:
            times = False
            data = utils.load_data(infodict[ex]['datadir'],infodict[ex]['dataf'],
                                    format='pickle')
            labels.append(ex)
            ave_stack = False
            for perm in data:
                if not times:
                    times = data[perm]['times']
                permdata = self.make_1D(data[perm]['values'])
                ave_stack = utils.vstack_loop(N.asarray(permdata),ave_stack)

            total_ave = N.average(ave_stack,axis=0)

            try:
                ls = infodict[ex]['ls']
            except KeyError:
                ls = '_'

            try:
                lcolor = infodict[ex]['col']
                plt.plot(times,total_ave,ls,color=lcolor)
            except KeyError:
                plt.plot(times,total_ave,ls,)

        plt.legend(labels,loc=2,handlelength=3)
        if ylim:
            plt.ylim(ylim)
        times_tup = [time.gmtime(t) for t in times]
        time_str = ["{2:02d}/{3:02d}".format(*t) for t in times_tup]
        plt.gca().set_xticks(times[::1])
        plt.gca().set_xticklabels(time_str[::1])
        units = r'$m^{2}s^{-2}$'
        plt.gca().set_ylabel("Difference {0} Energy ({1})".format(
                            energy.title(),units))
        fname = self.create_fname('allensembles',f_prefix=f_prefix,f_suffix=f_suffix)

        plt.tick_params(top='off')

        fpath = os.path.join(outdir,fname)
        plt.gca().relim
        plt.gca().autoscale_view()
        fig.savefig(fpath)

        plt.close()
        print(("Saved plot to {0}.".format(fpath)))

    def plot_delta(self,vrbl,utc,level=False,ncdir1=False,ncdir2=False,
                            outdir=False,ncf1=False,ncf2=False,nct=False,
                            f_prefix=0,f_suffix=False,
                            dom=1,plottype='contourf',smooth=1,
                            fig=False,ax=False,clvs=False,cmap=False,
                            locations=False,cb=True,match_nc=False,
                            Nlim=False,Elim=False,Slim=False,Wlim=False,
                            other=False):

        if ncdir1 is False or ncdir2 is False:
            raise Exception
        if outdir is False:
            outdir = os.path.expanduser("~")

        if level:
            level = self.get_level_string(level)

        # Match domain
        if not Nlim and isinstance(match_nc,str):
            MATCH = WRFOut(match_nc)
            Nlim, Elim, Slim, Wlim = MATCH.get_limits()

        # import pdb; pdb.set_trace()
        # Data
        self.W1 = self.get_netcdf(ncdir1,ncf=ncf1,nct=nct,dom=dom)
        self.W2 = self.get_netcdf(ncdir2,ncf=ncf2,nct=nct,dom=dom)
        # lats, lons = self.W.get_limited_domain(bounding)
        data1 = self.W1.get(vrbl,utc=utc,level=level,lons=False,lats=False,other=other)[0,0,:,:]
        data2 = self.W2.get(vrbl,utc=utc,level=level,lons=False,lats=False,other=other)[0,0,:,:]
        data = data1-data2
        # import pdb; pdb.set_trace()
        # Needs to be shape [1,1,nlats,nlons].
        if smooth>1:
            data = stats.gauss_smooth(data,smooth)

        if Nlim:
            data,lats,lons = utils.return_subdomain(data,self.W1.lats1D,self.W1.lons1D,
                                Nlim,Elim,Slim,Wlim,fmt='latlon')
        else:
            lats = False
            lons = False

        # Scales
        if clvs is False and cmap is False:
            S = Scales(vrbl,level)
            clvs = S.clvs
            cmap = S.cm
        elif clvs is False:
            S = Scales(vrbl,level)
            # clvs = S.clvs
            # Auto plot differences, this is a delta.
        elif cmap is False:
            S = Scales(vrbl,level)
            cmap = S.cm


        # Figure
        fname = self.create_fname(vrbl,utc,level,other=other,f_prefix='delta')
        F = BirdsEye(self.W1,fig=fig,ax=ax)
        # import pdb; pdb.set_trace()
        F.plot2D(data,fname,outdir,lats=lats,lons=lons,
                    plottype=plottype,smooth=smooth,
                    clvs=clvs,cmap=cmap,locations=locations,
                    cb=cb)



    def plot_axes_of_dilatation(self,utc,level=False,ncdir=False,outdir=False,
                ncf=False,nct=False,f_prefix=0,f_suffix=False,
                dom=1,plottype='contourf',smooth=1,
                fig=False,ax=False,clvs=False,cmap=False,
                locations=False,cb=True,match_nc=False,
                Nlim=False,Elim=False,Slim=False,Wlim=False,
                other=False):

        if ncdir is False:
            ncdir = os.path.expanduser("~")
        if outdir is False:
            outdir = os.path.expanduser("~")

        # Test at surface
        level = 2000
        if level:
            level = self.get_level_string(level)

        # Match domain
        if not Nlim and isinstance(match_nc,str):
            MATCH = WRFOut(match_nc)
            Nlim, Elim, Slim, Wlim = MATCH.get_limits()


        # Data
        self.W = self.get_netcdf(ncdir,ncf=ncf,nct=nct,dom=dom)
        # lats, lons = self.W.get_limited_domain(bounding)
        # U = self.W.get('U10',utc=utc,level=level,lons=False,lats=False,other=other)[0,0,:,:]
        # V = self.W.get('V10',utc=utc,level=level,lons=False,lats=False,other=other)[0,0,:,:]
        # import pdb; pdb.set_trace()
        # Needs to be shape [1,1,nlats,nlons].
        # if smooth>1:
            # data = stats.gauss_smooth(data,smooth)

        # data =
        xdata, ydata = self.W.return_axis_of_dilatation_components(utc)

        # if Nlim:
            # data,lats,lons = utils.return_subdomain(data,self.W.lats1D,self.W.lons1D,
                                # Nlim,Elim,Slim,Wlim,fmt='latlon')
        # else:
        lats = False
        lons = False

        # Figure
        fname = self.create_fname('axofdil',utc,level,other=other)
        F = BirdsEye(self.W,fig=fig,ax=ax)
        # import pdb; pdb.set_trace()
        F.axes_of_dilatation(xdata,ydata,fname,outdir,
                    lats=lats,lons=lons, locations=locations)

    def plot_diff_energy_spectrum(self,energy,ncfiles,utc=False,outdir=False):
        """
        Compute total KE/TE and the power spectrum of DKE/DTE for a time.
        """

        diff_data = stats.compute_diff_energy('2D',energy,files,utc,upper=None,lower=None,
                        d_save=False,d_return=True)

        mean_energy = {}
        for nc in ncfiles:
            NC = WRFOut(nc)
            U = NC.get('U',utc=utc,level=False,lons=False,lats=False)[0,0,:,:]
            V = NC.get('V',utc=utc,level=False,lons=False,lats=False)[0,0,:,:]
            T = NC.get('T',utc=utc,level=False,lons=False,lats=False)[0,0,:,:]
            R = 287.0 # Universal gas constant (J / deg K * kg)
            Cp = 1004.0 # Specific heat of dry air at constant pressure (J / deg K * kg)
            kappa = (R/Cp)
            mean_energy[nc] = 0.5*(U**2 + V**2 + kappa*(T**2))
        # mean_energy

    def meteogram(self,vrbl,loc,ncfiles,outdir=False,ncf=False,nct=False,dom=1):
        NCs = []
        for enspath in ncfiles:
            NCs.append(self.get_netcdf(enspath,ncf=ncf,nct=nct,dom=dom))

        TS = TimeSeries(NCs,list(loc.values())[0],list(loc.keys())[0])
        TS.meteogram(vrbl,outdir=outdir)
