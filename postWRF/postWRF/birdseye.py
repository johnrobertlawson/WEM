"""
Top-down 2D plots, which are so common in meteorology
that they get their own file here.

Subclass of Figure.
"""

import pdb
import matplotlib as M
M.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as N
import collections
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

from wrfout import WRFOut
from defaults import Defaults
from figure import Figure
import WEM.utils as utils
from scales import Scales
import stats

class BirdsEye(Figure):
    def __init__(self,wrfout,ax=0,fig=0):
        super(BirdsEye,self).__init__(wrfout,ax=ax,fig=fig)

    def get_contouring(self,vrbl='user',lv='user',**kwargs):
        """
        Returns colourmap and contouring levels

        Options keyword arguments:
        V   :   manually override contour levels
        """
        data = self.data.reshape((self.la_n,self.lo_n))

        # List of args and dictionary of kwargs
        plotargs = [self.x,self.y,data]
        plotkwargs = kwargs

        # cmap = getattr(kwargs,'cmap',plt.cm.jet)


            # if self.mplcommand == 'contour':
                # multiplier = S.get_multiplier(vrbl,lv)
        if 'clvs' in kwargs:
            if isinstance(kwargs['clvs'],N.ndarray):
                plotkwargs['levels'] = kwargs['clvs']
                kwargs.pop('clvs')

        if 'cmap' in kwargs:
            cmap = eval('M.cm.{0}'.format(kwargs['cmap']))
            plotkwargs['cmap'] = cmap
        # pdb.set_trace()

        if vrbl=='user':
            pass

        else:
            S = Scales(vrbl,lv)
            if S.cm:
                plotkwargs['cmap'] = S.cm
            if isinstance(S.clvs,N.ndarray):
                plotkwargs['levels'] = S.clvs

        return plotargs, plotkwargs

    def plot_data(self,data,time,outdir,fname,plottype='contourf',
                    save=1,smooth=1):
        """
        Generic method that plots any matrix of data on a map

        Inputs:
        data        :   2D matrix of data
        outdir      :   path to plots
        outf        :   filename for output (with or without .png)

        Optional:
        plottype    :   matplotlib function for plotting
        smooth      :   Gaussian smooth by this many grid spaces
        clvs        :   scale for contours
        title       :   title on plot
        save        :   whether to save to file
        """
        # INITIALISE
        self.bmap,self.x,self.y = self.basemap_setup(smooth=1)#ax=self.ax)
        self.mplcommand = mplcommand
        self.data = data
        self.la_n = self.data.shape[-2]
        self.lo_n = self.data.shape[-1]

        plotargs, plotkwargs = self.get_contouring(**kwargs)

        if self.mplcommand == 'contour':
            f1 = self.bmap.contour(*plotargs,**plotkwargs)
        elif self.mplcommand == 'contourf':
            f1 = self.bmap.contourf(*plotargs,**plotkwargs)
        elif self.mplcommand == 'pcolor':
            f1 = self.bmap.pcolor(*plotargs,**plotkwargs)
        elif self.mplcommand == 'pcolormesh':
            f1 = self.bmap.pcolormesh(*plotargs,**plotkwargs)
        elif self.mplcommand == 'scatter':
            f1 = self.bmap.scatter(*plotargs,**plotkwargs)
        else:
            print("Specify correct plot type.")
            raise Exception

        # LABELS, TITLES etc
        """
        Change these to hasattr!
        """
        #if self.C.plot_titles:
        if 'title' in kwargs:
            title_str = utils.string_from_time('title',pt,tupleformat=0)
            plt.title(title_str)
        plot_colorbar = 1
        if plot_colorbar:
            # self.fig.colorbar(f1,orientation='horizontal')
            self.fig.colorbar(f1,orientation='vertical')
        # plt.show(self.fig)
        # div0 = make_axes_locatable(self.ax)
        # cax0 = div0.append_axes("bottom", size="20%", pad=0.05)
        # cb0 = self.fig.colorbar(f1, cax=cax0)


        # SAVE FIGURE
        datestr = utils.string_from_time('output',pt,tupleformat=0)
        # self.fname = self.create_fname(fpath) # No da variable here
        if save:
            self.save(p2p,fname)

        plt.close(self.fig)
        return f1

        # print("Plot saved to {0}.".format(os.path.join(p2p,fname)))

    #def plot2D(self,va,**kwargs):
    def plot2D(self,vrbl,utc,level,outdir,dom=1,bounding=0,smooth=1,
                plottype='contourf',save=1,return_data=0,title=False):
        """
        Inputs:

        vrbl        :   variable string
        utc         :   date/time in (YYYY,MM,DD,HH,MM,SS) or datenum format
                        If tuple of two dates, it's start time and
                        end time, e.g. for finding max/average.
        level       :   level
        dom         :   domain
        outpath     :   absolute path to output
        bounding    :   list of four floats (Nlim, Elim, Slim, Wlim):
            Nlim    :   northern limit
            Elim    :   eastern limit
            Slim    :   southern limit
            Wlim    :   western limit
        smooth      :   smoothing. 1 is off. integer greater than one is
                        the degree of smoothing, to be specified.
        save        :   whether to save to file
        """
        # INITIALISE
        self.m,self.x,self.y = self.basemap_setup(smooth=1)
        self.plottype = plottype

        # Get indices for time, level, lats, lons
        tidx = self.W.get_time_idx(t)
        if title:
            title_str = utils.string_from_time('title',t)

        date_str = utils.string_from_time('output',t)

        # Until pressure coordinates are fixed TODO
        latidx, lonidx = self.get_limited_domain(bounding,smooth=1)

        if lv== 2000:
            lvidx = 0

        # if vc == 'surface':
        #     lv_idx = 0
        # elif lv == 'all':
        #     lv_idx = 'all'
        # else:
        #     print("Need to sort other levels")
        #     raise Exception

        # FETCH DATA
            ncidx = {'t': tidx, 'lv': lvidx, 'la': latidx, 'lo': lonidx}
            self.data = self.W.get(vrbl,ncidx)#,**vardict)
        else:
            self.data = self.W.get_p(vrbl,t,lv)
            # TODO: include bounding box for get_p

        if smooth>1:
            self.data = stats.gauss_smooth(self.data,smooth,pad_values='nan')

        self.la_n = self.data.shape[-2]
        self.lo_n = self.data.shape[-1]

        # COLORBAR, CONTOURING
        plotargs, plotkwargs = self.get_contouring(vrbl,lv)

        # S = Scales(vrbl,lv)

        # multiplier = S.get_multiplier(vrbl,lv)

        # if S.cm:
            # plotargs = (self.x,self.y,data.reshape((la_n,lo_n)),S.clvs)
            # cmap = S.cm
        # elif isinstance(S.clvs,N.ndarray):
            # if plottype == 'contourf':
                # plotargs = (self.x,self.y,data.reshape((la_n,lo_n)),S.clvs)
                # cmap = plt.cm.jet
            # else:
                # plotargs = (self.x,self.y,data.reshape((la_n,lo_n)),S.clvs)
        # else:
            # plotargs = (self.x,self.y,data.reshape((la_n,lo_n)))
            # cmap = plt.cm.jet
        # pdb.set_trace()

        if self.mplcommand == 'contourf':
            # f1 = self.bmap.contourf(*plotargs,cmap=cmap)
            f1 = self.bmap.contourf(*plotargs,**plotkwargs)
        elif self.mplcommand == 'contour':
            plotkwargs['colors'] = 'k'
            f1 = self.bmap.contour(*plotargs,**plotkwargs)
            # scaling_func = M.ticker.FuncFormatter(lambda x, pos:'{0:d}'.format(int(x*multiplier)))
            plt.clabel(f1, inline=1, fontsize=9, colors='k')

        # LABELS, TITLES etc
        if self.C.plot_titles:
            plt.title(title)
        if self.mplcommand == 'contourf' and self.C.colorbar:
            plt.colorbar(f1,orientation='horizontal')

        # SAVE FIGURE
        # pdb.set_trace()
        lv_na = utils.get_level_naming(vrbl,lv)
        naming = [vrbl,lv_na,datestr]
        if dom:
            naming.append(dom)
        self.fname = self.create_fname(*naming)
        if save:
            self.save(outpath,self.fname)
        plt.close()
        if isinstance(self.data,N.ndarray):
            return self.data.reshape((self.la_n,self.lo_n))


    def plot_streamlines(self,lv,pt,outpath,da=0):
        m,x,y = self.basemap_setup()

        time_idx = self.W.get_time_idx(pt)

        if lv==2000:
            lv_idx = None
        else:
            print("Only support surface right now")
            raise Exception

        lat_sl, lon_sl = self.get_limited_domain(da)

        slices = {'t': time_idx, 'lv': lv_idx, 'la': lat_sl, 'lo': lon_sl}

        if lv == 2000:
            u = self.W.get('U10',slices)[0,:,:]
            v = self.W.get('V10',slices)[0,:,:]
        else:
            u = self.W.get('U',slices)[0,0,:,:]
            v = self.W.get('V',slices)[0,0,:,:]
        # pdb.set_trace()

        #div = N.sum(N.dstack((N.gradient(u)[0],N.gradient(v)[1])),axis=2)*10**4
        #vort = (N.gradient(v)[0] - N.gradient(u)[1])*10**4
        #pdb.set_trace()
        lv_na = utils.get_level_naming('wind',lv=2000)

        m.streamplot(x[self.W.x_dim/2,:],y[:,self.W.y_dim/2],u,v,
                        density=2.5,linewidth=0.75,color='k')
        #div_Cs = N.arange(-30,31,1)
        #divp = m.contourf(x,y,vort,alpha=0.6)
        #divp = m.contour(x,y,vort)

        #plt.colorbar(divp,orientation='horizontal')
        if self.C.plot_titles:
            title = utils.string_from_time('title',pt)
            self.ax.set_title(title)
        datestr = utils.string_from_time('output',pt)
        na = ('streamlines',lv_na,datestr)
        fname = self.create_fname(*na)
        self.save(outpath,fname)

    def spaghetti(self,t,lv,va,contour,wrfouts,outpath,da=0,dom=0):
        """
        wrfouts     :   list of wrfout files

        Only change dom if there are multiple domains.
        """
        m,x,y = self.basemap_setup()

        time_idx = self.W.get_time_idx(t)

        colours = utils.generate_colours(M,len(wrfouts))

        # import pdb; pdb.set_trace()
        if lv==2000:
            lv_idx = None
        else:
            print("Only support surface right now")
            raise Exception

        lat_sl, lon_sl = self.get_limited_domain(da)

        slices = {'t': time_idx, 'lv': lv_idx, 'la': lat_sl, 'lo': lon_sl}

        # self.ax.set_color_cycle(colours)
        ctlist = []
        for n,wrfout in enumerate(wrfouts):
            self.W = WRFOut(wrfout)
            data = self.W.get(va,slices)[0,...]
            # m.contour(x,y,data,levels=[contour,])
            ct = m.contour(x,y,data,colors=[colours[n],],levels=[contour,],label=wrfout.split('/')[-2])
            print("Plotting contour level {0} for {1} from file \n {2}".format(
                            contour,va,wrfout))
            # ctlist.append(ct)
            # self.ax.legend()

        # labels = [w.split('/')[-2] for w in wrfouts]
        # print labels
        # self.fig.legend(handles=ctlist)
        # plt.legend(handles=ctlist,labels=labels)
        #labels,ncol=3, loc=3,
        #                bbox_to_anchor=[0.5,1.5])

        datestr = utils.string_from_time('output',t,tupleformat=0)
        lv_na = utils.get_level_naming(va,lv)
        naming = ['spaghetti',va,lv_na,datestr]
        if dom:
            naming.append(dom)
        fname = self.create_fname(*naming)
        self.save(outpath,fname)
