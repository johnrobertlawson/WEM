"""
All matplotlib figures are subclasses of Figure.

"""

# Imports
import numpy as N
import matplotlib as M
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pdb
import os

# Custom imports
import WEM.utils as utils
from defaults import Defaults

class Figure(object):
    def __init__(self,config,wrfout,ax=0,fig=0,plotn=(1,1)):
        """
        C   :   configuration settings
        W   :   data
        """

        self.C = config
        self.W = wrfout
        self.D = Defaults()
        self.output_fpath = self.C.output_root

        #if wrfout=='RUC':
        #    pass
        #else:
        #    self.W = wrfout

        # Get settings for figure
        dpi = getattr(self.C,'DPI',self.D.dpi)
        
        # Create main figure
        if ax and fig:
            self.ax = ax
            self.fig = fig
        else:
            self.fig, self.ax = plt.subplots(nrows=plotn[0],ncols=plotn[1])
        self.fig.set_dpi(dpi)
        # self.ax = self.fig.add_subplot(111)
    
    def create_fname(self,*naming):
        """Default naming should be:
        Variable + time + level
        """
        fname = '_'.join([str(a) for a in naming])
        #pdb.set_trace()
        return fname
 
    def title_time(self):
        self.T = utils.padded_times(self.timeseq)
        pdb.set_trace()

    def figsize(self,defwidth,defheight,fig):
        width = getattr(self.C,'width',defwidth)
        height = getattr(self.C,'height',defheight)
        fig.set_size_inches(width,height)
        return fig

    def save(self,p2p,fname):
        # fig.tight_layout()
        utils.trycreate(p2p)
        fpath = os.path.join(p2p,fname)
        #self.fig.savefig(fpath)
        self.fig.savefig(fpath,bbox_inches='tight')
        print("Saving figure {0}".format(fpath))

    def get_limited_domain(self,da,smooth=1):
        if da:  # Limited domain area
            N_idx = self.W.get_lat_idx(da['N'])
            E_idx = self.W.get_lon_idx(da['E'])
            S_idx = self.W.get_lat_idx(da['S'])
            W_idx = self.W.get_lon_idx(da['W'])

            lat_sl = slice(S_idx,N_idx,smooth)
            lon_sl = slice(W_idx,E_idx,smooth)
        else:
            lat_sl = slice(None,None,smooth)
            lon_sl = slice(None,None,smooth)
        return lat_sl, lon_sl

    def just_one_colorbar(self,fpath):
        """
        docstring for just_one_colorbar"""
        try:
            with open(fpath): pass
        except IOError:
            self.create_colorbar(fpath)

    def create_colorbar(self,fpath,fname,cf,label=''):
        """
        Create colorbar.

        Inputs:
        fpath   :   path to file
        fname   :   filename
        cf      :   contour filling for legend
        label   :   colorbar label

        """
        fig = plt.figure()
        CBax = fig.add_axes([0.15,0.05,0.7,0.02])
        CB = plt.colorbar(cf,cax=CBax,orientation='horizontal')
        CB.set_label(label)
        self.save(fig,fpath,fname)

    def basemap_setup(self,smooth=1):
        # Fetch settings
        basemap_res = getattr(self.C,'basemap_res',self.D.basemap_res)

        width_m = self.W.dx*(self.W.x_dim-1)
        height_m = self.W.dy*(self.W.y_dim-1)
        
        m = Basemap(
            projection='lcc',width=width_m,height=height_m,
            lon_0=self.W.cen_lon,lat_0=self.W.cen_lat,lat_1=self.W.truelat1,
            lat_2=self.W.truelat2,resolution=basemap_res,area_thresh=500,
            ax=self.ax)
        m.drawcoastlines()
        m.drawstates()
        m.drawcountries()

        # Draw meridians etc with wrff.lat/lon spacing
        # Default should be a tenth of width of plot, rounded to sig fig

        s = slice(None,None,smooth)
        x,y = m(self.W.lons[s,s],self.W.lats[s,s])
        return m, x, y
