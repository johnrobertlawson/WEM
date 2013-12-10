"""Take config settings and run plotting scripts.

Classes:
C = Config = configuration settings set by user, passed from user script
D = Defaults = used when user does not specify a non-essential item
W = Wrfout = wrfout file
F = Figure = a superclass of figures
    map = Birdseye = a lat--lon slice through data with basemap
    xs = CrossSection = distance--height slice through data with terrain 

"""

import os
import matplotlib as M
M.use('Agg')
import matplotlib.pyplot as plt

from wrfout import WRFOut
from axes import Axes
from figure import Figure
from birdseye import BirdsEye
import scales
from defaults import Defaults
import utils

class PyWRFEnv:
    def __init__(self,config):
        self.C = config

        # Set defaults if they don't appear in user's settings
        self.D = Defaults()
        self.C.domain = getattr(self.C,'domain',self.D.domain)
        self.font_prop = getattr(self.C,'font_prop',self.D.font_prop)
        self.usetex = getattr(self.C,'usetex',self.D.usetex)
        self.dpi = getattr(self.C,'dpi',self.D.dpi)
        self.title = getattr(self.C,'plot_title',self.D.title) 

        # Set some general settings
        M.rc('text',usetex=self.usetex)
        M.rc('font',**self.font_prop)
        M.rcParams['savefig.dpi'] = self.dpi

        # Assign filenames - will need to do for all files
        pass
    
        # Assign padded times for figures and wrfout file
        for obj in objs:
            utils.padded_times(obj)

        # Create instance representing wrfout file
        # Will need way to have many wrfout files (for ensemble mean etc)
        self.W = WRFOut(self.C)

        # Same for figures
        # Some figures may have multiple variables and times?
        self.F = Figure(self.C)

        # Could edit this to put variables within W rather than self?
        self.fname = self.W.get_fname(self.C)
        #self.W.wrfout_abspath = os.path.join(C.wrfout_rootdir,C.datafolder,fname)
        #self.wrftimes = self.W.get_wrf_times(C)
        #self.dx = self.W.get_dx(C)
        #self.dy = self.W.get_dy(C)
        #self.lvs = self.W.get_lvs(C)
        #self.plottime = self.W.get_plot_time(C)
        #self.lats = self.W.get_wrf_lats(C)
        #self.lons = self.W.get_wrf_lons(C)

    def plot_CAPE(self,datatype='MLCAPE'):
        pass
    
    def plot_shear(self,upper=3,lower=0):
        #self.C.plottype = getattr(self.C,'plottype','contourf')
        #fig = axes.setup(C)
        shear = WRFOut.compute_shear(upper,lower)
    
    def plot_cross_section(self,var,latA,lonA,latB,lonB):
        xs = CrossSection()
        xs.plot(var,latA,lonA,latB,lonB)
        
    def plot_DKE(self,dims=1):
        # dims = 1 leads to time series
        # dims = 2 vertically integrates to xxx hPa or xx sigma level?
        pass
    
    def plot_DTE(self):
        pass
    
    def plot_sim_ref(self,reftype='composite'):
        self.figsize(8,8)   # Sets default width/height if user does not specify
        map = BirdsEye(self.W)
        scale, cmap = scales.comp_ref() # Import scale and cmap
        self.bmap, x, y = map.basemap_setup()
        if reftype == 'composite':
            data = self.W.compute_comp_ref()
            title = 'Composite Simulated Reflectivity'
        elif isinstance(reftype,int):
            data = self.W.compute_simref_atlevel()
            title = 'Simulated Reflectivity at model level #' + str(reftype)

        self.bmap.contourf(x,y,data,scale)
        if self.title:
            plt.title(title)
        self.save_fig()
        
    def title_time(self):
        date_str = 
        return date_str

    def plot_var(self,varlist):
        pass
        # This could be a combination of surface and upper-air data
    
    def sfc_data(self,varlist):
        # Varlist will be dictionary
        # Key is variable; value is plot type (contour, contour fill)
        # Some way of choosing plotting order?
        for v,p in varlist:
            # Plot data on top of each other in some order?
            pass            
    def upper_lev_data(self,level):
        # Levels: isentropic, isobaric, geometric,
        pass 
    
    def save_fig(self):
        loc = self.C.output_dir # For brevity
        utils.trycreate(loc)
        fname = 'blah.png'
        fpath = os.path.join(loc,fname)
        #self.fig.savefig(fpath)
        plt.gcf().savefig(fpath,bbox_inches='tight')
        
    def figsize(self,defwidth,defheight):
        width = getattr(self.C,'width',defwidth)
        height = getattr(self.C,'height',defheight)
        self.fig.set_size_inches(width,height)


 
