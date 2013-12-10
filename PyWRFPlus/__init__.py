"""Take config settings and run plotting scripts
"""
import os
import matplotlib as M
M.use('Agg')
import matplotlib.pyplot as plt

from wrfout import WRFOut
from axes import Axes
from figure import Figure, BirdsEye, CrossSection
import scales
from defaults import Defaults

class PyWRFEnv:
    def __init__(self,config):
        self.config = config

        # Set defaults if they don't appear in user's settings
        self.D = Defaults()
        self.config.domain = getattr(self.config,'domain',self.D.domain)
        self.font_prop = getattr(self.config,'font_prop',self.D.font_prop)
        self.usetex = getattr(self.config,'usetex',self.D.usetex)
        self.dpi = getattr(self.config,'dpi',self.D.dpi)
        self.title = getattr(self.config,'plot_title',self.D.title) 

        # Set some general settings
        M.rc('text',usetex=self.usetex)
        M.rc('font',**self.font_prop)
        M.rcParams['savefig.dpi'] = self.dpi

        # Create instance representing wrfout file
        # Will need way to have many wrfout files (for ensemble mean etc)
        self.W = WRFOut(self.config)

        # Same for figures
        # Some figures may have multiple variables and times?
        self.F = Figure()

        # Assign padded strings for date and time for initialisation (wrfout) time
        self.padded_time(self.W)


        # Could edit this to put variables within W rather than self?
        fname = self.W.get_fname(self.config)
        self.W.wrfout_abspath = os.path.join(config.wrfout_rootdir,config.datafolder,fname)
        self.wrftimes = self.W.get_wrf_times(config)
        self.dx = self.W.get_dx(config)
        self.dy = self.W.get_dy(config)
        self.lvs = self.W.get_lvs(config)
        self.plottime = self.W.get_plot_time(config)
        self.lats = self.W.get_wrf_lats(config)
        self.lons = self.W.get_wrf_lons(config)


    def plot_CAPE(self,datatype='MLCAPE'):
        if not self.config.width:
            self.config.width = 8
        if not self.config.height:
            self.config.height = 8
        self.fig.set_size_inches(self.config.width,self.config.height)
        axes.setup(config)
    
    def plot_shear(self,upper=3,lower=0):
        self.config.plottype = getattr(self.config,'plottype','contourf')
        #fig = axes.setup(config)
        shear = WRFOut.compute_shear(upper,lower)
        BirdsEye.plot2D(shear,config)
    
    def plot_cross_section(self,var,latA,lonA,latB,lonB):
        xs.plot(config,var,latA,lonA,latB,lonB)
        
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
        loc = self.config.output_dir # For brevity
        self.trycreate(loc)
        fname = 'blah.png'
        fpath = os.path.join(loc,fname)
        #self.fig.savefig(fpath)
        plt.gcf().savefig(fpath,bbox_inches='tight')
        
    
    def trycreate(self,loc):
        try:
            os.stat(loc)
        except:
            os.makedirs(loc)
    
    def figsize(self,defwidth,defheight):
        width = getattr(self.config,'width',defwidth)
        height = getattr(self.config,'height',defheight)
        self.fig.set_size_inches(width,height)

    def padded_times(self,obj)
        padded = ['{0:04d}'.format(t) for t in obj.timeseq]
        obj.yr, obj.mth, obj.day, obj.hr, obj.min, obj.sec = padded
 
