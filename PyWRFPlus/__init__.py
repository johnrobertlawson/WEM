"""Take config settings and run plotting scripts
"""
import os
import matplotlib.pyplot as plt

from wrfout import WRFOut
from axes import Axes
from birdseye import BirdsEye

class PyWRFEnv:
    def __init__(self,config):
        self.config = config
        # Set default if they don't appear in user's settings
        self.config.domain = getattr(self.config,'domain',1)

        # Create instance of WRFOut
        wrff = WRFOut(self.config) # This instance is the wrfout file
        fname = wrff.get_fname(self.config)
        wrff.wrfout_abspath = os.path.join(config.wrfout_rootdir,config.datafolder,fname)
        self.wrftimes = wrff.get_wrf_times(config)
        self.dx = wrff.get_dx(config)
        self.dy = wrff.get_dy(config)
        self.lvs = wrff.get_lvs(config)
        self.plottime = wrff.get_plot_time(config)
        self.lats = wrff.get_wrf_lats(config)
        self.lons = wrff.get_wrf_lons(config)
        self.fig = plt.figure()

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
        map = BirdsEye(self.fig,wrff,data)
        scale = scales.comp_ref # Import from dictionary in scales file
        fig,x,y = map.basemap_setup()
        if reftype == 'composite':
            data =  wrff.compute_comp_ref()
            longtitle = 'Composite Simulated Reflectivity'
        elif type(reftype) == 'int':
            data = wrff.compute_simref_atlevel()
            longtitle = 'Simulated Reflectivity at model level #' + str(reftype)

        self.fig.contourf(x,y,data,scale)
        self.save_fig()
        
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
        loc = self.config.output_dir
        self.trycreate(loc)
        fname = 'blah.png'
        fpath = os.path.join(loc,fname)
        self.fig.savefig(fpath)
    
    def trycreate(loc):
        try:
            os.stat(loc)
        except:
            os.makedirs(loc)

