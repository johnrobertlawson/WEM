"""Take config settings and run plotting scripts
"""

from wrfout import WRFOut
from axes import Axes
from birdseye import BirdsEye

class PyWRFEnv:
    def __init__(self,config):
        self.wrftimes = WRFOut.get_wrf_times(config)
        self.dx = WRFOut.get_dx(config)
        self.dy = WRFOut.get_dy(config)
        self.lvs = WRFOut.get_lvs(config)
        self.plottime = WRFOut.get_plot_time(config)
        self.lats = WRFOut.get_wrf_lats(config)
        self.lons = WRFOut.get_wrf_lons(config)
        self.fig = plt.figure()

    def plot_CAPE(self,datatype='MLCAPE'):
        if not config.width:
            config.width = 8
        if not config.height:
            config.height = 8
        fig.set_size_inches(config.width,config.height)
        axes.setup(config)
    
    def plot_shear(self,upper=3,lower=0):
        if not config.plottype:
            config.plottype = 'contourf'
        fig = axes.setup(config)
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
    
    def plot_sim_ref(self,type='composite'):
        pass
    
    def plot_var(self,varlist):
        pass
        # This could be a combination of surface and upper-air data
    
    def sfc_data(self,varlist):
        # Varlist will be dictionary
        # Key is variable; value is plot type (contour, contour fill)
        # Some way of choosing plotting order?
        for v,p in varlist:
            # Plot data on top of each other in some order?
            
    def upper_lev_data(self,level,
        # Levels: isentropic, isobaric, geometric,
        
        
