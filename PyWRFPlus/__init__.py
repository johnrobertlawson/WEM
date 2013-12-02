"""Take config settings and run plotting scripts
"""

import WRFOut

class PyWRFEnv:
    def __init__(self,config):
        times = WRFOut.get_times(config)
        dx = WRFOut.get_dx(config)
        dy = WRFOut.get_dy(config)
        lvs = WRFOut.get_lvs(config)
        
    def plot_CAPE(self,type='MLCAPE'):
        Axes.setup(config)
    
    def plot_shear(self,upper=3,lower=0):
        pass
    
    def plot_cross_section(self,var,latA,lonA,latB,lonB):
        xs.plot(config,var,latA,lonA,latB,lonB)
        
    def compute_DKE(self):
        pass
    
    def compute_DTE(self):
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
        
    def 