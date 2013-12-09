from netCDF4 import Dataset
from meteogeneral.WRF import wrf_tools

class WRFOut:

    def __init__(self,config):
        nc = Dataset(config.wrfout_fname,'r')
         
    def get_wrf_times(self,config):
        times = nc.variables['Times'][:]        
        return times

    def get_dx(self,config):
        dx = nc.DX 
        return dx

    def get_dy(self,config):
        dy = nc.DY 
        return dy

    def get_plot_time(self,config):
        time = wrf_tools.find_time_index(config.wrftimes,config.plottime)
        return time

    def get_lvs(self,config):
        #nc.variables[
        pass

    def get_wrfout_fname(self,config):
        f = ''.join((config.wrfout_rootdir, config.wrfout_prefix+config.wrfout_inittime,
                    ''))
        return f 

    def compute_shear(self,upper,lower):
        pass
        return shear

    def compute_DTE(self,dims=1,upper=500,lower=None):
        if dims==1:
            pass
        elif dims==2:
            pass
        else:
            print "Dimensions for DTE computation are too large."
            raise Exception
        return DTE

    def compute_DKE:
        pass
        return DTE

    def get_wrf_lats(self,config):
        lats = nc.variables['XLAT'][:]
        return lats

    def get_wrf_lons(self,config):
        lons = nc.variables['XLONG'][:]
        return lons
