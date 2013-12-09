from netCDF4 import Dataset
import sys
import os
import numpy as N

sys.path.append('/home/jrlawson/gitprojects/meteogeneral/')
from meteogeneral.WRF import wrf_tools

class WRFOut:

    def __init__(self,config):
        self.fname = self.get_fname(config)
        wrfout_abspath = os.path.join(config.wrfout_rootdir,config.datafolder,self.fname)
        self.nc = Dataset(wrfout_abspath,'r')
        
    def get_fname(self,config):
        yr = "%04u" % config.time[0]
        mth = "%02u" % config.time[1]
        day = "%02u" % config.time[2]
        hr = "%02u" % config.time[3]
        min = "%02u" % config.time[4]
        sec = "%02u" % config.time[5]
        dom = 'd0' + str(config.domain)
        datestr = '_'.join(('-'.join((yr,mth,day)),':'.join((hr,min,sec))))
        fname = '_'.join((config.wrfout_prefix,dom,datestr))
        return fname
 
    def get_wrf_times(self,config):
        self.times = self.nc.variables['Times'][:]        
        return self.times

    def get_dx(self,config):
        dx = self.nc.DX 
        return dx

    def get_dy(self,config):
        dy = self.nc.DY 
        return dy

    def get_plot_time(self,config):
        time = wrf_tools.find_time_index(self.times,config.plottime)
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

    def compute_DKE(self):
        pass
        return DTE

    def get_wrf_lats(self,config):
        lats = self.nc.variables['XLAT'][:]
        return lats

    def get_wrf_lons(self,config):
        lons = self.nc.variables['XLONG'][:]
        return lons

    def compute_comp_ref(self):
        T2 = self.var('T2')
        QR = self.var('QR')
        PSFC = self.var('PSFC')
        try:
            QS = self.var['QSNOW']
        except:
            QS = N.zeros(N.shape(QR))
        rhor = 1000.0
        rhos = 100.0
        rhog = 400.0
        rhoi = 917.0

        no_rain = 8.0E6
        # How do I access this time?
        no_snow = 2.0E6 * N.exp(-0.12*(T2[time]-273.15))
        no_grau = 4.0E6

        density = N.divide(PSFC[time],(287.0 * T2[time]))
        Qra_all = QR[time]
        Qsn_all = QS[time]

        for j in range(len(Qra_all[1,:,1])):
            curcol_r = []
            curcol_s = []
            for i in range(len(Qra_all[1,1,:])):
                    maxrval = N.max(Qra_all[:,j,i])
                    maxsval = N.max(Qsn_all[:,j,i])
                    curcol_r.append(maxrval)
                    curcol_s.append(maxsval)
            np_curcol_r = N.array(curcol_r)
            np_curcol_s = N.array(curcol_s)
            if j == 0:
                Qra = N_curcol_r
                Qsn = N_curcol_s
            else:
                Qra = N.row_stack((Qra, np_curcol_r))
                Qsn = N.row_stack((Qsn, np_curcol_s))

        # Calculate slope factor lambda
        lambr = (N.divide((3.14159 * no_rain * rhor), N.multiply(density, Qra))) ** 0.25
        lambs = N.exp(-0.0536 * (T2[time] - 273.15))

        # Calculate equivalent reflectivity factor
        Zer = (720.0 * no_rain * (lambr ** -7.0)) * 1E18
        Zes = (0.224 * 720.0 * no_snow * (lambr ** -7.0) * (rhos/rhoi) ** 2) * 1E18
        Zes_int = N.divide((lambs * Qsn * density), no_snow)
        Zes = ((0.224 * 720 * 1E18) / (3.14159 * rhor) ** 2) * Zes_int ** 2

        Ze = np.add(Zer, Zes)
        dBZ = N.nan_to_num(10*N.log10(Ze))
        return dBZ

    def compute_simref_atlevel(self,level=1):
        pass
        return data

    def var(self,var):
        if var=='P':
            pass
        else:
            data = self.nc.variables[var]
        return data
