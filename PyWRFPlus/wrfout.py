from netCDF4 import Dataset
import sys
import os
import numpy as N
import calendar

#sys.path.append('/home/jrlawson/gitprojects/meteogeneral/')
#from meteogeneral.WRF import wrf_tools

class WRFOut:

    def __init__(self,config):
        self.C = config
        self.timeseq = self.C.inittime
        self.fname = self.get_fname(self.C)
        wrfout_abspath = os.path.join(self.C.wrfout_rootdir,self.C.datafolder,self.fname)
        self.nc = Dataset(wrfout_abspath,'r')
        self.wrftime = self.nc.variables['Times']
 
    def get_fname(self,C): # This is redundant with '__init__.padded_times()'
        yr = "%04u" % C.inittime[0]
        mth = "%02u" % C.inittime[1]
        day = "%02u" % C.inittime[2]
        hr = "%02u" % C.inittime[3]
        min = "%02u" % C.inittime[4]
        sec = "%02u" % C.inittime[5]
        dom = 'd0' + str(C.domain)
        datestr = '_'.join(('-'.join((yr,mth,day)),':'.join((hr,min,sec))))
        fname = '_'.join((C.wrfout_prefix,dom,datestr))
        return fname
    
    def format_time(self):
        pass 

 
    def get_wrf_times(self,C):
        self.times = self.nc.variables['Times'][:]        
        return self.times

    def get_dx(self,C):
        dx = self.nc.DX 
        return dx

    def get_dy(self,C):
        dy = self.nc.DY 
        return dy

    def get_plot_time_idx(self,timeseq):
        self.timeseq = timeseq
        self.reqtime = calendar.timegm(self.timeseq)
        nt = self.wrftime.shape[0]
        self.pytime = N.zeros([nt,1])
        t = self.wrftime
        # Now convert WRF time to Python time
        for i in range(nt):
            yr = int(''.join(t[i,0:4]))
            mth = int(''.join(t[i,5:7]))
            day = int(''.join(t[i,8:10]))
            hr = int(''.join(t[i,11:13]))
            min = int(''.join(t[i,14:16]))
            sec = int(''.join(t[i,17:19]))
            self.pytime[i] = calendar.timegm([yr,mth,day,hr,min,sec])

        # Now find closest WRF time
        self.time_idx = N.where(abs(self.pytime-self.reqtime) == abs(self.pytime-self.reqtime).min())[0][0]
        return self.time_idx

    def get_lvs(self,C):
        #nc.variables[
        pass

    def get_wrfout_fname(self,C):
        f = ''.join((C.wrfout_rootdir, C.wrfout_prefix+C.wrfout_inittime,
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

    def get_wrf_lats(self,C):
        lats = self.nc.variables['XLAT'][:]
        return lats

    def get_wrf_lons(self,C):
        lons = self.nc.variables['XLONG'][:]
        return lons

    def compute_comp_ref(self,time):
        self.time = time
        T2 = self.var('T2')
        QR = self.var('QRAIN')
        PSFC = self.var('PSFC')
        try:
            QS = self.var('QSNOW')
        except:
            QS = N.zeros(N.shape(QR))
        rhor = 1000.0
        rhos = 100.0
        rhog = 400.0
        rhoi = 917.0

        no_rain = 8.0E6
        # How do I access this time?
        no_snow = 2.0E6 * N.exp(-0.12*(T2[self.time]-273.15))
        no_grau = 4.0E6

        density = N.divide(PSFC[self.time],(287.0 * T2[self.time]))
        Qra_all = QR[self.time]
        Qsn_all = QS[self.time]

        for j in range(len(Qra_all[1,:,1])):
            curcol_r = []
            curcol_s = []
            for i in range(len(Qra_all[1,1,:])):
                    maxrval = N.max(Qra_all[:,j,i])
                    maxsval = N.max(Qsn_all[:,j,i])
                    curcol_r.append(maxrval)
                    curcol_s.append(maxsval)
            N_curcol_r = N.array(curcol_r)
            N_curcol_s = N.array(curcol_s)
            if j == 0:
                Qra = N_curcol_r
                Qsn = N_curcol_s
            else:
                Qra = N.row_stack((Qra, N_curcol_r))
                Qsn = N.row_stack((Qsn, N_curcol_s))

        # Calculate slope factor lambda
        lambr = (N.divide((3.14159 * no_rain * rhor), N.multiply(density, Qra))) ** 0.25
        lambs = N.exp(-0.0536 * (T2[self.time] - 273.15))

        # Calculate equivalent reflectivity factor
        Zer = (720.0 * no_rain * (lambr ** -7.0)) * 1E18
        Zes = (0.224 * 720.0 * no_snow * (lambr ** -7.0) * (rhos/rhoi) ** 2) * 1E18
        Zes_int = N.divide((lambs * Qsn * density), no_snow)
        Zes = ((0.224 * 720 * 1E18) / (3.14159 * rhor) ** 2) * Zes_int ** 2

        Ze = N.add(Zer, Zes)
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
