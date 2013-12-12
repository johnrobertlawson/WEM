from netCDF4 import Dataset
import sys
import os
import numpy as N
import calendar

#sys.path.append('/home/jrlawson/gitprojects/meteogeneral/')
#from meteogeneral.WRF import wrf_tools

class WRFOut:

    def __init__(self,fpath):
        #self.C = config
        self.nc = Dataset(fpath,'r')

        self.wrf_times = self.get('Times')
        self.dx = self.nc.DX
        self.dy = self.nc.DY
        #self.lvs = 
        self.lats = self.get('XLAT')   # Need to get lat and lon into plottable formats
        self.lons = self.get('XLONG')
        
        self.cen_lat = float(self.nc.CEN_LAT)
        self.cen_lon = float(self.nc.CEN_LON)
        self.truelat1 = float(self.nc.TRUELAT1)
        self.truelat2 = float(self.nc.TRUELAT2)
        self.x_dim = len(self.nc.dimensions['west_east'])
        self.y_dim = len(self.nc.dimensions['south_north'])
        
    def get_plot_time_idx(self,plot_time_seq):
        self.plot_time_seq = plot_time_seq
        self.time_epoch = calendar.timegm(self.plot_time_seq)
        nt = self.wrf_times.shape[0]
        self.wrf_times_epoch = N.zeros([nt,1])
        t = self.wrf_times   # For brevity

        for i in range(nt):
            yr = int(''.join(t[i,0:4]))
            mth = int(''.join(t[i,5:7]))
            day = int(''.join(t[i,8:10]))
            hr = int(''.join(t[i,11:13]))
            mins = int(''.join(t[i,14:16]))
            sec = int(''.join(t[i,17:19]))
            self.wrf_times_epoch[i] = calendar.timegm([yr,mth,day,hr,mins,sec])

        # Now find closest WRF time
        self.time_idx = N.where(abs(self.wrf_times_epoch-self.plot_time_epoch) == abs(self.wrf_times_epoch-self.plot_time_epoch).min())[0][0]
        return self.time_idx
        
    def get(self,var,time_idx):
        if var=='pressure':
            pass
        elif var=='sim_ref':
            data = self.compute_comp_ref(time_idx)
        elif var=='shear':
            data = self.compute_shear(0,3,time_idx)
        else:
            data = self.nc.variables[var][time_idx,...]
        return data

    def compute_shear(self,lower,upper):
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

    def compute_comp_ref(self,time):
        self.time = time
        T2 = self.get('T2')
        QR = self.get('QRAIN')
        PSFC = self.get('PSFC')
        try:
            QS = self.get('QSNOW')
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


