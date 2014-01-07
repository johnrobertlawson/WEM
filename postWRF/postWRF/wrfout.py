"""Compute or load data from netCDF file.

Dimensions of 4D variable X are X.dimensions:
(Time,bottom_top,south_north,west_east_stag)
Time, levels, latitude, longitude
"""

from netCDF4 import Dataset
import sys
import os
import numpy as N
import calendar
import pdb

#sys.path.append('/home/jrlawson/gitprojects/meteogeneral/')
#from meteogeneral.WRF import wrf_tools

class WRFOut:

    def __init__(self,fpath):
        #self.C = config
        self.nc = Dataset(fpath,'r')

        self.wrf_times = self.nc.variables['Times'][:]
        self.dx = self.nc.DX
        self.dy = self.nc.DY
        #self.lvs = 
        self.lats = self.nc.variables['XLAT'][0,...] # Might fail if only one time?
        self.lons = self.nc.variables['XLONG'][0,...]
        
        self.cen_lat = float(self.nc.CEN_LAT)
        self.cen_lon = float(self.nc.CEN_LON)
        self.truelat1 = float(self.nc.TRUELAT1)
        self.truelat2 = float(self.nc.TRUELAT2)
        self.x_dim = len(self.nc.dimensions['west_east'])
        self.y_dim = len(self.nc.dimensions['south_north'])
        
    def get_time_idx(self,t,tuple_format=0):
        
        """
        Input:
        
        t           :   time, not tuple format by default
        
        Output:
        
        time_idx    :   index of time in WRF file
        """
        if tuple_format:
            t_epoch = calendar.timegm(t)
        else:
            t_epoch = t
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
        self.time_idx = N.where(
                        abs(self.wrf_times_epoch-t_epoch) == 
                        abs(self.wrf_times_epoch-t_epoch).min()
                        )[0][0]
        return self.time_idx
        
    def get(self,var,PS):
        """ Fetch a numpy array containing variable data.
        
        Slice according to arguments.
        
        Destagger if required.
    
        Returns unstaggered, sliced data.
        
        var     :   netCDF variable name
        PS      :   Plot Settings, keys as follows:

        t       :   time index
        lv      :   level index
        la      :   latitude slice indices
        lo      :   longitude slice indices
        
        """
        # Check if computing required
        # When data is loaded from nc, it is destaggered
        if var=='pressure':
            if lv_idx == 0:
                data = self.load('PSFC',PS)
        elif var=='sim_ref':
            data = self.compute_comp_ref(PS)
        elif var=='shear':
            data = self.compute_shear(0,3,PS)
        elif var=='wind':
            u = self.load('U',PS)
            v = self.load('V',PS)
            data = N.sqrt(u**2 + v**2)
        else:
            data = self.load(var,PS)
        
        return data

    def load(self,var,PS):
    
        # First, check dimension that is staggered (if any)
        PS['destag_dim'] = self.check_destagger(var)

        # Next, fetch dimension names
        PS['dim_names'] = self.get_dims(var)

        d = self.nc.variables[var]
        sl = self.create_slice(PS)
        data = self.destagger(d[sl],PS['destag_dim'])
        return data

    def create_slice(self,PS):
        # See which dimensions are present in netCDF file variable
        sl = []
        if any('Time' in p for p in PS['dim_names']):
            sl.append(slice(PS['t'],PS['t']+1))
        if any('bottom' in p for p in PS['dim_names']):
            sl.append(slice(PS['lv'],PS['lv']+1))
        if any('north' and 'west' in p for p in PS['dim_names']):
            sl.append(PS['la']) 
            sl.append(PS['lo']) 
        
        return sl

    def check_destagger(self,var):
        """ Looks up dimensions of netCDF file without loading data.
        
        Returns dimension number that requires destaggering
        """
        stag_dim = None
        for n,dname in enumerate(self.nc.variables[var].dimensions):
            if 'stag' in dname:
                stag_dim = n
        
        return stag_dim

    def get_dims(self,var):
        dims = self.nc.variables[var].dimensions
        return dims

    def destagger(self,data,ax):
        """ Destagger data which needs it doing.
        
        data    :   numpy array of data requiring destaggering
        ax      :   axis requiring destaggering
        
        Theta always has unstaggered points in all three spatial dimensions (axes=1,2,3).
        
        Data should be 4D but just the slice required to reduce unnecessary computation time.
        """
        if ax==None:
            return data
        else:
            nd = data.ndim
            sl0 = []     # Slices to take place on staggered axis
            sl1 = []
            
            for n in range(nd):
                if n is not ax:
                    sl0.append(slice(None))
                    sl1.append(slice(None))
                else:
                    sl0.append(slice(None,-1))
                    sl1.append(slice(1,None))
                    
            data_unstag = 0.5*(data[sl0] + data[sl1])
            return data_unstag

    def compute_shear(self,lower,upper):
        pass
        return shear



    def compute_comp_ref(self,PS):
        """Amend this so variables obtain at start fetch only correct date, lats, lons
        All levels need to be fetched as it's composite reflectivity
        """
        T2 = self.get('T2',PS)
        QR = self.nc.variables['QRAIN'][PS['t'],:,PS['la'],PS['lo']]
        PSFC = self.get('PSFC',PS)
        try:
            QS = self.nc.variables['QRAIN'][PS['t'],:,PS['la'],PS['lo']]
        except:
            QS = N.zeros(N.shape(QR))
        rhor = 1000.0
        rhos = 100.0
        rhog = 400.0
        rhoi = 917.0

        no_rain = 8.0E6
        # How do I access this time?
        no_snow = 2.0E6 * N.exp(-0.12*(T2-273.15))
        no_grau = 4.0E6

        density = N.divide(PSFC,(287.0 * T2))
        Qra_all = QR
        Qsn_all = QS

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
        lambs = N.exp(-0.0536 * (T2 - 273.15))

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

    def get_XY(self,lat,lon):
        """Return grid indices for lat/lon pair.
        """
      
    def get_lat_idx(self,lat):
        lat_idx = N.where(abs(self.lats-lat) == abs(self.lats-lat).min())[0][0]
        return lat_idx
        
    def get_lon_idx(self,lon):
        lon_idx = N.where(abs(self.lons-lon) == abs(self.lons-lon).min())[0][0]
        return lon_idx
        

            
