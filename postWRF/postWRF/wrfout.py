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
import constants as cc
import scipy.ndimage
import collections
import scipy.interpolate

import WEM.utils as utils
import metconstants as mc

class WRFOut(object):

    def __init__(self,fpath):
        self.path = fpath
        self.nc = Dataset(fpath,'r')

        self.wrf_times = self.nc.variables['Times'][:]
        self.dx = self.nc.DX
        self.dy = self.nc.DY

        #self.lvs =
        self.lats = self.nc.variables['XLAT'][0,...] # Might fail if only one time?
        self.lons = self.nc.variables['XLONG'][0,...]

        self.lats1D = self.lats[:,len(self.lats)/2]
        self.lons1D = self.lons[len(self.lons)/2,:]

        self.cen_lat = float(self.nc.CEN_LAT)
        self.cen_lon = float(self.nc.CEN_LON)
        self.truelat1 = float(self.nc.TRUELAT1)
        self.truelat2 = float(self.nc.TRUELAT2)
        self.x_dim = len(self.nc.dimensions['west_east'])
        self.y_dim = len(self.nc.dimensions['south_north'])
        self.z_dim = len(self.nc.dimensions['bottom_top'])

        self.P_top = self.nc.variables['P_TOP'][0]
        # Loads variable lists
        self.fields = self.nc.variables.keys()
        self.computed_fields = self.return_tbl().keys()
        self.available_vrbls = self.fields + self.computed_fields

        # Get times in nicer format
        self.wrf_times_epoch = self.wrftime_to_datenum()


    def wrftime_to_datenum(self):
        """
        Convert wrf's weird Times variable to datenum time.

        """
        times = self.wrf_times
        wrf_times_epoch = N.zeros([times.shape[0]])

        for n,t in enumerate(times):
            tstr = ''.join(t)

            yr = int(tstr[0:4])
            mth = int(tstr[5:7])
            day = int(tstr[8:10])
            hr = int(tstr[11:13])
            mins = int(tstr[14:16])
            sec = int(tstr[17:19])

            wrf_times_epoch[n] = calendar.timegm([yr,mth,day,hr,mins,sec])

        return wrf_times_epoch


    def get_time_idx(self,utc):

        """
        Input:

        utc     :   time, tuple or datenum
                    can be one or more

        Output:

        time_idx    :   index of time in WRF file
        """
        dn = utils.ensure_datenum(utc)
        tidx = utils.closest(self.wrf_times_epoch,dn)
        return tidx


    def check_compute(self,vrbl):
        """This method returns the required variables
        that need to be loaded from the netCDF file.
        """

        if vrbl in self.fields:
            return True
        else:
            return False

    def get(self,vrbl,utc=False,level=False,lats=False,lons=False,
                smooth=1,other=False):
        """
        Get data.

        Will interpolate onto pressure, height coordinates if needed.
        Will smooth if needed.

        Will accept the following.
        Time:
        indices (<500): integer/N.ndarray of integers
        time tuple: 6-item tuple or list/tuple of these
        datenum: integer >500 or list of them

        Level:
        indices: integer or N.ndarray of integers
        pressure: string ending in 'hPa'
        height: string ending in 'm' or 'km'
        isentropic: string ending in 'K'

        Lats:
        indices: integer or N.ndarray of integers
        lats: float or N.ndarray of floats

        Lons:
        indices: integer or N.ndarray of integers
        lons: float or N.ndarray of floats
        """
        # import pdb; pdb.set_trace()
        # Time
        if utc is False:
            tidx = False
        elif isinstance(utc,int) and utc<500:
            tidx = utc
        elif isinstance(utc,(list,tuple,int)): # and len(utc[0])==6:
            tidx = self.get_time_idx(utc)
        elif isinstance(utc,N.ndarray) and isinstance(utc[0],int):
            tidx = utc
        else:
            print("Invalid time selection.")
            raise Exception

        # Level
        # if not level:
        # import pdb; pdb.set_trace()
        coords = utils.check_vertical_coordinate(level)
        if level is False:
            lvidx = False
        elif coords == 'index':
            lvidx = level
        elif isinstance(coords,basestring):
            if coords == 'surface':
                lvidx = 0
            else:
                lvidx = coords
        else:
            print("Invalid level selection.")
            raise Exception

        # Lat/lon
        if not type(lats)==type(lons):
            # What about case where all lats with one lon?
            raise Exception
        if lats is False:
            lonidx = False
            latidx = False
        elif isinstance(lons,(list,tuple,N.ndarray)):
            if isinstance(lons[0],int):
                lonidx = lons
                latidx = lats
            elif isinstance(lons[0],float):
                # Interpolate to lat/lon
                lonidx = False
                latidx = False
        elif isinstance(lons,int):
            lonidx = lons
            latidx = lats
        elif isinstance(lons,float):
            # Interpolate to lat/lon
            lonidx = False
            latidx = False
        else:
            print("Invalid lat/lon selection.")
            raise Exception
        # import pdb; pdb.set_trace()
        # Check if computing required
        # When data is loaded from nc, it is destaggered
        if self.check_compute(vrbl):
            if lvidx is 'isobaric':
                data = self.get_p(vrbl,tidx,level,lonidx,
                            latidx)[N.newaxis,N.newaxis,:,:]
            elif isinstance(lvidx,(tuple,list,N.ndarray,int)):
                data = self.load(vrbl,tidx,lvidx,lonidx,latidx)
            else:
                raise Exception
        else:
            if lvidx is 'isobaric':
                data = self.get_p(vrbl,tidx,level,lonidx,
                            latidx)[N.newaxis,N.newaxis,:,:]
            else:
                data = self.compute(vrbl,tidx,lvidx,lonidx,latidx,other)
        return data

    def load(self,vrbl,tidx,lvidx,lonidx,latidx):

        # First, check dimension that is staggered (if any)
        destag_dim = self.check_destagger(vrbl)

        # Next, fetch dimension names
        dim_names = self.get_dims(vrbl)

        vrbldata = self.nc.variables[vrbl]
        sl = self.create_slice(vrbl,tidx,lvidx,lonidx,latidx,dim_names)
        # import pdb; pdb.set_trace()
        # If that dimension has a slice of indices, it doesn't need staggering.
        if destag_dim and isinstance(sl[destag_dim],N.ndarray):
            destag_dim = None

        data = self.destagger(vrbldata[sl],destag_dim)
        return data

    def create_slice(self,vrbl,tidx,lvidx,lonidx,latidx,dim_names):
        """
        Create slices from indices of level, time, lat, lon.
        Absence of a key means pick all indices.

        If PS is a list, this is to be converted to slices.
        """
        # See which dimensions are present in netCDF file variable
        sl = []
        # pdb.set_trace()
        if any('Time' in p for p in dim_names):
            if tidx is False:
                sl.append(slice(None,None))
            elif isinstance(tidx,slice) or isinstance(tidx,N.ndarray):
                sl.append(tidx)
            else:
                sl.append(slice(tidx,tidx+1))

        if any('bottom' in p for p in dim_names):
            if lvidx is False:
                sl.append(slice(None,None))
            elif isinstance(lvidx,int):
                sl.append(slice(lvidx,lvidx+1))
            elif isinstance(lvidx,N.ndarray):
                sl.append(lvidx)
            else: 
                sl.append(slice(None,None))

        if any('west' in p for p in dim_names):
            if lonidx is False:
                sl.append(slice(None,None))
            elif isinstance(lonidx,slice) or isinstance(lonidx,N.ndarray):
                sl.append(lonidx)
            elif isinstance(lonidx,int):
                sl.append(slice(lonidx,lonidx+1))
            else:
                sl.append(slice(None,None))

        if any('north' in p for p in dim_names):
            if latidx is False:
                sl.append(slice(None,None))
            elif isinstance(latidx,slice) or isinstance(latidx,N.ndarray):
                sl.append(latidx)
            elif isinstance(latidx,int):
                sl.append(slice(latidx,latidx+1))
            else:
                sl.append(slice(None,None))

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

        Don't destagger in x/y for columns

        """
        # Check for dimensions of 1.
        # If it exists, don't destagger it.

        shp = data.shape
        for n,size in enumerate(shp):
            if (size==1) and (n==ax):
                ax = None
                break

        #pdb.set_trace()

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

    def return_tbl(self):
        tbl = {}
        tbl['shear'] = self.compute_shear
        tbl['thetae'] = self.compute_thetae
        tbl['cref'] = self.compute_comp_ref
        tbl['wind10'] = self.compute_wind10
        tbl['wind'] = self.compute_wind
        tbl['CAPE'] = self.compute_CAPE
        tbl['Td'] = self.compute_Td
        tbl['pressure'] = self.compute_pressure
        tbl['temps'] = self.compute_temps
        tbl['theta'] = self.compute_theta
        tbl['geopot'] = self.compute_geopotential
        tbl['Z'] = self.compute_geopotential_height
        tbl['dptp'] = self.compute_dptp #density potential temperature pert.
        tbl['dpt'] = self.compute_dpt #density potential temperature .
        tbl['buoyancy'] = self.compute_buoyancy
        tbl['strongestwind'] = self.compute_strongest_wind
        tbl['PMSL'] = self.compute_pmsl
        tbl['RH'] = self.compute_RH
        tbl['dryairmass'] = self.compute_dryairmass
        tbl['QTOTAL'] = self.compute_qtotal
        tbl['olr'] =self.compute_olr

        return tbl

    def compute(self,vrbl,tidx,lvidx,lonidx,latidx,other,lookup=0):
        """ Look up method needed to return array of data
        for required variable.

        Keyword arguments include settings for computation
        e.g. top and bottom of shear computation

        lookup      :   enables a check to see if something can be
                        computed. Returns true or false.
        """
        tbl = self.return_tbl()
        if lookup:
            response = lookup in tbl
        else:
            response = tbl[vrbl](tidx,lvidx,lonidx,latidx,other)
        return response

    def compute_RH(self,tidx,lvidx,lonidx,latidx,other):
        T = self.get('temps',tidx,lvidx,lonidx,latidx,other='C')
        Td = self.get('Td',tidx,lvidx,lonidx,latidx)
        RH = N.exp(0.073*(Td-T))
        # pdb.set_trace()
        return RH*100.0

    def compute_dryairmass(self,tidx,lvidx,lonidx,latidx,other):
        MU = self.get('MU',tidx,lvidx,lonidx,latidx)
        MUB = self.get('MUB',tidx,lvidx,lonidx,latidx)
        return MU + MUB

    def compute_pmsl(self,tidx,lvidx,lonidx,latidx,other):
        P = self.get('PSFC',tidx,lvidx,lonidx,latidx)
        T2 = self.get('T2',tidx,lvidx,lonidx,latidx)
        HGT = self.get('HGT',tidx,lvidx,lonidx,latidx)

        temp = T2 + (6.5*HGT)/1000.0
        pmsl = P*N.exp(9.81/(287.0*temp)*HGT)

        #sm = kwargs.get('smooth',1)
        #data = pmsl[0,::sm,::sm]
        #return data
        return pmsl

    def compute_buoyancy(self,tidx,lvidx,lonidx,latidx,other):
        """
        Method from Adams-Selin et al., 2013, WAF
        """
        theta = self.get('theta',tidx,lvidx,lonidx,latidx)
        thetabar = N.mean(theta)
        qv = self.get('QVAPOR',tidx,lvidx,lonidx,latidx)
        qvbar = N.mean(qv)

        B = cc.g * ((theta-thetabar)/thetabar + 0.61*(qv - qvbar))
        return B

    def compute_mixing_ratios(self,tidx,lvidx,lonidx,latidx,other):
        qv = self.get('QVAPOR',tidx,lvidx,lonidx,latidx)
        qc = self.get('QCLOUD',tidx,lvidx,lonidx,latidx)
        qr = self.get('QRAIN',tidx,lvidx,lonidx,latidx)

        try:
            qi = self.get('QICE',tidx,lvidx,lonidx,latidx)
        except KeyError:
            print("MP scheme has no ice data.")
            qi = 0

        try:
            qs = self.get('QSNOW',tidx,lvidx,lonidx,latidx)
        except KeyError:
            print("MP scheme has no snow data.")
            qs = 0

        try:
            qg = self.get('QGRAUP',tidx,lvidx,lonidx,latidx)
        except KeyError:
            print("MP scheme has no graupel data.")
            qg = 0

        rh = qc + qr + qi + qs + qg
        rv = qv

        return rh, rv

    def compute_qtotal(self,tidx,lvidx,lonidx,latidx,other):
        qtotal, _ = self.compute_mixing_ratios(tidx,lvidx,lonidx,latidx)
        return qtotal

    def compute_dptp(self,tidx,lvidx,lonidx,latidx,other):
        dpt = self.get('dpt',tidx,lvidx,lonidx,latidx)
        dpt_mean = N.mean(dpt)
        dptp = dpt - dpt_mean
        return dptp

    def compute_dpt(self,tidx,lvidx,lonidx,latidx,other):
        """
        Potential: if surface level is requested, choose sigma level just
        about the surface. I don't think this affects any
        other dictionaries around...
        """
        # if tidx,lvidx,lonidx,latidx['lv'] == 0:
            # tidx,lvidx,lonidx,latidx['lv'] = 0
        theta = self.get('theta',tidx,lvidx,lonidx,latidx)
        rh, rv = self.compute_mixing_ratios(tidx,lvidx,lonidx,latidx)

        dpt = theta * (1 + 0.61*rv - rh)
        return dpt

    def compute_geopotential_height(self,tidx,lvidx,lonidx,latidx,other):
        geopotential = self.get('PH',tidx,lvidx,lonidx,latidx) + self.get('PHB',tidx,lvidx,lonidx,latidx)
        Z = geopotential/9.81
        return Z

    def compute_geopotential(self,tidx,lvidx,lonidx,latidx,other):
        geopotential = self.get('PH',tidx,lvidx,lonidx,latidx) + self.get('PHB',tidx,lvidx,lonidx,latidx)
        return geopotential

    def compute_wind10(self,tidx,lvidx,lonidx,latidx,other):
        u = self.get('U10',tidx,lvidx,lonidx,latidx)
        v = self.get('V10',tidx,lvidx,lonidx,latidx)
        data = N.sqrt(u**2 + v**2)
        return data

    def compute_pressure(self,tidx,lvidx,lonidx,latidx,other):
        PP = self.get('P',tidx,lvidx,lonidx,latidx)
        PB = self.get('PB',tidx,lvidx,lonidx,latidx)
        pressure = PP + PB
        return pressure

    def compute_temps(self,tidx,lvidx,lonidx,latidx,other='K'):
        theta = self.get('theta',tidx,lvidx,lonidx,latidx)
        P = self.get('pressure',tidx,lvidx,lonidx,latidx)
        temps = theta*((P/100000.0)**(287.04/1004.0))
        if other=='K':
            return temps
        elif other=='C':
            return temps-273.15

    def compute_theta(self,tidx,lvidx,lonidx,latidx,other):
        theta = self.get('T',tidx,lvidx,lonidx,latidx)
        Tbase = 300.0
        theta = Tbase + theta
        return theta

    def compute_wind(self,tidx,lvidx,lonidx,latidx,other):
        # pdb.set_trace()
        u = self.get('U',tidx,lvidx,lonidx,latidx)
        v = self.get('V',tidx,lvidx,lonidx,latidx)
        data = N.sqrt(u**2 + v**2)
        return data

    def compute_shear(self,tidx,lvidx,lonidx,latidx,other):
        """
        top and bottom in km.
        kwargs['top']
        kwargs['bottom']

        Could make this faster with numpy.digitize()?
        """
        topm = kwargs['top']*1000
        botm = kwargs['bottom']*1000

        u = self.get('U',tidx,lvidx,lonidx,latidx)
        v = self.get('V',tidx,lvidx,lonidx,latidx)
        Z = self.get('Z',tidx,lvidx,lonidx,latidx)

        topidx = N.zeros((450,450))
        botidx = N.zeros((450,450))
        ushear = N.zeros((450,450))
        vshear = N.zeros((450,450))

        for i in range(self.x_dim):
            for j in range(self.y_dim):
                topidx[i,j] = round(N.interp(
                                topm,Z[0,:,i,j],range(self.z_dim)))
                botidx[i,j] = round(N.interp(
                                botm,Z[0,:,i,j],range(self.z_dim)))
                ushear[i,j] = u[0,topidx[i,j],i,j] - u[0,botidx[i,j],i,j]
                vshear[i,j] = v[0,topidx[i,j],i,j] - v[0,botidx[i,j],i,j]

        # Find indices of bottom and top levels
        # topidx = N.where(abs(Z-topm) == abs(Z-topm).min(axis=1))
        # botidx = N.where(abs(Z-botm) == abs(Z-botm).min(axis=1))

        # ushear = u[0,:,topidx] - u[0,:,botidx]
        # vshear = v[0,topidx,:,:] - v[0,botidx,:,:]

        shear = N.sqrt(ushear**2 + vshear**2)
        # pdb.set_trace()
        return shear

    def compute_thetae(self,tidx,lvidx,lonidx,latidx,other):
        P = self.get('pressure',tidx,lvidx,lonidx,latidx) # Computed
        Drybulb = self.get('temp',tidx,lvidx,lonidx,latidx)
        Q = self.get('Q',tidx,lvidx,lonidx,latidx)

        thetae = (Drybulb + (Q * cc.Lv/cc.cp)) * (cc.P0/P) ** cc.kappa
        return thetae

    def compute_olr(self,tidx,lvidx,lonidx,latidx,other):
        OLR = self.get('OLR',tidx,lvidx,lonidx,latidx)
        sbc = 0.000000056704
        ir = ((OLR/sbc)**0.25) - 273.15
        return ir

    def compute_comp_ref(self,tidx,lvidx,lonidx,latidx,other):
        """Amend this so variables obtain at start fetch only correct date, lats, lons
        All levels need to be fetched as this is composite reflectivity
        """
        T2 = self.get('T2',tidx,False,lonidx,latidx)
        # QR = self.nc.variables['QRAIN'][PS['t'],:,PS['la'],PS['lo']]
        QR = self.get('QRAIN',tidx,False,lonidx,latidx) # This should get all levels
        PSFC = self.get('PSFC',tidx,False,lonidx,latidx)

        try:
            QS = self.get('QSNOW',tidx,False,lonidx,latidx)
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
        Qra_all = QR[0,...]
        Qsn_all = QS[0,...]

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
        lambr = (N.divide((3.14159 * no_rain * rhor), N.multiply(density, Qra)+N.nextafter(0,1))) ** 0.25
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

    def compute_DCP(self):
        """
        Derecho Composite Parameter (Evans and Doswell, 2001, WAF)
        And info from SPC Mesoanalyses
        """
        DCAPE = self.get('DCAPE')
        MUCAPE = self.get('MUCAPE')
        shear_0_6 = self.get('shear',0,6)
        meanwind_0_6 = self.get('meanwind',0,6)
        DCP = (DCAPE/980.0)*(MUCAPE/2000.0)*(shear_0_6/20.0)*(meanwind_0_6/16.0)
        return DCP

    def compute_DCAPE(self):

        pass

    def compute_thetae(self,tidx,lvidx,lonidx,latidx,other):
        P = self.get('pressure',tidx,lvidx,lonidx,latidx)
        T = self.get('temps',tidx,lvidx,lonidx,latidx,units='K')
        Td = self.get('Td',tidx,lvidx,lonidx,latidx)
        p2, t2 = thermo.drylift(P,T,Td)
        x = thermo.wetlift(p2,t2,100.0)
        thetae = thermo.theta(100.0, x, 1000.0)
        return thetae


    def compute_Td(self,tidx,lvidx,lonidx,latidx,other):
        """
        Using HootPy equation
        """
        Q = self.get('QVAPOR',tidx,lvidx,lonidx,latidx)
        P = self.get('pressure',tidx,lvidx,lonidx,latidx)
        w = N.divide(Q, N.subtract(1,Q))
        e = N.divide(N.multiply(w,P), N.add(0.622,w))/100.0
        a = N.multiply(243.5,N.log(N.divide(e,6.112)))
        b = N.subtract(17.67,N.log(N.divide(e,6.112)))
        Td = N.divide(a,b)
        # pdb.set_trace()
        return Td

    def compute_CAPE(self,tidx,lvidx,lonidx,latidx,other):
        """
        INCOMPLETE!

        CAPE method based on GEMPAK's pdsuml.f function

        Inputs:

        tidx,lvidx,lonidx,latidx  :   dictionary of level/time/lat/lon



        Outputs:
        CAPE    :   convective available potential energy
        CIN     :   convective inhibition
        """
        # Make sure all levels are obtained
        #tidx,lvidx,lonidx,latidx['lv'] = slice(None,None)

        totalCAPE = 0
        totalCIN = 0

        theta = self.get('theta',tidx,lvidx,lonidx,latidx)
        Z = self.get('Z',tidx,lvidx,lonidx,latidx)

        for lvidx in range(theta.shape[1]-1):
            if lvidx < 20:
                continue
            # This should loop over the levels?
            """
            z1      :   bottom of layer (index)
            z2      :   top of layer (index)
            th1     :   theta (environ) at z1
            th2     :   theta (environ) at z2
            thp1    :   theta (parcel) at z1
            thp2    :   theta (parcel) at z2
            """

            z1 = Z[0,lvidx,...]
            z2 = Z[0,lvidx+1,...]

            th1 = theta[0,lvidx,...]
            th2 = theta[0,lvidx+1,...]

            thp1 = 0
            thp2 = 0

            capeT = 0.0
            cinT = 0.0

            dt2 = thp2 - th2
            dt1 = thp1 - th1

            dz = z2 - z1

            dt1_pos_ma = N.ma.masked_greater(dt1,0)
            dt1_neg_ma = N.ma.masked_less(dt1,0)

            dt2_pos_ma = N.ma.masked_greater(dt2,0)
            dt2_neg_ma = N.ma.masked_less(dt2,0)

            dt1_pos = N.select([dt1>0],[dt1])
            dt1_neg = N.select([dt1<0],[dt1])
            dt2_pos = N.select([dt2>0],[dt1])
            dt2_neg = N.select([dt2<0],[dt1])

            pdb.set_trace()

            if (dt1 > 0) and (dt2 > 0):
                capeT = ((dt2 + dt1)*dz)/(th2+th1)
            elif dt1 > 0:
                ratio = dt1/(dt1-dt2)
                zeq = z1 + (dz*ratio)
                teq = th1 + ((th2-th1)*ratio)
                capeT = (dt1*(zeq-z1))/(th1+teq)
                cinT = (dt2*(z2-zeq))/(th2+teq)
            elif dt2 > 0:
                ratio = dt2/(dt2-dt1)
                zfc = z2-(dz*ratio)
                tfc = th2-((th2-th1)*ratio)
                capeT = (dt2*(z2-zfc)/(tfc+th2))
                cinT = (dt1*(zfc-z1)/(tfc+th1))
            else:
                cinT = ((dt2+dt1)*dz)/(th2+th1)

            if capeT > 0:
                CAPE = capeT * cc.g
            else:
                CAPE = 0

            if cinT < 0:
                CIN = cinT * cc.g
            else:
                CIN = 0

                totalCAPE += CAPE
                totalCIN += CIN

        return totalCAPE,totalCIN

    def compute_ave(self,va,z1,z2):
        """
        Compute average values for variable in layer

        Inputs:
        va      :   variable
        z1      :   height at bottom
        z2      :   height at top

        Output:
        data    :   the averaged variable
        """

        # Get coordinate system
        vc = self.check_vcs(z1,z2)

    def compute_strongest_wind(self,tidx,lvidx,lonidx,latidx,other):
        wind = self.get('wind10',tidx,lvidx,lonidx,latidx)
        wind_max = N.amax(wind,axis=0)
        # wind_max_smooth = self.test_smooth(wind_max)
        # return wind_max_smooth
        return wind_max

    def check_vcs(self,z1,z2,exception=1):
        """
        Check the vertical coordinate systems

        If identical, return the system
        If not, raise an exception.
        """

        vc = utils.level_type(z1)
        vc = utils.level_type(z2)

        if vc1 != vc2:
            print("Vertical coordinate systems not identical.")
            return False
            if exception:
                raise Exception
        else:
            return vc1

    def get_XY(self,lat,lon):
        """Return grid indices for lat/lon pair.
        """
        pass

    def get_limited_domain(self,da,skip=1,return_array='idx'):
        """
        Return smaller array of lats, lons depending on
        input dictionary of N, E, S, W limits.

        skip            :   for creating thinned domains
        return_type     :   if idx, return array of idx
                            if slice, return slice only
                            if latlon, return lat/lon values
        """
        
        if isinstance(da,dict): 
            N_idx = self.get_lat_idx(da['Nlim'])
            E_idx = self.get_lon_idx(da['Elim'])
            S_idx = self.get_lat_idx(da['Slim'])
            W_idx = self.get_lon_idx(da['Wlim'])
        else:
            N_idx = self.lats1D.shape[0]
            E_idx = self.lons1D.shape[0]
            S_idx = 0
            W_idx = 0

        if return_array=='latlon' or return_array=='slice':
            if isinstance(da,dict):
                lat_sl = slice(S_idx,N_idx,skip)
                lon_sl = slice(W_idx,E_idx,skip)
            else:
                lat_sl = slice(None,None,skip)
                lon_sl = slice(None,None,skip)
            
            if return_array == 'latlon':
                return self.lats1D[lat_sl], self.lons1D[lon_sl]
            else: 
                return lat_sl, lon_sl
        elif return_array=='idx':
            return N.arange(S_idx,N_idx,skip), N.arange(W_idx,E_idx,skip)
        else:
            print("Invalid selection for return_array.")
            raise Exception

    def get_lat_idx(self,lat):
        lat_idx = N.where(abs(self.lats-lat) == abs(self.lats-lat).min())[0][0]
        return lat_idx

    def get_lon_idx(self,lon):
        lon_idx = N.where(abs(self.lons-lon) == abs(self.lons-lon).min())[0][0]
        return lon_idx

    def get_p(self,vrbl,tidx,level,lonidx,latidx):
        """
        Return an pressure level isosurface of given variable.
        Interpolation is linear so watch out.

        Dimensions returns as (height,lat,lon)
        Or is it (height,lon, lat!?)

        TODO: Need to include limited domain functionality
        """
        if isinstance(level,basestring) and level.endswith('hPa'):
            hPa = 100.0*int(level.split('h')[0])
            nlv = 1
        elif isinstance(level,(float,int)):
            hPa = level*100.0
            nlv = 1
        elif isinstance(level,(tuple,list)):
            hPa = [l*100.0 for l in level]
            nlv = len(hPa)
        else:
            print("Use XXXhPa, an integer, or list of integers for level.")
            raise Exception

        # If this breaks, user is requesting non-4D data
        # Duck-typing for the win

        datain = self.get(vrbl,utc=tidx,lons=lonidx,lats=latidx)[0,...]
        P = self.get('pressure',utc=tidx,lons=lonidx,lats=latidx)[0,...]
        # import pdb; pdb.set_trace()
        dataout = N.zeros([nlv,P.shape[-2],P.shape[-1]])
        # pdb.set_trace()
        for (i,j), p in N.ndenumerate(dataout[0,:,:]):
            dataout[:,i,j] = N.interp(hPa,P[:,i,j][::-1],datain[:,i,j][::-1])
        # dataout = scipy.interpolate.griddata(P.flatten(),datain.flatten(),hPa)
        if nlv == 1:
            # Return 2D if only one level requested
            return dataout[0,:,:]
        else:
            return dataout

    def interp_to_p_fortran(self,config,nc_path,var,lv):
        """ Uses p_interp fortran code to put data onto a pressure
        level specified.

        Input:
        config  :   contains directory of p_interp files
        nc_path :   path to original netCDF file data
        var     :   variable(s) to compute
        lv      :   pressure level(s) to compute

        Returns:
        fpath   :   path to new netCDF file with p co-ords
        """
        # Fetch paths
        p_interp_path = os.path.join(
                            config.p_interp_root,'p_interp')
        namelist_path = os.path.join(
                            config.p_interp_root,'namelist.pinterp')
        nc_root, nc_fname = os.path.split(nc_path)# Root directory of wrfout file
        output_root = nc_root # Directory to dump output file (same)

        """
        Can we add a suffix to the new netCDF file?
        Check to see if file already exists
        """
        # Copy old p_interp for backup
        command1 = ' '.join(('cp',p_interp_path,p_interp_path+'.bkup'))
        os.system(command1)

        # Edit p_interp's namelist settings
        edit_namelist(path_to_interp,'path_to_input',nc_root,col=18)
        edit_namelist(path_to_interp,'input_name',nc_fname,col=18)
        edit_namelist(path_to_interp,'path_to_output',output_root,col=18)
        edit_namelist(path_to_interp,'process','list',col=18)
        edit_namelist(path_to_interp,'fields',var,col=18)
        edit_namelist(path_to_interp,'met_em_output','.FALSE.',col=18)
        edit_namelist(path_to_interp,'fields',var,col=18)

        command2 = os.path.join('./',p_interp_path)
        os.system(command2) # This should execute the script

        return fpath


    def edit_namelist(self,fpath,old,new,incolumn=1,col=23):
        """col=23 is default for wps namelists.
        """
        flines = open(fpath,'r').readlines()
        for idx, line in enumerate(flines):
            if old in line:
                # Prefix for soil intermediate data filename
                if incolumn==1:
                    flines[idx] = flines[idx][:col] + new + " \n"
                else:
                    flines[idx] = ' ' + old + ' = ' + new + "\n"
                nameout = open(fpath,'w')
                nameout.writelines(flines)
                nameout.close()
                break

    def get_limits(self):
        Nlim = self.lats[-1]
        Elim = self.lons[-1]
        Slim = self.lats[0]
        Wlim = self.lons[0]
        return Nlim, Elim, Slim, Wlim

    def cold_pool_strength(self,X,time,swath_width=100,env=0,dz=0):
        """
        Returns array the same shape as WRF domain.

        X   :   cross-section object with given path
                This path goes front-to-back through a bow
        km  :   width in the line-normal direction
        env :   (x,y) for location to sample environmental dpt
        """

        # Set up slices
        time_idx = self.get_time_idx(time)
        # lv_idx = 0

        # slices = {'t': time_idx, 'lv': lv_idx, 'la': lat_sl, 'lo': lon_sl}
        tidx,lvidx,lonidx,latidx = {'t':time_idx}

        # Get wind data
        wind10 = self.get('wind10',tidx,lvidx,lonidx,latidx)[0,...]
        T2 = self.get('T2',tidx,lvidx,lonidx,latidx)[0,...]

        # This is the 2D plane for calculation data
        coldpooldata = N.zeros(wind10.shape)

        # Compute required C2 fields to save time
        dpt = self.get('dpt',tidx,lvidx,lonidx,latidx)[0,...]
        Z = self.get('Z',tidx,lvidx,lonidx,latidx)[0,...]
        HGT = self.get('HGT',tidx,lvidx,lonidx,latidx)[0,...]
        heights = Z-HGT
        # pdb.set_trace()

        if isinstance(env,collections.Sequence):
            xx_env = N.arange(env[0]-2,env[0]+3)
            yy_env = N.arange(env[1]-2,env[1]+3)
            dpt_env = N.mean(dpt[:,yy_env,xx_env],axis=1)

        # All cross-sections (parallel)

        # xxx = [X.xx,]
        # yyy = [X.yy,]
        X.translate_xs(-swath_width/2)
        for n in range(swath_width):
            print("Cross section #{0}".format(n))
        # for xx, yy in zip(xxx, yyy):
            X.translate_xs(1)
            xx = X.xx
            yy = X.yy
            xx = xx.astype(int)
            yy = yy.astype(int)
            wind_slice = wind10[yy,xx]
            T2_slice = T2[yy,xx]
            slice_loc = self.find_gust_front(wind_slice,T2_slice,X.angle)

            gfx = xx[slice_loc]
            gfy = yy[slice_loc]
            gf_pts = N.intersect1d(N.where(xx==gfx)[0],N.where(yy==gfy)[0])
            gf_pt = gf_pts[int(len(gf_pts)/2.0)]
            xx_cp = xx[:gf_pt]
            yy_cp = yy[:gf_pt]
            # pdb.set_trace()

            # Compute enviromental dpt at each height
            # Average all levels from the location of gust front
            # forwards to the end of the cross-section.
            if not env:
                xx_env = xx[gf_pt+1:]
                yy_env = yy[gf_pt+1:]
                dpt_env = N.mean(dpt[:,yy_env,xx_env],axis=1)
            # pdb.set_trace()

            for x,y in zip(xx_cp,yy_cp):
            #for x,y in zip(xx,yy):
                if dz:
                    coldpooldata[y,x] = self.compute_cpdz(x,y,dpt[:,y,x],heights[:,y,x],dpt_env)
                    # pdb.set_trace()
                else:
                    coldpooldata[y,x] = N.sqrt(self.compute_C2(x,y,dpt[:,y,x],heights[:,y,x],dpt_env))

        return coldpooldata

    def compute_cpdz(self,x,y,dpt,heights,dpt_env):
        """
        Cold pool depth

        x       :   x location in domain
        y       :   y location in domain
        dpt     :   density potential temperature slice
        heights :   height AGL slice
        dpt_env :   environmental dpt, column
        """

        dz, zidx = self.cold_pool_depth(dpt,heights,dpt_env)
        # import pdb; pdb.set_trace()
        return dz

    def compute_C2(self,x,y,dpt,heights,dpt_env):
        """
        C^2 as found in James et al. 2006 MWR

        x       :   x location in domain
        y       :   y location in domain
        dpt     :   density potential temperature slice
        heights :   height AGL slice
        dpt_env :   environmental dpt, column
        """

        dz, zidx = self.cold_pool_depth(dpt,heights,dpt_env)
        C2 = -2*mc.g*((dpt[zidx]-dpt_env[zidx])/dpt_env[zidx])*dz
        # print("dpt = {0} ... dpt_env = {1} ... C2 = {2}".format(dpt[0],dpt_env[0],C2))
        return C2

    def cold_pool_depth(self,dpt,heights,dpt_env):
        dz = 0
        # thresh = -2.0
        thresh = -1.0
        for d,z, de in zip(dpt[1:],heights[1:],dpt_env[1:]):
            dptp = d - de
            if dptp > thresh:
                break
            dz = z

        if isinstance(dz,float):
            zidx = N.where(heights==dz)[0]
        else:
            zidx = 0
        # import pdb; pdb.set_trace()
        return dz, zidx

    def find_gust_front(self,wind_slice,T2_slice,angle,method=3):
        """
        Find location of maximum shear in the horizontal wind along a
        1D slice.

        wind_slice      :   1D numpy array
        T2_slice        :   temp 2m slice
        angle           :   angle of slice cross-section
        method          :   way to locate gust front
        """

        shp = wind_slice.shape

        # Compute gradient quantities
        shear = N.zeros(shp)
        T2grad = N.zeros(shp)

        for n in range(shp[0]):
            if n == 0 or n == shp[0]-1:
                shear[n] = 0
            else:
                len1 = abs(self.dx / N.sin(angle))
                len2 = abs(self.dx / N.cos(angle))
                hyp = min((len1,len2))
                # In kilometres:
                shear[n] = ((wind_slice[n+1]-wind_slice[n-1])/(2*hyp))*1000.0
                T2grad[n] = ((T2_slice[n+1]-T2_slice[n-1])/(2*hyp))*1000.0
                # N.W.DX

        # pdb.set_trace()

        # Go from B to A
        # Find first location where T2 drops and shear is ?

        if method==1:
            ### METHOD 1: USING THRESHOLDS
            # By default
            gfidx = shp/2
            for n, s, t in zip(range(shp)[::-1],shear[::-1],T2grad[::-1]):
                if (abs(s)>2.0) and (t<2.0):
                    gfidx = n
                    break

        elif method==2 or method==3:

            ### METHOD 2: FINDING MAX GRADIENTS AND AVERAGING
            shear = abs(shear)
            T2grad = abs(T2grad)

            xsh_idx = N.where(shear == shear.max())[0][0]
            xtg_idx = N.where(T2grad == T2grad.max())[0][0]
            print("Max shear loc: {0} ... max tempgrad loc: {1}".format(xsh_idx,xtg_idx))

            if method==2:
                gfidx = int((xsh_idx + xtg_idx)/2.0)
            else:
                gfidx = max([xsh_idx,xtg_idx])

        return gfidx
        # maxshearloc[0][0] returns the integer

    def compute_frontogenesis(self,time,level):
        """
        Note that all variables fetched with self.get have been
        destaggered and are at the same location.

        Output:
        Front       :   Frontgenesis in Kelvin per second.
        """
        # import pdb; pdb.set_trace()
        #ds = 1 # Number of grid point to compute horizontal gradient over
        #dx = ds
        #dy = ds
        #dz = 1 # Normal for vertical
        dp = 15 # hPa to compute vertical gradients
        tidx = self.get_time_idx(time)
        if (tidx == 0) or (tidx == self.wrf_times.shape[0]-1):
            return None
        elif level == 2000:
            tidxs = (tidx-1,tidx,tidx+1)
            T = {}
            TH2 = {}
            Z = {}
            dTdx = {}
            dTdy = {}
            dTdz = {}
            U = {}
            V = {}
            W = {}
            for n, tidx in enumerate(tidxs):
                # Just assume surface
                lvidx = 0
                U[n] = self.get('U10',{'t':tidx})[0,:,:]
                V[n] = self.get('V10',{'t':tidx})[0,:,:]
                W[n] = self.get('W',{'t':tidx, 'lv':0})[0,0,:,:]
                T[n] = self.get('T',{'t':tidx, 'lv':[0,1]})[0,:,:,:]
                TH2[n] = self.get('TH2',{'t':tidx})[0,:,:]
                # Z[n] = self.get('Z',{'t':tidx, 'lv' :[0,1]})[0,:,:,:]
                # We want gap between levels so not destaggered:
                Z[n] = (self.nc.variables['PH'][tidx,0:2,:,:] +
                        self.nc.variables['PHB'][tidx,0:2,:,:])/mc.g
                # Each grid point is DX km apart:
                dTdx[n], dTdy[n] = N.gradient(TH2[n])/self.dx

                # import pdb; pdb.set_trace()
                levelgap = Z[n][1,:,:] - Z[n][0,:,:]
                _, __, dTdz[n] = N.gradient(T[n])/levelgap
                # _, __, dTdz_2 = N.gradient(T[n])/levelgap
                # N.gradient needs interpolating to one level here?
                # dTdz[n] = (dTdz_2[1,...]-dTdz_2[0,...])/2.0

        elif isinstance(level,int):
            tidxs = (tidx-1,tidx,tidx+1)

            # Get sizes of array
            ny,nx = self.get_p('U',tidx,level).shape

            # Initialise them
            U = N.zeros([3,3,ny,nx])
            V = N.zeros_like(U)
            W = N.zeros_like(U)
            T = N.zeros_like(U)
            # omega = N.zeros_like(U)

            for n, t in enumerate(tidxs):
                U[n,...] = self.get_p('U',t,level)
                V[n,...] = self.get_p('V',t,level)
                W[n,...] = self.get_p('W',t,level)

                # 3D array has dimensions (vertical, horz, horz)
                T[n,...] = self.get_p('T',t,(level-dp,level,level+dp))

                # Compute omega
                # P = rho* R* drybulb
                # drybulb = T/((P0/P)^(R/cp)

            drybulb = 273.15 + (T/((100000.0/(level*100.0))**(mc.R/mc.cp)))
            rho = (level*100.0)/(mc.R*drybulb)
            omega = -rho * mc.g * W

            # Time difference in sec
            dt = self.wrf_times_epoch[tidx+1]-self.wrf_times_epoch[tidx]
            dTdt, dTdz, dTdy, dTdx = N.gradient(T,dt,dp*100.0,self.dy, self.dx)
            # Gradient part
            grad = (dTdx**2 + dTdy**2)**0.5
            # Full derivative - value wrong for dgraddz here
            dgraddt, dgraddz, dgraddy, dgraddx = N.gradient(grad,dt,dp*100.0,self.dy, self.dx)
            # Full equation
            Front = dgraddt[1,1,:,:] + U[1,1,:,:]*dgraddx[1,1,:,:] + V[1,1,:,:]*dgraddy[1,1,:,:] # + omega[1,1,:,:]*dgraddz[1,1,:,:]
        return Front
