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
#sys.path.append('/home/jrlawson/gitprojects/meteogeneral/')
#from meteogeneral.WRF import wrf_tools

class WRFOut:

    def __init__(self,fpath,config=0):
        self.path = fpath
        if config:
            self.C = config
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
        
        # Loads variable list 
        self.fields = self.nc.variables.keys()
        
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


    def check_compute(self,var):
        """This method returns the required variables
        that need to be loaded from the netCDF file.
        """
        
        if var in self.fields:
            return True
        else:
            return False
            
    def get(self,var,slices,**kwargs):
        """ Fetch a numpy array containing variable data.
        
        If field isn't present in the WRFOut file, compute it.
        
        Slice according to arguments.
        
        Destagger if required.
    
        Returns unstaggered, sliced data.
        
        var     :   netCDF variable name
        slices  :   dict, keys as follows:

        t       :   time index
        lv      :   level index
        la      :   latitude slice indices
        lo      :   longitude slice indices
        
        keyword arguments might include:
        
        Shear between 0 and 3 km:
        {'top':3, 'bottom',0}

        
        """
        # Check if computing required
        # When data is loaded from nc, it is destaggered
        isDefault = self.check_compute(var)
        if isDefault:
            data = self.load(var,slices)
        else:
            data = self.compute(var,slices,**kwargs)
        
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

    def compute(self,var,slices,**kwargs):
        """ Look up method needed to return array of data
        for required variable.
        
        Keyword arguments include settings for computation
        e.g. top and bottom of shear computation
        """
        
        tbl = {}
        tbl['shear'] = self.compute_shear
        tbl['thetae'] = self.compute_thetae
        tbl['cref'] = self.compute_comp_ref
        tbl['wind10'] = self.compute_wind10
        tbl['wind'] = self.compute_wind
        tbl['CAPE'] = self.compute_CAPE

        data = tbl[var](slices,**kwargs)
        return data

    def compute_wind10(self,slices):
        u = self.get('U10',slices)
        v = self.get('V10',slices)
        data = N.sqrt(u**2 + v**2)
        return data
        
    def compute_wind(self,slices):
        u = self.get('U',slices)
        v = self.get('V',slices)
        data = N.sqrt(u**2 + v**2)        
        return data

    def compute_shear(self,slices,**kwargs):
        """
        kwargs['top']
        kwargs['bottom']
        """
        pass
        #return shear

    def compute_thetae(self,slices):
        P = self.get('pressure',slices) # Computed
        Drybulb = self.get('temp',slices)
        Q = self.get('Q',slices)
        
        thetae = (Drybulb + (Q * cc.Lv/cc.cp)) * (cc.P0/P) ** cc.kappa
        return thetae

    def compute_comp_ref(self,PS):
        """Amend this so variables obtain at start fetch only correct date, lats, lons
        All levels need to be fetched as this is composite reflectivity
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
        
        DCAPE = cc.g 
        
    def compute_CAPE(self,slices):
        """
        CAPE method based on GEMPAK's pdsuml.f function
        
        Inputs:
        
        slices  :   dictionary of level/time/lat/lon

        
        
        Outputs:
        CAPE    :   convective available potential energy
        CIN     :   convective inhibition
        """

        totalCAPE = 0
        totalCIN = 0
        
        for lvidx in range(slices['lv']):
            # This should loop over the levels?
            """
            z1      :   bottom of layer (index)
            z2      :   top of layer (index)
            th1     :   theta (environ) at z1
            th2     :   theta (environ) at z2
            thp1    :   theta (parcel) at z1
            thp2    :   theta (parcel) at z2
            """
    
            z1 = lvidx
            z2 = lvidx + 1
    
            th1 = self.get('theta')
            th2 = self.get('theta')
            thp1 = pass
            thp2 = pass
    
            capeT = 0.0
            cinT = 0.0
            
            dt2 = thp2 - th2
            dt1 = thp1 - th1
            
            dz = z2 - z1
            
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
      
    def get_lat_idx(self,lat):
        lat_idx = N.where(abs(self.lats-lat) == abs(self.lats-lat).min())[0][0]
        return lat_idx
        
    def get_lon_idx(self,lon):
        lon_idx = N.where(abs(self.lons-lon) == abs(self.lons-lon).min())[0][0]
        return lon_idx
        
    def interp_to_p(self,config,nc_path,var,lv):
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


            
