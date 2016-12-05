import os
import matplotlib as M
M.use('Agg')
import matplotlib.pyplot as plt
import numpy as N
from netCDF4 import Dataset
import pdb
from mpl_toolkits.basemap import Basemap
import time
import glob
import calendar 

import sys
#sys.path.append('/home/jrlawson/gitprojects/')
import WEM.utils as utils
from WEM.utils import metconstants as mc
from .figure import Figure
from .defaults import Defaults
from .wrfout import WRFOut

"""
RUC/RAP data will probably need to be cut down to fit the WRF domain
it is compared to.

This script should inherit WRFOut and override the 'get' command.
"""

debug_get = 0

class RUC(WRFOut):
    def __init__(self,fpath,wrfdir=False):
        """
        config  :   configuration settings
        t       :   time, datenum format

        optional key-word arguments:
        wrfdir  :   if picked, domain is cut down
        """
        self.fpath = fpath
        # import pdb; pdb.set_trace()
        self.nc = Dataset(self.fpath)
        self.fields = [v for v in self.nc.variables]

        self.timekey = 'dummy'
        self.lvkey = 'lv_ISBL'
        self.lonkey = 'xgrid'
        self.latkey = 'ygrid'

        raw_time = self.nc.variables[self.fields[0]].initial_time
        self.utc = self.get_utc_time(raw_time)
        self.version = self.get_version()

        # self.fname = self.get_fname()
        # self.fpath = os.path.join(self.path_to_data,self.fname+'.nc')

        # Original lat and lon grids
        self.lats, self.lons = self.get_latlon()
        # Original lat/lon 1D arrays
        self.lats1D = self.lats[:,self.lats.shape[1]/2]
        # if self.version != 1:
        self.lons1D = self.lons[self.lons.shape[0]/2,:]
        # else:
            # self.lons1D = self.lons[:,self.lons.shape[0]/2]
        self.levels = self.get('pressure')[:].flatten()
       
        self.DX, self.DY = self.get_grid_spacing()
        self.dx = self.DX
        self.dy = self.DY
 
        if self.nc.variables[self.get_key('pressure')].units == 'Pa':
            self.levels = self.levels/100.0

        # Set dimension names, lengths.
        self.dimensions = [d for d in self.nc.dimensions]
        x,y,z = self.dimensions[:3]
            
        self.x_dim = len(self.nc.dimensions[x])
        self.y_dim = len(self.nc.dimensions[y])
        self.z_dim = len(self.nc.dimensions[z])
        del x,y,z

        if wrfdir:
            # It means all data should be cut to this size
            self.limits = self.colocate_WRF_map(wrfdir)
            self.lats2D = self.cut_2D_array(self.lats)
            self.lons2D = self.cut_2D_array(self.lons)
            self.lats, self.lons = self.cut_lat_lon()
            self.y_dim = len(self.lats)
            self.x_dim = len(self.lons)
            #self.lats1D = self.lats[:,self.lats.shape[1]/2]
            self.lats1D = self.lats
            #self.lons1D = self.lons[self.lons.shape[0]/2,:]
            self.lons1D = self.lons
        else:
            # Leave the dimensions the way they were
            self.y_dim = self.lats.shape[0]
            self.x_dim = self.lats.shape[1]

        # import pdb; pdb.set_trace()
        print(('RUC file loaded from {0}'.format(self.fpath)))
        # import pdb; pdb.set_trace()

    def get_utc_time(self,rawtime,fmt='datenum'):
        utc = time.strptime(rawtime,'%m/%d/%Y (%H:%M)')
        if fmt == 'datenum':
            utc = calendar.timegm(utc)
        return utc

    def check_compute(self,vrbl):
        """
        OVERRIDE WRFOUT VERSION
        """

        # try:
            # vrbl = self.get_key(vrbl)
        # except:
            # return False
        # else:
            # if vrbl in self.fields:
                # return True
            # else:
                # return False
        # if self.get_key(vrbl):
        return self.get_key(vrbl)

    def load(self,vrbl,tidx,lvidx,lonidx,latidx):
        """
        OVERRIDE WRFOUT VERSION
        """
        # Don't have to de-stagger RUC?
        # destag_dim = self.check_destagger(vrbl)

        # Do we need the dim names?
        # dim_names = self.get_dims(vrbl)

        # Next, correct the variable key
        vrbl = self.get_key(vrbl)

        # New axis for time (length = 1)
        # Way to account for 2D, 3D, 4D variables in RUC
        vrbldata = self.nc.variables[vrbl]
        # import pdb; pdb.set_trace()
        dim_names = self.get_dims(vrbl)
        sl = self.create_slice(vrbl,tidx,lvidx,lonidx,latidx,dim_names)
        # if vrbl is 'gridlat_0':
            # pdb.set_trace()
        vrbldata = vrbldata[sl]

        if len(vrbldata.shape) == 1:
            vrbldata = N.expand_dims(vrbldata,axis=-1)
            vrbldata = N.expand_dims(vrbldata,axis=-1)
            vrbldata = N.expand_dims(vrbldata,axis=0)
        elif len(vrbldata.shape) == 2:
            # Ugly!
            vrbldata = N.expand_dims(vrbldata,axis=0)
            vrbldata = N.expand_dims(vrbldata,axis=0)
        elif len(vrbldata.shape) == 3:
            vrbldata = N.expand_dims(vrbldata,axis=0)
            pass
        # elif len(vrbldata.shape) == 3:
            # vrbldata = vrbldata[N.newaxis,:,:,:]

        # Top/bottom is different to WRF?

        # Don't need to destagger, again?

        # Now add axis for time (missing)
        # vrbldata = N.expand_dims(vrbldata,axis=0)
        # if destag_dim and isinstance(sl[destag_dim],N.ndarray):
            # destag_dim = None
        # data = self.destagger(vrbldata[sl],destag_dim)
        # return data
        # pdb.set_trace()

        # Flip vertical to match WRF
        if len(vrbldata.shape) == 4:
            # Should always be the case!
            return vrbldata[:,::-1,:,:]
        else:
            raise Exception

    def get_MAYBE(self,vrbl,utc=False,level=False,lats=False,lons=False,
                smooth=1,other=False):
        """
        Overwrite WRFOut method here.
        """

        RUCvrbl = self.get_key(vrbl)
        
        coords = utils.check_vertical_coordinate(level)
        if level is False:
            lvidx = False
        elif coords == 'index':
            lvidx = level
        elif isinstance(coords,str):
            if coords == 'surface':
                lvidx = 0
            elif coords == 'isobaric':
                if level in self.levels:
                    lvidx = N.where(self.levels==level)
            else:
                lvidx = coords
        else:
            print("Invalid level selection.")
            raise Exception


    def cut_lat_lon(self):
        """ Return a smaller array of data
        depending of specified limits
        """
        lats = self.lats1D
        lons = self.lons1D
        lat_idx_max,lon_idx_min,a,b = utils.gridded_data.getXY(lats,lons,self.limits['Nlim'],self.limits['Wlim'])
        lat_idx_min,lon_idx_max,c,d = utils.gridded_data.getXY(lats,lons,self.limits['Slim'],self.limits['Elim'])

        # Dummy variables for exact lat/lons returned
        del a,b,c,d
        lats_out = lats[lat_idx_min:lat_idx_max]
        lons_out = lons[lon_idx_min:lon_idx_max]
        return lats_out, lons_out



    def cut_2D_array(self,data_in):
        """ Return a smaller array of data
        depending of specified limits
        """
        lats = self.lats1D
        lons = self.lons1D
        lat_idx_max,lon_idx_min,a,b = gridded_data.getXY(lats,lons,self.limits['Nlim'],self.limits['Wlim'])
        lat_idx_min,lon_idx_max,c,d = gridded_data.getXY(lats,lons,self.limits['Slim'],self.limits['Elim'])

        # Dummy variables for exact lat/lons returned
        del a,b,c,d
        data_out = data_in[lat_idx_min:lat_idx_max+1,lon_idx_min:lon_idx_max+1]
        return data_out

    def get_version(self):
        version = utils.RUC_version(self.utc)
        return version

    def plot(self,variables,lv,**kwargs):
        for va in variables:
            printtime = ('/'.join(["%02u" %s for s in self.ts[:4]]) +
                             "%02u" %self.ts[4] + ' UTC')
            print(("Plotting {0} for level {1} at time {2}".format(
                va,lv,printtime)))
            if 'scale' in kwargs:
                kwargs['scale'] = variables[va]['scale']
            if 'top' in kwargs:
                kwargs['top'] = variables[va]['top']
                kwargs['bottom'] = variables[va]['bottom']
            if va == 'streamlines':
                self.plot_streamlines(va,lv,**kwargs)
            else:
                self.plot_variable(va,lv,**kwargs)

    def plot_streamlines(self,va,lv,**kwargs):
        fig = plt.figure()

        # Scales, colourmaps in here

        # Get data
        u_all = self.get('U10',**kwargs)[:]
        v_all = self.get('V10',**kwargs)[:]

        u = self.cut_2D_array(u_all)
        v = self.cut_2D_array(v_all)
        m, x, y  = self.basemap_setup(**kwargs)
        """
        # Density depends on which version
        # Wanting to match 3 km WRF (which had 2.5 density)
        # Should work out dx, dy in __init__ method!

        WRF_density = 2.5
        WRF_res = 3.0

        if self.version == 3:
            RUC_res = 13.0

        density = WRF_density * RUC_res/WRF_res
        """
        density = 2.5

        #x = N.array(range(u.shape[0]))
        #y = N.array(range(v.shape[1]))

        #pdb.set_trace()
        #m.streamplot(x[self.x_dim/2,:],y[:,self.y_dim/2],u,v,density=density,linewidth=0.75,color='k')
        #m.streamplot(x[y.shape[1]/2,:],y[:,x.shape[0]/2],u,v,density=density,linewidth=0.75,color='k')
        #plt.streamplot(x[:,0],y[0,:],u,v)#,density=density,linewidth=0.75,color='k')
        m.streamplot(y[0,:],x[:,0],u,v,density=density,linewidth=0.75,color='k')
        #m.quiver(x,y,u,v)

        if self.C.plot_titles:
            title = utils.string_from_time('title',self.t)
            plt.title(title)

        # SAVE FIGURE
        datestr = utils.string_from_time('output',self.t)
        lv_na = utils.get_level_naming(va,lv=lv)
        na = (va,lv_na,datestr)
        self.fname = self.create_fname(*na)
        # pdb.set_trace()
        self.save(fig,self.output_root,self.fname)
        plt.close()

    def plot_variable(self,va,lv,**kwargs):
        self.fig = plt.figure()
        m, x, y  = self.basemap_setup(nc,**kwargs)
        # Scales, colourmaps in here

        data = self.get(va,**kwargs)
        if not 'scale' in kwargs:
            m.contourf(x,y,data)
        else:
            m.contourf(x,y,data,N.arange(*kwargs['scale']))

        if self.C.plot_titles:
            title = utils.string_from_time('title',self.t)
            plt.title(title)
        if self.C.colorbar:
            plt.colorbar(orientation='horizontal')

        # SAVE FIGURE
        datestr = utils.string_from_time('output',self.t)
        lv_na = utils.get_level_naming(lv,va)
        na = (va,lv_na,datestr)
        self.fname = self.create_fname(*na)
        # pdb.set_trace()
        self.save(self.fig,self.output_root,self.fname)
        plt.close()

    def get_OLD(self,va,**kwargs):
        if va == 'wind10':
            data = self.compute_wind10(self.nc)
        elif va == 'shear':
            data = self.compute_shear(self.nc,**kwargs)
        else:
            key = self.get_key(va)
            data = self.nc.variables[key][:]
        return data

    def get_latlon(self):
        lat_key = self.get_key('lats')
        lon_key = self.get_key('lons')

        lats = self.nc.variables[lat_key][...]
        lons = self.nc.variables[lon_key][...]

        return lats,lons

    def get_time_seq(self):
        """
        Makes sure time is time sequence
        (YYYY,MM,DD,HH,MM,SS)
        """
        if isinstance(self.t,int):
            # Time is datenum -> make sequence
            ts = time.gmtime(self.t)
        else:
            # Already a sequence
            ts = self.t
        return ts

    def get_fname(self):
        if self.ts[4] is not 0:
            print("No RUC data for this time")
            raise Exception
        # Depending on the RUC version, this might change

        if self.version==0:
            prefix = 'ruc2_252_'
        elif self.version==1:
            prefix = 'ruc2anl_252_'
        elif self.version==2:
            prefix = 'ruc2anl_130_'
        elif self.version==3:
            prefix = 'rap_130_'
        else:
            raise Exception

        fname = '{0}{1:04d}{2:02d}{3:02d}_{4:02d}00_000'.format(prefix,*self.ts)
        return fname


    def basemap_setup(self,**kwargs):
        # Fetch settings
        #pdb.set_trace()
        basemap_res = getattr(self.C,'basemap_res',self.D.basemap_res)
        #if 'lats' or 'lons' in kwargs:
        #    # These are 1D arrays
        #    lats = kwargs['lats']
        #    lons = kwargs['lons']
        #else:
        lats = self.lats1D
        lons = self.lons1D

        try:
            self.limits
        except:
            lllat = lats[0,0]
            lllon = lons[0,0]
            urlat = lats[-1,-1]
            urlon = lons[-1,-1]
        else:
            lllat = self.limits['Slim']
            lllon = self.limits['Wlim']
            urlat = self.limits['Nlim']
            urlon = self.limits['Elim']

        dx = 13.0
        dy = 13.0
        width_m = dx*(len(lons)-1)
        height_m = dy*(len(lats)-1)
        lat0 = lats[len(lats)/2]
        lon0 = lons[len(lons)/2]

        m = Basemap(
            projection='lcc',width=width_m,height=height_m,
            llcrnrlon=lllon,llcrnrlat=lllat,
            urcrnrlon=urlon,urcrnrlat=urlat,
            lat_0=lat0,lon_0=lon0,
            resolution=basemap_res,area_thresh=500
            )
        m.drawcoastlines()
        m.drawstates()
        m.drawcountries()

        # Draw meridians etc with wrff.lat/lon spacing
        # Default should be a tenth of width of plot, rounded to sig fig

        mx, my = N.meshgrid(self.lons,self.lats)
        y,x = m(mx,my)
        #pdb.set_trace()
        return m, x, y

    def colocate_WRF_map(self,wrfdir):
        searchfor = os.path.join(wrfdir,'wrfout*')
        # pdb.set_trace()
        files = glob.glob(searchfor)
        W = WRFOut(files[0])
        limits = {}
        limits['Nlim'] = W.lats.max()
        limits['Slim'] = W.lats.min()
        limits['Wlim'] = W.lons.min()
        limits['Elim'] = W.lons.max()

        return limits

    def compute_shear_old(self,nc,**kwargs):
        """
        top and bottom in km.
        kwargs['top']
        kwargs['bottom']

        """
        topm = kwargs['top']*1000
        botm = kwargs['bottom']*1000

        u = self.get('U')[::-1,...]
        v = self.get('V')[::-1,...]
        Z = self.get('Z')[::-1,...]

        topidx = N.zeros((225,301))
        botidx = N.zeros((225,301))
        ushear = N.zeros((225,301))
        vshear = N.zeros((225,301))

        for i in range(225):
            for j in range(301):
                topidx[i,j] = round(N.interp(
                                topm,Z[:,i,j],list(range(37))))
                botidx[i,j] = round(N.interp(
                                botm,Z[:,i,j],list(range(37))))
                ushear[i,j] = u[topidx[i,j],i,j] - u[botidx[i,j],i,j]
                vshear[i,j] = v[topidx[i,j],i,j] - v[botidx[i,j],i,j]

        # Find indices of bottom and top levels
        # topidx = N.where(abs(Z-topm) == abs(Z-topm).min(axis=1))
        # botidx = N.where(abs(Z-botm) == abs(Z-botm).min(axis=1))

        # ushear = u[0,:,topidx] - u[0,:,botidx]
        # vshear = v[0,topidx,:,:] - v[0,botidx,:,:]

        shear = N.sqrt(ushear**2 + vshear**2)
        # pdb.set_trace()
        return shear

    def compute_wind10_2(self,nc):
        """ Version '3' for RAP has U10 and V10
        """
        u = self.get('U')[-1,...]
        v = self.get('V')[-1,...]
        return N.sqrt(u**2 + v**2)

    def get_p(self,vrbl,tidx,level,lonidx,latidx):
        if isinstance(level,str) and level.endswith('hPa'):
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

        if vrbl=='pressure':
            dshape = self.get('U',utc=tidx,lons=lonidx,lats=latidx)[0,:,:,:].shape
            dataout = N.ones([nlv,dshape[-2],dshape[-1]])*hPa
        else:
            levels = 100*self.levels.flatten()
            datain = self.get(vrbl,utc=tidx,lons=lonidx,lats=latidx)[0,:,:,:]
            dataout = N.zeros([nlv,datain.shape[-2],datain.shape[-1]])
            for (i,j), p in N.ndenumerate(dataout[0,:,:]):
                # dataout[:,i,j] = N.interp(hPa,levels,datain[:,i,j])
                dataout[:,i,j] = N.interp(hPa,levels,datain[:,i,j][::-1])
        # pdb.set_trace()
        # data = N.expand_dims(dataout,axis=0)
        # import pdb; pdb.set_trace()
        return dataout

    def get_key(self,vrbl):
        """
        Returns the netcdf key for the desired variable
        """
        # Version 0/1 same and 2/3 same.
        key_no = {0:0,1:0,2:3,3:3}

        KEYS = {}
        KEYS['U'] = {0:'U_GRD_252_ISBL',1:'UGRD_P0_L100_GLC0',2:'UGRD_P0_L100_GLC0',3:'UGRD_P0_L100_GLC0'}
        KEYS['V'] = {0:'V_GRD_252_ISBL',1:'VGRD_P0_L100_GLC0',2:'VGRD_P0_L100_GLC0',3:'VGRD_P0_L100_GLC0'}
        KEYS['lats'] = {0:'gridlat_252',1:'gridlat_252',2:'gridlat_0',3:'gridlat_0'}
        KEYS['lons'] = {0:'gridlon_252',1:'gridlon_252',2:'gridlon_0',3:'gridlon_0'}
        KEYS['Z'] = {0:'HGT_252_ISBL',1:'HGT_P0_L100_GLC0',2:'HGT_P0_L100_GLC0',3:'HGT_P0_L100_GLC0'}
        KEYS['Td2'] = {0:'DPT_252_HTGL'}
        KEYS['U10'] = {3:'UGRD_P0_L103_GLC0'}
        KEYS['V10'] = {3:'VGRD_P0_L103_GLC0'}
        KEYS['T2'] = {0:'TMP_252_SFC'}
        # This is specific humidity, not mixing ratio of water tut tut
        KEYS['Q2'] = {0:'SPF_H_252_HTGL'}
        # KEYS['Q2'] = 
        KEYS['pressure'] = {0:'lv_ISBL2',1:'lv_ISBL2',2:'lv_ISBL0',3:'lv_ISBL0'}
        # KEYS['T'] = {0:'TMP_252_ISBL'} I THINK TMP = DRYBULB
        KEYS['drybulb'] = {0: 'TMP_252_ISBL',1: 'TMP_P0_L100_GLC0',2:'TMP_P0_L100_GLC0',3:'TMP_P0_L100_GLC0'}
        KEYS['W'] = {0:'V_VEL_252_ISBL',1:'VVEL_P0_L100_GLC0',2:'VVEL_P0_L100_GLC0',3:'VVEL_P0_L100_GLC0'}
        KEYS['PSFC'] = {0:'PRES_252_SFC'}
        KEYS['HGT'] = {0:'HGT_252_SFC'}
        KEYS['PMSL'] = {0:'MSLMA_252_MSL'}
        KEYS['RH'] = {0:'R_H_252_ISBL',3:'RH_P0_L100_GLC0'}

        try:
            key = KEYS[vrbl][key_no[self.version]]
        except KeyError:
            if debug_get:
                print(("Can't find variable {0}".format(vrbl)))
            # raise Exception
            return False
        else:
            return key

    def get_grid_spacing(self):

        yr = time.gmtime(self.utc).tm_year
        mth = time.gmtime(self.utc).tm_mon
        day = time.gmtime(self.utc).tm_mday

        if (yr < 1998) and (mth < 4) and (day < 6):
            n = 60e3
        elif (yr < 2002) and (mth < 4) and (day < 17):
            n = 40e3
        elif (yr < 2005) and (mth < 6) and (day < 28):
            n = 20e3
        else:
            n = 13e3
        return n, n

    def compute_theta(self,tidx,lvidx,lonidx,latidx,other):
        """Override due to lack of "T" in RUC.
        """
        P = self.get('pressure',tidx,lvidx,lonidx,latidx)
        T = self.get('drybulb',tidx,lvidx,lonidx,latidx)
        theta = T*((P/100000.0)**(-mc.R/mc.cp))
        # pdb.set_trace()
        return theta
