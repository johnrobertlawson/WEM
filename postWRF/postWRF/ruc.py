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

import sys
#sys.path.append('/home/jrlawson/gitprojects/')
import WEM.utils as utils
from figure import Figure
from defaults import Defaults
from wrfout import WRFOut

"""
RUC/RAP data will probably need to be cut down to fit the WRF domain
it is compared to.

This script should inherit WRFOut and override the 'get' command.
"""


class RUC(WRFOut):
    def __init__(self,fpath,**kwargs):
        """
        config  :   configuration settings
        t       :   time, datenum format

        optional key-word arguments:
        wrfdir  :   if picked, domain is cut down
        """
        self.fpath = fpath
        # self.C = config
        # self.D = Defaults()

        # self.path_to_data = self.C.path_to_RUC
        # self.output_root = self.C.output_root

        # self.t = t
        # Convert the datenum into time sequence
        # self.ts = self.get_time_seq()

        # self.version = self.get_version()
        # self.fname = self.get_fname()
        # self.fpath = os.path.join(self.path_to_data,self.fname+'.nc')

        self.nc = Dataset(self.fpath)

        # Original lat and lon grids
        self.lats, self.lons = self.get_latlon()
        # Original lat/lon 1D arrays
        self.lats1D = self.lats[:,self.lats.shape[1]/2]
        self.lons1D = self.lons[self.lons.shape[0]/2,:]

        if 'wrfdir' in kwargs:
            # It means all data should be cut to this size
            self.limits = self.colocate_WRF_map(kwargs['wrfdir'])
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


        print('RUC file loaded from {0}'.format(self.fpath))

    def cut_lat_lon(self):
        """ Return a smaller array of data
        depending of specified limits
        """
        lats = self.lats1D
        lons = self.lons1D
        lat_idx_max,lon_idx_min,a,b = gridded_data.getXY(lats,lons,self.limits['Nlim'],self.limits['Wlim'])
        lat_idx_min,lon_idx_max,c,d = gridded_data.getXY(lats,lons,self.limits['Slim'],self.limits['Elim'])

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
        """Returns the version of RUC or RUC file
        """
        yr = self.ts[0]
        mth = self.ts[1]

        if (yr > 2012) and (mth > 3): # With a massive gap for RAP
            version = 3
        elif (yr > 2007) and (mth > 10): # Massive gap after 2012/05 (transition to RAP).
            version = 2
        elif (yr>2006):
            version = 1
        elif (yr>2004):
            version = 0
        return version


    def plot(self,variables,lv,**kwargs):
        for va in variables:
            printtime = ('/'.join(["%02u" %s for s in self.ts[:4]]) +
                             "%02u" %self.ts[4] + ' UTC')
            print("Plotting {0} for level {1} at time {2}".format(
                va,lv,printtime))
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

    def get(self,va,**kwargs):
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

        lats = self.nc.variables[lat_key]
        lons = self.nc.variables[lon_key]

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

    def compute_shear(self,nc,**kwargs):
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
                                topm,Z[:,i,j],range(37)))
                botidx[i,j] = round(N.interp(
                                botm,Z[:,i,j],range(37)))
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

    def compute_wind10(self,nc):
        """ Version '3' for RAP has U10 and V10
        """
        u = self.get('U')[-1,...]
        v = self.get('V')[-1,...]
        return N.sqrt(u**2 + v**2)

    def get_key(self,va):
        """
        Returns the netcdf key for the desired variable
        """

        KEYS = {}
        KEYS['U'] = {0:'U_GRD_252_ISBL',1:'',2:'',3:'UGRD_P0_L100_GLC0'}
        KEYS['V'] = {0:'V_GRD_252_ISBL',1:'',2:'',3:'VGRD_P0_L100_GLC0'}
        KEYS['lats'] = {0:'gridlat_252',3:'gridlat_0'}
        KEYS['lons'] = {0:'gridlon_252',3:'gridlon_0'}
        KEYS['Z'] = {0:'HGT_252_ISBL'}

        KEYS['U10'] = {3:'UGRD_P0_L103_GLC0'}
        KEYS['V10'] = {3:'VGRD_P0_L103_GLC0'}

        KEYS['Td'] = {0:'DPT_252_HTGL'}

        try:
            key = KEYS[va][self.version]
        except KeyError:
            print("Choose variable")
            raise Exception
        else:
            return key

    def compute_frontogenesis_NOTNEEDED():
        """
        Override WRFOut frontogenesis?
        """

        dp = 15 # hPa to compute vertical gradients
        tidx = self.get_time_idx(time)
        if (tidx == 0) or (tidx == self.wrf_times.shape[0]-1):
            return None
        elif level == 2000:
            pass
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
            dgraddt, dgraddz, dgraddy, dgraddx = N.gradient(grad,dt,dp*100.0,
                                                            self.dy, self.dx)
            # Full equation
            Front = (dgraddt[1,1,:,:] + U[1,1,:,:]*dgraddx[1,1,:,:] +
                        V[1,1,:,:]*dgraddy[1,1,:,:])
                        # + omega[1,1,:,:]*dgraddz[1,1,:,:]
        return Front
