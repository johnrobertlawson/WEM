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
import WEM.utils.utils as utils
from figure import Figure
from defaults import Defaults
from wrfout import WRFOut

class RUCPlot(Figure):
    def __init__(self,config):
        self.C = config
        self.path_to_data = self.C.path_to_RUC
        self.output_root = self.C.output_root
        self.D = Defaults()

    def plot(self,variables,**kwargs):
        for va in variables:
            for t in variables[va]['pt']:
                if 'lv' in variables[va]:
                    lv = variables[va]['lv']
                else:
                    lv = ''
                if 'scale' in variables[va]:
                    kwargs['scale'] = variables[va]['scale']
                if 'top' in variables[va]:
                    kwargs['top'] = variables[va]['top']
                    kwargs['bottom'] = variables[va]['bottom']
                self.plot_variable(va,t,lv,**kwargs)

    def plot_variable(self,va,t,lv,**kwargs):
        fname = self.get_fname(t)
        fpath = os.path.join(self.path_to_data,fname+'.nc')
        print fpath
        nc = Dataset(fpath)

        # pdb.set_trace()
        #lats, lons = get_latlon(nc)
        self.fig = plt.figure()
        m, x, y  = self.basemap_setup(nc,**kwargs)
        # Scales, colourmaps in here

        data = self.get(nc,va,**kwargs)
        if not 'scale' in kwargs:
            m.contourf(x,y,data)
        else:
            m.contourf(x,y,data,N.arange(*kwargs['scale']))

        if self.C.plot_titles:
            title = utils.string_from_time('title',t)
            plt.title(title)
        if self.C.colorbar:
            plt.colorbar(orientation='horizontal')

        # SAVE FIGURE
        datestr = utils.string_from_time('output',t)
        lv_na = utils.get_level_naming(lv,va)
        na = (va,lv_na,datestr)
        self.fname = self.create_fname(*na)
        # pdb.set_trace()
        self.save(self.fig,self.output_root,self.fname)
        plt.close()

    def get(self,nc,va,**kwargs):
        if va == 'CAPE':
            data = nc.variables['CAPE_252_SFC'][:]
        elif va == 'U':
            data = nc.variables['U_GRD_252_ISBL'][:]
        elif va == 'V':
            data = nc.variables['V_GRD_252_ISBL'][:]
        elif va == 'Z':
            data = nc.variables['HGT_252_ISBL'][:]
        elif va == 'wind10':
            data = self.compute_wind10(nc)
        elif va == 'shear':
            data = self.compute_shear(nc,**kwargs)
        elif va == 'dewpoint':
            data = nc.variables['DPT_252_HTGL'][:] - 273.15
        else:
            print("Choose variable")
            raise Exception
        return data

    def get_latlon(self,nc):
        # pdb.set_trace()
        lats = nc.variables['gridlat_252']
        lons = nc.variables['gridlon_252']
        return lats,lons

    def get_fname(self,t):
        if isinstance(t,int):
            # Time is datenum -> make sequence
            ts = time.gmtime(t)
        else:
            # Already a sequence
            ts = t

        if ts[4] is not 0:
            print("No RUC data for this time")
            raise Exception
        # Depending on the RUC version, this might change
        fname = 'ruc2_252_{0:04d}{1:02d}{2:02d}_{3:02d}00_000'.format(*ts)
        return fname


    def basemap_setup(self,nc,**kwargs):
        # Fetch settings
        basemap_res = getattr(self.C,'basemap_res',self.D.basemap_res)
        lats, lons = self.get_latlon(nc)
        lllat = lats[0,0]
        lllon = lons[0,0]
        urlat = lats[-1,-1]
        urlon = lons[-1,-1]

        if 'Wlim' in kwargs:
            lllat = kwargs['Slim']
            lllon = kwargs['Wlim']
            urlat = kwargs['Nlim']
            urlon = kwargs['Elim']

        dx = 13.0
        dy = 13.0
        x_dim = lats.shape[0]
        y_dim = lats.shape[1]
        width_m = dx*(x_dim-1)
        height_m = dy*(y_dim-1)
        lat0 = lats[x_dim/2,y_dim/2]
        lon0 = lons[x_dim/2,y_dim/2]

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
        # pdb.set_trace()
        mx, my = N.meshgrid(lons[0,:],lats[:,0])
        x,y = m(mx,my)
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

        u = self.get(nc,'U')[::-1,...]
        v = self.get(nc,'V')[::-1,...]
        Z = self.get(nc,'Z')[::-1,...]

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
        u = self.get(nc,'U')[-1,...]
        v = self.get(nc,'V')[-1,...]      
        return N.sqrt(u**2 + v**2)  

