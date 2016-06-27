import calendar
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from .defaults import Defaults
from mpl_toolkits.basemap import Basemap
import numpy as N

import WEM.utils as utils
import os

class ECMWF:
    def __init__(self,fpath,config):
        self.C = config
        self.D = Defaults()
        self.ec = Dataset(fpath,'r')
        self.times = self.ecmwf_times()
        # self.dx = 
        # self.dy =
        self.lats = self.ec.variables['g0_lat_2'][:] #N to S
        self.lons = self.ec.variables['g0_lon_3'][:] #W to E
        self.lvs = self.ec.variables['lv_ISBL1'][:] #jet to sfc
        self.fields = list(self.ec.variables.keys())
        self.dims = self.ec.variables['Z_GDS0_ISBL'].shape
        self.x_dim = self.dims[3]
        self.y_dim = self.dims[2]
        self.z_dim = self.dims[1]
        # Times, levels, lats, lons

    def ecmwf_times(self):
        ec_t = self.ec.variables['initial_time0_hours'][:]
        t = (ec_t*3600.0) - (1490184.0*3600.0)
        return t

    def find_level_idx(self,lv):
        lvs = list(self.lvs)
        lv_idx = lvs.index(lv)
        return lv_idx

    def find_time_idx(self,t):
        if isinstance(t,int):
            pass # It's a datenum
        else:
            t = calendar.timegm(t)
        ts = list(self.times)
        return ts.index(t)

    def get_key(self,va):
        keys = {}
        keys['Z'] = 'Z_GDS0_ISBL'
        keys['W'] = 'W_GDS0_ISBL'
        return keys[va]

    def get(self,va,lv,t):
        t_idx = self.find_time_idx(t)
        lv_idx = self.find_level_idx(lv)
        if va == 'wind':
            u = self.ec.variables['U_GDS0_ISBL'][t_idx,lv_idx,...]
            v = self.ec.variables['V_GDS0_ISBL'][t_idx,lv_idx,...]
            data = N.sqrt(u**2 + v**2)
        else:
            k = self.get_key(va)
            data = self.ec.variables[k][t_idx,lv_idx,...]
        return data


    def plot(self,va,lv,times,**kwargs):
        for t in times:

            fig = plt.figure()
            data = self.get(va,lv,t)
            m, x, y = self.basemap_setup()

            if 'scale' in kwargs:   
                S = kwargs['scale']
                f1 = m.contour(x,y,data,S,colors='k')
            else:
                f1 = m.contour(x,y,data,colors='k')

            if self.C.plot_titles:
                title = utils.string_from_time('title',t)
                plt.title(title)

            if 'wind_overlay' in kwargs:
                jet = kwargs['wind_overlay']
                wind = self.get('wind',lv,t)
                windplot = m.contourf(x,y,wind,jet,alpha=0.6)
                plt.colorbar(windplot)

            elif 'W_overlay' in kwargs:
                Wscale = kwargs['W_overlay']
                W = self.get('W',lv,t)
                windplot = m.contourf(x,y,W,alpha=0.6)
                plt.colorbar(windplot)


            # if self.C.colorbar:
                # plt.colorbar(orientation='horizontal')

            datestr = utils.string_from_time('output',t)
            fname = '_'.join(('ECMWF',va,str(lv),datestr)) + '.png'

            print(("Plotting {0} at {1} for {2}".format(
                        va,lv,datestr)))

            plt.clabel(f1, inline=1, fmt='%4u', fontsize=12, colors='k')

            utils.trycreate(self.C.output_root)
            plt.savefig(os.path.join(self.C.output_root,fname))
            plt.clf()
            plt.close()




    def basemap_setup(self,**kwargs):
        # Fetch settings
        basemap_res = getattr(self.C,'basemap_res',self.D.basemap_res)
        lllat = self.lats[-1]
        lllon = self.lons[0]
        urlat = self.lats[0]
        urlon = self.lons[-1]

        if 'Wlim' in kwargs:
            lllat = kwargs['Slim']
            lllon = kwargs['Wlim']
            urlat = kwargs['Nlim']
            urlon = kwargs['Elim']

        # dx = 13.0
        # dy = 13.0
        # x_dim = lats.shape[0]
        # y_dim = lats.shape[1]
        # width_m = dx*(x_dim-1)
        # height_m = dy*(y_dim-1)
        lat0 = self.lats[self.y_dim/2]
        lon0 = self.lons[self.x_dim/2]

        m = Basemap(
            projection='merc',
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
        mx, my = N.meshgrid(self.lons,self.lats)
        x,y = m(mx,my)
        return m, x, y
