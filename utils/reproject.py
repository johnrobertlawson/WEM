import os

from scipy.interpolate import griddata
from mpl_toolkits.basemap import Basemap
import numpy as N

from WEM.postWRF.postWRF.wrfout import WRFOut
from WEM.postWRF.postWRF.hrrr import HRRR

class WRF_native_grid:
    def __init__(self,fpath):
        """Generates a basemap object for a WRF file's domain.  
        """
        W = WRFOut(fpath)
        cen_lat = W.nc.CEN_LAT
        cen_lon = W.nc.CEN_LON
        tlat1 = W.nc.TRUELAT1
        tlat2 = W.nc.TRUELAT2
        lllon = W.lons[0,0]
        lllat = W.lats[0,0]
        urlon = W.lons[-1,-1]
        urlat = W.lats[-1,-1]
        self.m = Basemap(projection='lcc',lat_1=tlat1,lat_2=tlat2,lat_0=cen_lat,
                            lon_0=cen_lon,llcrnrlon=lllon,llcrnrlat=lllat,
                            urcrnrlon=urlon,urcrnrlat=urlat,resolution='i')
        self.lons, self.lats, self.xx, self.yy = self.m.makegrid(W.lons.shape[1],W.lons.shape[0],returnxy=True)

class HRRR_native_grid(WRF_native_grid):
    """AS WRF_native_grid, but with some info about operational HRRR.
    """
    def __init__(self,fpath):
        W = HRRR(fpath)
        cen_lat = 38.5
        cen_lon = -97.5
        tlat1 = 38.5
        tlat2 = 38.5
        # lllon = -105.43488 # [0,0]
        lllon = W.lons[0,0]
        #lllat = 35.835026 # [0,0]
        lllat = W.lats[0,0]
        #urlon = -96.506653 # [-1,-1]
        urlon = W.lons[-1,-1]
        #urlat = 42.708714 # [-1,-1]
        urlat = W.lats[-1,-1]
        self.m = Basemap(projection='lcc',lat_1=tlat1,lat_2=tlat2,lat_0=cen_lat,
                            lon_0=cen_lon,llcrnrlon=lllon,llcrnrlat=lllat,
                            urcrnrlon=urlon,urcrnrlat=urlat,resolution='i')
        self.lons, self.lats, self.xx, self.yy = self.m.makegrid(W.lons.shape[1],W.lons.shape[0],returnxy=True)
        # return m, lons, lats, xx[0,:], yy[:,0]


def create_new_grid(Nlim=None,Elim=None,Slim=None,Wlim=None,proj='merc',
                    lat_ts=None,resolution='i',nx=None,ny=None,
                    tlat1=30.0,tlat2=60.0,cen_lat=None,cen_lon=None,
                    lllon=None,lllat=None,urlat=None,urlon=None):
    """Create new domain for interpolating to, for instance.

    The following are mandatory arguments for mercator ('merc'):
    Nlim,Elim,Slim,Wlim = lat/lon/lat/lon limits for north/east/south/west respectively
    lat_ts = latitude of true scale?
    nx,ny = number of points in the x/y direction respectively
    """
    if proj == 'merc':
        if None in (Nlim,Elim,Slim,Wlim,lat_ts,nx,ny):
            print("Check non-optional arguments.")
            raise Exception
        m = Basemap(projection=proj,llcrnrlat=Slim,llcrnrlon=Wlim,
                    urcrnrlat=Nlim,urcrnrlon=Elim,lat_ts=lat_ts,resolution='h')
    elif proj == 'lcc':
        if None in (tlat1,tlat2,cen_lat,cen_lon,lllon,lllat,urlon,urlat,nx,ny):
            print("Check non-optional arguments.")
            raise Exception
        m = Basemap(projection='lcc',lat_1=tlat1,lat_2=tlat2,lat_0=cen_lat,
                            lon_0=cen_lon,llcrnrlon=lllon,llcrnrlat=lllat,
                            urcrnrlon=urlon,urcrnrlat=urlat,resolution='i')
        
    lons, lats, xx, yy = m.makegrid(nx,ny,returnxy=True)
    return m, lons, lats, xx[0,:], yy[:,0]

def create_WRFgrid(f):
    """Constructs an instance of WRF_native_grid to return the vitals.

    f   -   absolute path to wrfout file.
    """
    W = WRF_native_grid(f)
    xx = W.xx
    yy = W.yy
    lons = W.lons
    lats = W.lats
    m = W.m
    return xx,yy,lats,lons,m

def reproject(data_orig,xx_orig=False,yy_orig=False,lats_orig=False,lons_orig=False,newgrid=False,
                    xx_new=False,yy_new=False):
    """
    data_orig               -   N.ndarray of original data (2D?)
    xx_orig,yy_orig         -   x-/y-axis indices of original data (shape?)
    lats_orig,lons_orig     -   lats/lons of original data (shape?)
    newgrid                 -   basemap class classobject
    """
    # xx_new,yy_new = newgrid(lons_orig,lats_orig)
    xx_new_dim = len(xx_new)
    yy_new_dim = len(yy_new)
    mx,my = N.meshgrid(xx_new,yy_new)
    # data_new = griddata((xx_orig.flat,yy_orig.flat),data_orig.flat,(xx_new.flat,
        # yy_new.flat)).reshape(xx_new_dim,yy_new_dim)
    data_new = griddata((xx_orig.flatten(),yy_orig.flatten()),data_orig.flatten(),
                        (mx.flatten(),my.flatten())).reshape(xx_new_dim,yy_new_dim)
    return data_new

