import matplotlib as M
M.use('Agg')
import matplotlib.pyplot as plt
import datetime
import os
import numpy as N 
import cPickle as pickle
import matplotlib as M
from mpl_toolkits.basemap import Basemap
import netCDF4
# from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import griddata
import pdb

from WEM.postWRF.postWRF.sal import SAL
from WEM.postWRF.postWRF import WRFEnviron
import WEM.utils as utils

download = 0
SAL_precip_compute = 1
SAL_cref_compute = 0
SAL_climo__plot = 0
all_taSAL_plot = 0

NAMdir = '/chinook2/jrlawson/bowecho/paper4_NAMdata'
radardir = '/chinook2/jrlawson/bowecho/paper4_radardata' 
precipdir = '/chinook2/jrlawson/bowecho/paper4_precipdata' 
pickledir = '/chinook2/jrlawson/bowecho/paper4_pickles'

baseurl = 'http://nomads.ncdc.noaa.gov/data/meso-eta-hi'

startdate = datetime.datetime(2015,4,1,0,0,0)
# enddate = datetime.datetime(2015,4,3,0,0,0)
enddate = datetime.datetime(2015,9,1,0,0,0)
fhrs = N.arange(12,37,1)

def get_url(date,fhr,inittime=0):
    ym = '{0:04d}{1:02d}'.format(date.year,date.month)
    ymd = '{0:04d}{1:02d}{2:02d}'.format(date.year,date.month,date.day)
    fname = 'nam_218_{0}_{1:02d}{2:02d}_{3:03d}.grb'.format(ymd,date.hour,date.minute,fhr)
    url = os.path.join(baseurl,ym,ymd,fname)
    return url

def download_nam(url):
    fname = url.split('/')[-1]
    fpath = os.path.join(NAMdir,fname)
    cmnd = 'wget {0} -O {1}'.format(url,fpath)
    os.system(cmnd)
    return fpath

def convert_nam(fpath):
    fpath_new = fpath.replace('.grb','.nc')
    cmnd = 'ncl_convert2nc {0} -o {1}'.format(fpath,os.path.dirname(fpath))
    os.system(cmnd)
    return fpath_new

def ncks(fpath,vrbls):
    vrblstr = ','.join(vrbls)
    fpath_new = fpath.replace('.nc','_ncks.nc')
    cmnd = 'ncks -v {0} {1} {2}'.format(vrblstr,fpath,fpath_new)
    os.system(cmnd)
    return fpath_new

def delete_files(*args):
    for fpath in args:
        # answer = raw_input("Do you want to delete {0}? (Y/n)".format(fpath))
        # if answer == 'Y':
        os.system('rm -f {0}'.format(fpath))
    return

def get_ST4_fname(olddate,deltahour):
    date = olddate + datetime.timedelta(hours=int(deltahour))
    fname = 'ST4.{0:04d}{1:02d}{2:02d}{3:02d}.06h.nc'.format(date.year,date.month,date.day,date.hour)
    return fname

def get_NAM_fname(date,fcsthr):
    fname = 'nam_218_{0:04d}{1:02d}{2:02d}_{3:02d}{4:02d}_{5:03d}_ncks.nc'.format(
                        date.year,date.month,date.day,date.hour,date.minute,fcsthr)
    return fname

def get_SAL_native_grid():
    m = Basemap(projection='merc',llcrnrlat=34.0,llcrnrlon=-110.0,
                urcrnrlat=45.0,urcrnrlon=-85.0,lat_ts=39.5,)
    # m = Basemap(projection='lcc',llcrnrlat=34.0,llcrnrlon=-110.0,
                # urcrnrlat=45.0,urcrnrlon=-85.0,lat_0=30.0,lat_1=50.0,lon_0=-95.0)
    lons, lats, xx, yy = m.makegrid(537,308,returnxy=True)
    # print(xx[5,-1] - xx[5,-2])
    # print(xx[5,1] - xx[5,0])
    # print(yy[-1,5] - yy[-2,5])
    # print(yy[1,5] - yy[0,5])
    # import pdb; pdb.set_trace()
    # 4 km spacing
    return m, lons, lats, xx[0,:], yy[:,0]
    # return m, lons, lats, xx, yy

def get_ST4_native_grid():
    # m = Basemap(projection='npstere',llcrnrlon=-125.0,llcrnrlat=25.0,
            # urcrnrlon=-67.0,urcrnrlat=49.0,lon_0=-105.0,#lat_0=60.0,
                # boundinglat=24.701632)
    # m = Basemap(projection='stere',llcrnrlon=-125.0,llcrnrlat=25.0,
            # urcrnrlon=-67.0,urcrnrlat=49.0,lon_0=-105.0,lat_ts=60.0,
            # lat_0 = 24.701632)
    # lons, lats, xx, yy = m.makegrid(1121,881,returnxy=True)
    ST4ex =netCDF4.Dataset(os.path.join(precipdir,'ST4.2015090100.06h.nc'))
    lons = ST4ex.variables['g5_lon_1'][:]
    lats = ST4ex.variables['g5_lat_0'][:]
    # xx, yy = m(lons,lats)
    # pdb.set_trace()
    # return m, lons, lats, xx[0,:], yy[:,0]
    return lons, lats

def get_SAL_regrid_xy(m,lons,lats):
    xx,yy = m(lons,lats)
    return xx[0,:], yy[:,0]

def get_NAM_native_grid():
    m = Basemap(projection='lcc',llcrnrlon=-133.459,llcrnrlat=12.190,
            urcrnrlon=-49.420,urcrnrlat=57.328,lat_1=25.0,
            lon_0=-95.0)
    lons, lats, xx, yy = m.makegrid(614,428,returnxy=True)
    return m, lons, lats, xx[0,:], yy[:,0]

if download:
    date = startdate
    while date < enddate:
        for fhr in fhrs:
            url = get_url(date,fhr)
            fpath_grb = download_nam(url)
            fpath_nc = convert_nam(fpath_grb)
            vrbls = ['REFC_218_EATM',]
            if (fhr == 24) or (fhr == 36):
                vrbls.append('A_PCP_218_SFC_acc12h')
            fpath_ncks = ncks(fpath_nc,vrbls)
            delete_files(fpath_grb,fpath_nc)
        date = date + datetime.timedelta(days=1)


if SAL_precip_compute:
    DATA = {}
    # SAL (new) grid
    SALmap, SALlons, SALlats, SAL_native_xx, SAL_native_yy = get_SAL_native_grid()
    SAL_native_mx, SAL_native_my = N.meshgrid(SAL_native_xx,SAL_native_yy)

    # ST4 grid and coordinates within new grid
    ST4lons, ST4lats = get_ST4_native_grid()
    ST4_SALgrid_xx,ST4_SALgrid_yy = SALmap(ST4lons,ST4lats)

    # NAM grid and coordinates within new grid
    NAMmap, NAMlons, NAMlats, nam_native_xx, nam_native_yy = get_NAM_native_grid()
    # nam_native_mx, nam_native_my = N.meshgrid(nam_native_xx, nam_native_yy)

    NAM_SALgrid_xx,NAM_SALgrid_yy = SALmap(NAMlons,NAMlats)
    # SAL_NAMgrid_xx, SAL_NAMgrid_yy = get_SAL_regrid_xy(NAMmap,SALlons,SALlats)
    # nam_mx, nam_my = N.meshgrid(SAL_NAMgrid_xx,SAL_NAMgrid_yy)

    date = startdate
    while date < enddate:
        broken = False
        # Load raw obs data - in mm
        # Add together date +18, +24, +30, +36.
        for deltahour in (18,24,30,36):
            obs_fname = get_ST4_fname(date,deltahour)
            obs_f = os.path.join(precipdir,obs_fname)
            try:
                OBS = netCDF4.Dataset(obs_f)
            except RuntimeError:
                broken = 'ST4_missing'
            else:
                obs_data = OBS.variables['A_PCP_GDS5_SFC_acc6h'][:]
                # obs_data = N.swapaxes(OBS.variables['A_PCP_GDS5_SFC_acc6h'][:],1,0)

                if deltahour == 18:
                    stack = obs_data
                else:
                    stack = N.ma.dstack((obs_data,stack))

        if broken is False:
            ST4data_ST4grid = N.ma.sum(stack,axis=2).data

            # Interpolate onto our grid
            print("Starting ST4 interpolation for init time {0}".format(date))
            ST4data_SALgrid = griddata((ST4_SALgrid_xx.flat,ST4_SALgrid_yy.flat),
                        ST4data_ST4grid.flat,(SAL_native_mx.flat,SAL_native_my.flat)).reshape(308,537)  
            print("Done.")

            obplotproj = False
            # TEST
            if obplotproj:
                SALmap.drawcoastlines()
                SALmap.drawstates()
                data2 = data_SALgrid.reshape(308,537)
                SALmap.contourf(SAL_native_mx,SAL_native_my,data2,levels=N.arange(5,75,5)) 
                plt.savefig('/home/jrlawson/public_html/bowecho/test_ob_regrid_SAL.png')

            # Load NAM data for this time
            for fcsthr in (24,36):
                nam_fname = get_NAM_fname(date,fcsthr)
                nam_f = os.path.join(NAMdir,nam_fname)
                try:
                    NAM = netCDF4.Dataset(nam_f)
                except RuntimeError:
                    broken = 'NAM_missing'
                else:
                # nam_data = N.swapaxes(NAM.variables['A_PCP_218_SFC_acc12h'][:],1,0)
                    nam_data = NAM.variables['A_PCP_218_SFC_acc12h'][:]

                    if fcsthr == 24:
                        stack = nam_data
                    else:
                        stack = N.dstack((nam_data,stack))
        
        if broken is False:
            data_NAMgrid = N.sum(stack,axis=2)

            # Interpolate onto our grid
            print("Starting NAM interpolation for init time {0}".format(date))
            NAMdata_SALgrid = griddata((NAM_SALgrid_xx.flat,NAM_SALgrid_yy.flat),
                        data_NAMgrid.flat,(SAL_native_mx.flat,SAL_native_my.flat)).reshape(308,537)
            print("Done.")

            namplotproj = 0
            if namplotproj:
                # test
                SALmap.drawcoastlines()
                SALmap.drawstates()
                SALmap.contourf(SAL_native_mx,SAL_native_my,NAMdata_SALgrid,levels=N.arange(5,75,5)) 
                plt.savefig('/home/jrlawson/public_html/bowecho/test_NAM_regrid_SAL.png')

            # Compute SAL
            utc = date + datetime.timedelta(hours=36)
            sal = SAL(ST4data_SALgrid,NAMdata_SALgrid,'accum_precip',utc,accum_hr=24,
                        footprint=200,datafmt='array',dx=4.0,dy=4.0)
            # sal = SAL(ctrl_fpath,mod_fpath,'REFL_comp',utc)
            DATA[date] = {}
            DATA[date]['S'] = sal.S
            DATA[date]['A'] = sal.A
            DATA[date]['L'] = sal.L
        else:
            DATA[date] = {'note':broken}
            DATA[date]['S'] = -9999
            DATA[date]['A'] = -9999
            DATA[date]['L'] = -9999


        date = date + datetime.timedelta(days=1)

    # Save results to pickle for this day
    picklefname = 'SAL_{0}_{1}f_{2}fp.pickle'.format('accum_precip',7,200)
    picklef = os.path.join(pickledir,picklefname)
    pickle.dump(DATA, open(picklef, 'wb'))
