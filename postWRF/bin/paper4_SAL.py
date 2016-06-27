import matplotlib as M
M.use('Agg')
import matplotlib.pyplot as plt
import datetime
import os
import numpy as N 
import pickle as pickle
import matplotlib as M
from mpl_toolkits.basemap import Basemap
import netCDF4
# from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.interpolate import griddata
from scipy.interpolate import SmoothBivariateSpline as SBS
import pdb
from ecmwfapi import ECMWFDataServer
import sys
sys.path.append(os.path.dirname(__file__))
# from cm_list import SALcm
from cm_list import viridis as SALcm
import time
import random

from WEM.postWRF.postWRF.sal import SAL
from WEM.postWRF.postWRF import WRFEnviron
import WEM.utils as utils
from WEM.postWRF.postWRF.scales import Scales
from WEM.postWRF.postWRF.obs import Radar

method1 = 1
method2 = 0
method3 = 0

if method1:
    mst = 'gdata_px_rand2'
elif method2:
    mst = 'RBS'
elif method3:
    mst = 'SBS'

download = 0
SAL_precip_compute = 0
SAL_cref_compute = 0
SAL_cref_random_compute = 0
SAL_climo_plot = 0
# climovrbl = 'accum_precip'
# accum_hr = 24
climovrbl = 'cref'
# plotthresh = 5
plotthresh = (5,15,30,40)
all_taSAL_plot = 0
plot_domain = 0
countsize = 0
active_pixels = 1

NAMdir = '/chinook2/jrlawson/bowecho/paper4_NAMdata'
radardir = '/chinook2/jrlawson/bowecho/paper4_radardata' 
precipdir = '/chinook2/jrlawson/bowecho/paper4_precipdata' 
pickledir = '/chinook2/jrlawson/bowecho/paper4_pickles'
outdir = '/home/jrlawson/public_html/bowecho/paper4/'
baseurl = 'http://nomads.ncdc.noaa.gov/data/meso-eta-hi'

startdate = datetime.datetime(2015,4,1,0,0,0)
# enddate = datetime.datetime(2015,4,11,0,0,0)
enddate = datetime.datetime(2015,9,1,0,0,0)
fhrs = N.arange(12,37,1)


def highResPoints(x,y,RESFACT=24,NPOINTS=24):
    import numpy as np
    '''
    Take points listed in two vectors and return them at a higher
    resultion. Create at least factor*len(x) new points that include the
    original points and those spaced in between.

    Returns new x and y arrays as a tuple (x,y).
    '''

    # r is the distance spanned between pairs of points
    r = [0]
    for i in range(1,len(x)):
        dx = x[i]-x[i-1]
        dy = y[i]-y[i-1]
        r.append(np.sqrt(dx*dx+dy*dy))
    r = np.array(r)

    # rtot is a cumulative sum of r, it's used to save time
    rtot = []
    for i in range(len(r)):
        rtot.append(r[0:i].sum())
    rtot.append(r.sum())

    dr = rtot[-1]/(NPOINTS*RESFACT-1)
    print(dr)
    xmod=[x[0]]
    ymod=[y[0]]
    rPos = 0 # current point on walk along data
    rcount = 1 
    while rPos < r.sum():
        x1,x2 = x[rcount-1],x[rcount]
        y1,y2 = y[rcount-1],y[rcount]
        dpos = rPos-rtot[rcount] 
        theta = np.arctan2((x2-x1),(y2-y1))
        rx = np.sin(theta)*dpos+x1
        ry = np.cos(theta)*dpos+y1
        xmod.append(rx)
        ymod.append(ry)
        rPos+=dr
        while rPos > rtot[rcount+1]:
            rPos = rtot[rcount+1]
            rcount+=1
            if rcount>rtot[-1]:
                break

    return xmod,ymod

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
    global Nlim
    Nlim = 45.0
    global Elim
    Elim = -85.0
    global Slim
    Slim = 34.0
    global Wlim
    Wlim = -110.0
    m = Basemap(projection='merc',llcrnrlat=Slim,llcrnrlon=Wlim,
                urcrnrlat=Nlim,urcrnrlon=Elim,lat_ts=39.5,resolution='h')
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


def plot_SAL(data,vrbl='accum_precip',hr='all'):

    cm = M.colors.ListedColormap(N.flipud(N.array(SALcm)))

    def scatplot(ax,v,SS,AA,LL):
        if (abs(v['S'])<2.0) and (abs(v['A'])<2.0) and (abs(v['L'])<2.0) and  (abs(v['S'])>0.0) and (abs(v['A'])>0.0) and (abs(v['L'])>0.0):
            sc= ax.scatter(v['S'],v['A'],c=v['L'],vmin=0,vmax=1,s=25,
                        cmap=cm,
                        # cmap=plt.cm.get_cmap('nipy_spectral_r'),
                        alpha=1.0,edgecolor='k',linewidth=0.15,
                        zorder=500)
            SS.append(v['S'])
            AA.append(v['A'])
            LL.append(v['L'])
        else:
            print(("Skipping due to missing data: {0}".format(k)))
            sc = None
        return sc, SS, AA, LL
        # ax.annotate(k[1:],xy=(v['S']+0.03,v['A']+0.03),xycoords='data',fontsize=5)

    if vrbl == 'cref' and hr == 'all':
        hrs = N.arange(12,36,1)
    else:
        hrs = (36,)

    for hr in hrs:
        print(('For hour {0}:'.format(hr)))
        SS = []
        AA = []
        LL = []

        # std/mode/mean stats?
        # EOF z-scores for covariance?

        fig = plt.figure(1,figsize=(5,5))
        ax = fig.add_subplot(111)
        # L_range = N.linspace(0,2,9)
        # colors = plt.cm.coolwarm(L_range)
        plt.axhline(0, color='k')
        plt.axvline(0, color='k')
        ax.set_xlim([-2,2])
        ax.set_ylim([-2,2])

        if vrbl == 'accum_precip':
            for k,v in list(DATA.items()):
                sc, SS, AA, LL = scatplot(ax,v,SS,AA,LL)
                if sc:
                    scs = sc
        elif vrbl == 'cref':
            for k,day in list(DATA.items()):
                sc,SS,AA,LL = scatplot(ax,day[hr],SS,AA,LL)
                if sc:
                    scs = sc

        # Square graph!
        ax.set_aspect('equal',adjustable='box')

        # Median S/A
        medS = N.median(SS)
        medA = N.median(AA)

        medkwargs = {'color':'black','zorder':300,
                'linewidth':0.8,'linestyle':':'}
        ax.axvline(medS,**medkwargs)#ls='b--')
        ax.axhline(medA,**medkwargs)#ls='b--')

        # Quartiles grey  box
        lbS = N.percentile(SS,25)
        ubS = N.percentile(SS,75)
        lbA = N.percentile(AA,25)
        ubA = N.percentile(AA,75)

        width = ubS-lbS
        height = ubA-lbA

        ax.add_patch(M.patches.Rectangle((lbS,lbA),width,height,
                            facecolor='white',alpha=1.0,linewidth=0.5,
                            zorder=100))

        ax.set_axis_bgcolor('lightgrey')
        ax.set_xlabel("Structural component")
        ax.set_ylabel("Amplitude component")
        cbax = fig.add_axes([0.17, 0.17, 0.22, 0.05])
        cblab = N.array([0.0,0.25,0.5,0.75,1.0,2.0])


        cb = plt.colorbar(scs,cax=cbax,
                    ticks=cblab,orientation='horizontal',)
        cb.set_label('Location component',labelpad=-38)
        cbax.set_xticklabels(cblab)
        fig.tight_layout()

        if len(hrs) > 1:
            hrstr = 'hr{0}'.format(hr)
        else:
            hrstr = ''

        if vrbl == 'cref':
            threshstr = '{0}dBZ'.format(plotthresh)
        else:
            threshstr = ''
        fig.savefig(os.path.join(outdir,'SAL_{0}_{1}{2}_{3}.png'.format(vrbl,threshstr,hrstr,mst)))
        plt.close(fig)
        # import pdb; pdb.set_trace()

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

if plot_domain:

    m = Basemap(projection='merc',llcrnrlat=30.0,llcrnrlon=-115.0,
                urcrnrlat=48.0,urcrnrlon=-80.0,lat_ts=39.5,resolution='h',
                area_thresh=1000)

    SALmap, lons,lats,Sxx,Syy = get_SAL_native_grid()
    xx,yy = m(lons,lats)
    m.drawmapboundary(fill_color='royalblue')
    m.fillcontinents(color='navajowhite',lake_color='royalblue')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    pc = 'darkred'
    lw = 2
    m.plot(xx[0,:],yy[0,:],pc,lw=lw)
    m.plot(xx[-1,:],yy[-1,:],pc,lw=lw)
    m.plot(xx[:,0],yy[:,0],pc,lw=lw)
    m.plot(xx[:,-1],yy[:,-1],pc,lw=lw)
    plt.gcf().tight_layout()
    plt.savefig('/home/jrlawson/public_html/bowecho/paper4/domains.png')

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
        for dn, deltahour in enumerate([18,24,30,36]):
            obs_fname = get_ST4_fname(date,deltahour)
            obs_f = os.path.join(precipdir,obs_fname)
            try:
                OBS = netCDF4.Dataset(obs_f)
            except RuntimeError:
                broken = 'ST4_missing'
            else:
                obs_data = OBS.variables['A_PCP_GDS5_SFC_acc6h'][:]
                # obs_data = N.swapaxes(OBS.variables['A_PCP_GDS5_SFC_acc6h'][:],1,0)

                if dn == 0:
                    stack = obs_data
                else:
                    stack = N.ma.dstack((obs_data,stack))

        if broken is False:
            ST4data_ST4grid = N.ma.sum(stack,axis=2).data


            """
            print("Starting radar ob interpolation for time {0}".format(utc))
            method1 = 0
            method2 = 1
            if method1:
                obsdata_SALgrid = griddata((obs_SALgrid_xx.flat,obs_SALgrid_yy.flat),
                        obdata_obgrid.flat,(SAL_native_mx.flat,SAL_native_my.flat)).reshape(308,537)  
            elif method2:
                rbs = RBS(obs_SALgrid_xx,obs_SALgrid_yy,obdata_obgrid) 
                obsdata_SALgrid = N.swapaxes(rbs(SAL_native_xx,SAL_native_yy),1,0)#grid=True)
            print("Done.")
            """


            # Interpolate onto our grid
            print(("Starting ST4 interpolation for init time {0}".format(date)))
            if method1:
                ST4data_SALgrid = griddata((ST4_SALgrid_xx.flat,ST4_SALgrid_yy.flat),
                        ST4data_ST4grid.flat,(SAL_native_mx.flat,SAL_native_my.flat),method='nearest').reshape(308,537)  
            elif method2:
                ST4rbs = RBS(ST4_SALgrid_xx,ST4_SALgrid_yy,ST4data_ST4grid)
                ST4data_SALgrid = N.swapaxes(ST4rbs(SAL_native_xx,SAL_native_yy),1,0)
            elif method3:
                ST4_F = SBS(ST4_SALgrid_xx.flatten(),ST4_SALgrid_yy.flatten(),ST4data_ST4grid.flatten())
                ST4data_SALgrid = ST4_F(SAL_native_xx,SAL_native_yy)
            print("Done.")
            # import pdb; pdb.set_trace()

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

            """
            print("Starting radar ob interpolation for time {0}".format(utc))
            method1 = 0
            method2 = 1
            if method1:
                obsdata_SALgrid = griddata((obs_SALgrid_xx.flat,obs_SALgrid_yy.flat),
                        obdata_obgrid.flat,(SAL_native_mx.flat,SAL_native_my.flat)).reshape(308,537)  
            elif method2:
                rbs = RBS(obs_SALgrid_xx,obs_SALgrid_yy,obdata_obgrid) 
                obsdata_SALgrid = N.swapaxes(rbs(SAL_native_xx,SAL_native_yy),1,0)#grid=True)
            print("Done.")
            """


            # Interpolate onto our grid
            print(("Starting NAM interpolation for init time {0}".format(date)))
            if method1:
                NAMdata_SALgrid = griddata((NAM_SALgrid_xx.flat,NAM_SALgrid_yy.flat),
                        data_NAMgrid.flat,(SAL_native_mx.flat,SAL_native_my.flat),method='nearest').reshape(308,537)
            if method2:
                NAMrbs = RBS(NAM_SALgrid_xx,NAM_SALgrid_yy)
                NAMdata_SALgrid = N.swapaxes(NAMrbs(SAL_native_xx,SAL_native_yy),1,0)
            print("Done.")
            # import pdb; pdb.set_trace()


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
    picklefname = 'SAL_{0}_{1}f_{2}fp_{3}.pickle'.format('accum_precip',7,200,mst)
    picklef = os.path.join(pickledir,picklefname)
    pickle.dump(DATA, open(picklef, 'wb'))

if SAL_cref_compute:
    hourarr = N.arange(12,36,1)
    # SAL (new) grid
    SALmap, SALlons, SALlats, SAL_native_xx, SAL_native_yy = get_SAL_native_grid()
    SAL_native_mx, SAL_native_my = N.meshgrid(SAL_native_xx,SAL_native_yy)

    NAMmap, NAMlons, NAMlats, nam_native_xx, nam_native_yy = get_NAM_native_grid()
    NAM_SALgrid_xx,NAM_SALgrid_yy = SALmap(NAMlons,NAMlats)

    for thresh in plotthresh:
    # for thresh in (15,):
        DATA = {}
        date = startdate
        # date = datetime.datetime(2015,4,8,0,0,0)
        while date < enddate:
            DATA[date] = {}
            # for deltahour in N.arange(24,36,1):
            ckpt1 = time.time()
            # for deltahour in N.arange(34,36,1):
            for deltahour in N.arange(12,36,1):
            # random.shuffle(hourarr)
            # for deltahour in hourarr[0:2]:
                DATA[date][deltahour] = {}
                utc = date + datetime.timedelta(hours=deltahour)

                # download, load radar data
                RADAR = Radar(utc,radardir)
                # ob_lats_raw = RADAR.lats
                # ob_lons_raw = RADAR.lons

                # Cut down radar data to zone of interest
                RADAR.get_subdomain(Nlim=Nlim,Elim=Elim,Slim=Slim,
                                    Wlim=Wlim,overwrite=True)

                ob_lats = RADAR.lats[::-1]
                ob_lons = RADAR.lons
                ob_mlat, ob_mlon = N.meshgrid(ob_lats,ob_lons)
                obs_SALgrid_mx = SALmap(ob_mlon,ob_mlat)[0]
                obs_SALgrid_xx = SALmap(ob_mlon,ob_mlat)[0][:,0]
                obs_SALgrid_my = SALmap(ob_mlon,ob_mlat)[1]
                obs_SALgrid_yy = SALmap(ob_mlon,ob_mlat)[1][0,:]
                #### I SWAPPED [1] AND [0] HERE FoR XX/yy
                #### Also swapped [a,b] to [b,a]
                # obs_SALgrid_xx, obs_SALgrid_yy = SALmap(ob_lons,ob_lats)

                # obdata_obgrid = N.swapaxes(RADAR.get_dBZ(data='self'),1,0)
                # obdata_obgrid = N.flipud(N.swapaxes(RADAR.get_dBZ(data='self'),1,0))
                obdata_obgrid = N.fliplr(N.swapaxes(RADAR.get_dBZ(data='self'),1,0))
                # obdata_obgrid = N.fliplr(RADAR.get_dBZ(data='self'))

                # import pdb; pdb.set_trace()

                # THIS METHOD2 DEFINITELY WORKS
                print(("Starting radar ob interpolation for time {0}".format(utc)))
                # if method1:
                    # This requires a nan_to_num method
                    # obsdata_SALgrid = griddata((obs_SALgrid_mx.flat,obs_SALgrid_my.flat),
                            # obdata_obgrid.flat,(SAL_native_mx.flat,SAL_native_my.flat)).reshape(308,537)  
                # elif method2:
                OBSrbs = RBS(obs_SALgrid_xx,obs_SALgrid_yy,obdata_obgrid) 
                obsdata_SALgrid = N.swapaxes(OBSrbs(SAL_native_xx,SAL_native_yy),1,0)#grid=True)
                print("Done.")

                testtest = 0
                if testtest:
                    fig, ax = plt.subplots(2,figsize=(4,6))
                    S = Scales('cref',False)
                    clvs = S.clvs
                    cmap = S.cm
                    origdata = N.swapaxes(obdata_obgrid,1,0)
                    cbf = ax.flat[1].pcolormesh(origdata,cmap=cmap,vmin=5.0,vmax=70.0)
                    ax.flat[0].axis([0,200,0,200])
                    cbf.cmap.set_under("white")
                    ax.flat[1].pcolormesh(obsdata_SALgrid[300:500,0:200],cmap=cmap,vmin=5.0,vmax=70.0)
                    ax.flat[0].set_aspect('equal')
                    ax.flat[1].set_aspect('equal')
                    ax.flat[1].axis([0,200,0,200])
                    plt.colorbar(cbf,orientation='horizontal')
                    plt.tight_layout()
                    plt.savefig('/home/jrlawson/public_html/bowecho/interp_test5') 
                    # import pdb; pdb.set_trace()


                # NAM data
                nam_fname = get_NAM_fname(date,fcsthr=deltahour)
                nam_f = os.path.join(NAMdir,nam_fname)
                try:
                    NAM = netCDF4.Dataset(nam_f)
                except RuntimeError:
                    DATA[date][deltahour] = {'S':-9999,
                                            'A':-9999,
                                            'L':-9999}
                else:
                # nam_data = N.swapaxes(NAM.variables['A_PCP_218_SFC_acc12h'][:],1,0)
                    data_NAMgrid = NAM.variables['REFC_218_EATM'][:]

                    # Interpolate onto our grid
                    print(("Starting NAM interpolation for fcst time {0}".format(utc)))
                    if method1:
                        NAMdata_SALgrid = griddata((NAM_SALgrid_xx.flat,NAM_SALgrid_yy.flat),
                                data_NAMgrid.flat,(SAL_native_mx.flat,SAL_native_my.flat),method='linear').reshape(308,537)
                    elif method2:
                        raise Exception

                    testgrid = 0
                    if testgrid:
                        NAM_nn = griddata((NAM_SALgrid_xx.flat,NAM_SALgrid_yy.flat),
                                data_NAMgrid.flat,(SAL_native_mx.flat,SAL_native_my.flat),method='nearest').reshape(308,537)
                        NAM_lin = NAMdata_SALgrid
                        fig, ax = plt.subplots(4,figsize=(4,6))
                        S = Scales('cref',False)
                        clvs = S.clvs
                        cmap = S.cm
                        ax.flat[0].pcolormesh(data_NAMgrid[150:310,250:375],cmap=cmap,vmin=5.0,vmax=70.0)
                        cbf = ax.flat[1].pcolormesh(NAM_lin,cmap=cmap,vmin=5.0,vmax=70.0)
                        ax.flat[0].axis([0,180,0,110])
                        cbf.cmap.set_under("white")
                        ax.flat[2].pcolormesh(NAM_nn,cmap=cmap,vmin=5.0,vmax=70.0)
                        ax.flat[0].set_aspect('equal')
                        ax.flat[1].set_aspect('equal')
                        ax.flat[2].set_aspect('equal')
                        ax.flat[1].axis([0,537,0,308])
                        ax.flat[2].axis([0,537,0,308])
                        ax.flat[3].axis('off')
                        plt.colorbar(cbf,orientation='horizontal')
                        plt.tight_layout()
                        plt.savefig('/home/jrlawson/public_html/bowecho/interp_test6.png')
                    # print("Done.")

                    # import pdb; pdb.set_trace()
                    testplot1 = 0
                    testplot2 = 0
                    if testplot1:
                        SALmap.drawcoastlines()
                        SALmap.drawstates()
                        SALmap.contourf(SAL_native_mx,SAL_native_my,obsdata_SALgrid,levels=N.arange(5,70,5)) 
                        plt.gcf().tight_layout()
                        plt.savefig('/home/jrlawson/public_html/bowecho/test_obcref_SAL_actpx.png')
                        import pdb; pdb.set_trace()
                    if testplot2:
                        SALmap.drawcoastlines()
                        SALmap.drawstates()
                        SALmap.contourf(SAL_native_mx,SAL_native_my,NAMdata_SALgrid,levels=N.arange(5,70,5)) 
                        plt.gcf().tight_layout()
                        plt.savefig('/home/jrlawson/public_html/bowecho/test_NAMcref_SAL_actpx.png')
                        import pdb; pdb.set_trace()

                    sal = SAL(obsdata_SALgrid,NAMdata_SALgrid,'cref',utc,thresh=thresh,
                                footprint=200,ctrl_fmt='array',
                                mod_fmt='array',dx=4.0,dy=4.0)
                    # import pdb; pdb.set_trace()
                    DATA[date][deltahour]['S'] = sal.S
                    DATA[date][deltahour]['A'] = sal.A
                    DATA[date][deltahour]['L'] = sal.L
                    DATA[date][deltahour]['active_px_obs'] = sal.active_px('ctrl')
                    DATA[date][deltahour]['active_px_NAM'] = sal.active_px('mod')
                    print(('Active pixels for obs and NAM data:\n {0}% and {1}%.'.format(
                            DATA[date][deltahour]['active_px_obs'],
                            DATA[date][deltahour]['active_px_NAM'])))
                    # import pdb; pdb.set_trace()

            ckpt2 = time.time()
            print(("One day took this long: {0}".format(ckpt2-ckpt1)))
            date = date + datetime.timedelta(days=1)

        # Save results to pickle for this day
        picklefname = 'SAL_{0}_{1}dBZ_{2}fp_{3}.pickle'.format('cref',thresh,200,mst)
        picklef = os.path.join(pickledir,picklefname)
        pickle.dump(DATA, open(picklef, 'wb'))



if SAL_cref_random_compute:
    # SAL (new) grid
    SALmap, SALlons, SALlats, SAL_native_xx, SAL_native_yy = get_SAL_native_grid()
    SAL_native_mx, SAL_native_my = N.meshgrid(SAL_native_xx,SAL_native_yy)

    NAMmap, NAMlons, NAMlats, nam_native_xx, nam_native_yy = get_NAM_native_grid()
    NAM_SALgrid_xx,NAM_SALgrid_yy = SALmap(NAMlons,NAMlats)

    # Generate random matching of times/dates


    for thresh in (plotthresh,):
        DATA = {}
        date = startdate
        while date < enddate:
            DATA[date] = {}
            # for deltahour in N.arange(24,36,1):
            for deltahour in N.arange(12,36,1):
                DATA[date][deltahour] = {}
                utc = date + datetime.timedelta(hours=deltahour)

                # download, load radar data
                RADAR = Radar(utc,radardir)
                ob_lats = RADAR.lats[::-1]
                ob_lons = RADAR.lons
                ob_mlat, ob_mlon = N.meshgrid(ob_lats,ob_lons)
                obs_SALgrid_xx = SALmap(ob_mlon,ob_mlat)[0][:,0]
                obs_SALgrid_yy = SALmap(ob_mlon,ob_mlat)[1][0,:]
                # obs_SALgrid_xx, obs_SALgrid_yy = SALmap(ob_lons,ob_lats)

                # obdata_obgrid = N.swapaxes(RADAR.get_dBZ(data='self'),1,0)
                # obdata_obgrid = N.flipud(N.swapaxes(RADAR.get_dBZ(data='self'),1,0))
                obdata_obgrid = N.fliplr(N.swapaxes(RADAR.get_dBZ(data='self'),1,0))

                print(("Starting radar ob interpolation for time {0}".format(utc)))
                if method1:
                    obsdata_SALgrid = griddata((obs_SALgrid_xx.flat,obs_SALgrid_yy.flat),
                            obdata_obgrid.flat,(SAL_native_mx.flat,SAL_native_my.flat)).reshape(308,537)  
                elif method2:
                    OBSrbs = RBS(obs_SALgrid_xx,obs_SALgrid_yy,obdata_obgrid) 
                    obsdata_SALgrid = N.swapaxes(OBSrbs(SAL_native_xx,SAL_native_yy),1,0)#grid=True)
                print("Done.")


                # NAM data
                nam_fname = get_NAM_fname(date,fcsthr=deltahour)
                nam_f = os.path.join(NAMdir,nam_fname)
                try:
                    NAM = netCDF4.Dataset(nam_f)
                except RuntimeError:
                    DATA[date][deltahour] = {'S':-9999,
                                            'A':-9999,
                                            'L':-9999}
                else:
                # nam_data = N.swapaxes(NAM.variables['A_PCP_218_SFC_acc12h'][:],1,0)
                    data_NAMgrid = NAM.variables['REFC_218_EATM'][:]

                    # Interpolate onto our grid
                    print(("Starting NAM interpolation for fcst time {0}".format(utc)))
                    if method1:
                        NAMdata_SALgrid = griddata((NAM_SALgrid_xx.flat,NAM_SALgrid_yy.flat),
                                data_NAMgrid.flat,(SAL_native_mx.flat,SAL_native_my.flat)).reshape(308,537)
                    elif method2:
                        raise Exception
                    print("Done.")

                    testplot1 = 1
                    testplot2 = 0
                    if testplot1:
                        SALmap.drawcoastlines()
                        SALmap.drawstates()
                        # SALmap.contourf(SAL_native_mx,SAL_native_my,obsdata_SALgrid,levels=N.arange(5,70,5)) 
                        plt.gcf().tight_layout()
                        plt.savefig('/home/jrlawson/public_html/bowecho/test_obcref_SAL.png')
                    if testplot2:
                        SALmap.drawcoastlines()
                        SALmap.drawstates()
                        SALmap.contourf(SAL_native_mx,SAL_native_my,NAMdata_SALgrid,levels=N.arange(5,70,5)) 
                        plt.savefig('/home/jrlawson/public_html/bowecho/test_NAMcref_SAL.png')
                        import pdb; pdb.set_trace()

                    sal = SAL(obsdata_SALgrid,NAMdata_SALgrid,'cref',utc,thresh=thresh,
                                footprint=200,ctrl_fmt='array',
                                mod_fmt='array',dx=4.0,dy=4.0)
                    DATA[date][deltahour]['S'] = sal.S
                    DATA[date][deltahour]['A'] = sal.A
                    DATA[date][deltahour]['L'] = sal.L

            date = date + datetime.timedelta(days=1)

        # Save results to pickle for this day
        picklefname = 'SAL_{0}_{1}dBZ_{2}fp_{3}.pickle'.format('cref',thresh,200,mst)
        picklef = os.path.join(pickledir,picklefname)
        pickle.dump(DATA, open(picklef, 'wb'))

if SAL_climo_plot:
    # Load data
    if climovrbl == 'accum_precip':
        picklefname = 'SAL_{0}_{1}f_{2}fp_{3}.pickle'.format('accum_precip',7,200,mst)
    elif climovrbl == 'cref':
        picklefname = 'SAL_{0}_{1}dBZ_{2}fp_{3}.pickle'.format('cref',plotthresh,200,mst)
    picklef = os.path.join(pickledir,picklefname)
    DATA = pickle.load(open(picklef, 'rb'))

    plot_all = 1
    covar_plot = 0
    traj_plot = 0

    if plot_all:
        plot_SAL(DATA,climovrbl)

    if covar_plot:
        SALS = {}
        for thresh in (5,15,30):
            picklefname = 'SAL_{0}_{1}dBZ_{2}fp_{3}.pickle'.format('cref',plotthresh,200,mst)
            picklef = os.path.join(pickledir,picklefname)
            SALS[thresh] = pickle.load(open(picklef, 'rb'))
        ss = []
        aa = []
        ll = []
        date = startdate
        while date < enddate:
            for t in N.arange(12,36,1):
                for thresh in (5,15,30):
                    SALS[thresh][date][t]['S']
                if t == 35:
                    date = date + datetime.timedelta(days=1)


    if traj_plot:
        S_meds = {}
        A_meds = {}
        L_meds = {}
        TRAJ = {}
        for th in (5,15,30,40):
            print(("Thresh {0}".format(th)))
            TRAJ[th] = {}
            picklefname = 'SAL_{0}_{1}dBZ_{2}fp_{3}.pickle'.format('cref',th,200,mst)
            picklef = os.path.join(pickledir,picklefname)
            DATA = pickle.load(open(picklef, 'rb'))
            TRAJ[th]['S_med'] = []
            TRAJ[th]['A_med'] = []
            TRAJ[th]['L_med'] = []
            for hr in N.arange(12,36,1):
                SS = []
                AA = []
                LL = []
                for k,day in list(DATA.items()):
                    # Compute median S,A,L for each time/threshold
                    S = day[hr]['S']
                    A = day[hr]['A']
                    L = day[hr]['L']

                    if (abs(S)<2.0) and (abs(A)<2.0) and (abs(L)<2.0) and  (abs(S)>0.0) and (abs(A)>0.0) and (abs(L)>0.0):
                        SS.append(S)
                        AA.append(A)
                        LL.append(L)

                TRAJ[th]['S_med'].append(N.median(SS))
                TRAJ[th]['A_med'].append(N.median(AA))
                TRAJ[th]['L_med'].append(N.median(LL))

        fig = plt.figure(1,figsize=(6,3))
        ax = fig.add_subplot(111)
        # L_range = N.linspace(0,2,9)
        # colors = plt.cm.coolwarm(L_range)
        plt.axhline(0, color='k')
        plt.axvline(0, color='k')
        ax.set_xlim([0.3,1.7])
        ax.set_ylim([0.4,1.1])

        term_mark = 0
        scattersw = 1
        # Plot all on a trajectory diagram
        # handles = []
        # labels = []
        legpatch = []
        for th,color in zip((5,15,30),('#E69F00','#56B4E9','#009E73')):
        # for th,color in zip((5,15,30,40),('r','k','b','g')):
            legpatch.append(M.patches.Patch(color=color,label='{0} dBZ'.format(th)))

            HiS_meds,HiA_meds = highResPoints(TRAJ[th]['S_med'],TRAJ[th]['A_med'])
            MAP = 'cool'
            # plotcm = plt.get_cmap(MAP)

            npointsHiRes = len(HiS_meds)
            # ax.set_color_cycle([plotcm(1.0*i/(npointsHiRes-1)) for i in range(npointsHiRes-1)])        
            for i,alphi in zip(list(range(npointsHiRes-1)),N.linspace(0.10,1.00,num=npointsHiRes+1)):
                ax.plot(HiS_meds[i:i+2],HiA_meds[i:i+2],
                     # alpha=float(i)/(npointsHiRes-1),
                     alpha=alphi,
                     color=color)
                    # )
            if term_mark:
                ax.scatter(HiS_meds[0],HiA_meds[0],c=color,marker='s',
                                s=25,edgecolor='k',linewidth=0.15,zorder=500)
            # ax.plot(HiS_meds,HiA_meds)
            # fig.savefig(os.path.join(outdir,'SAL_trajectories.png'))
            # assert True==False

            if scattersw:
                cm = M.colors.ListedColormap(N.flipud(N.array(SALcm)))
                scs = ax.scatter(TRAJ[th]['S_med'],TRAJ[th]['A_med'],c=TRAJ[th]['L_med'],
                            vmin=0.15,vmax=0.25,s=25,
                            cmap=cm, alpha=1.0,edgecolor='k',linewidth=0.15,
                            zorder=500)

        ax.set_aspect('equal',adjustable='box')
        ax.set_axis_bgcolor('lightgrey')
        ax.set_xlabel("Structural component")
        ax.set_ylabel("Amplitude component")

        if scattersw:
            cbax = fig.add_axes([0.17, 0.83, 0.22, 0.05])
            cblab = N.arange(0.1,0.5,0.05)
            cb = plt.colorbar(scs,cax=cbax,
                        ticks=cblab,orientation='horizontal',)
            cb.set_label('Location component',labelpad=-38)
            cbax.set_xticklabels(cblab)

        if term_mark:
            ax.legend(handles=legpatch,
                    bbox_to_anchor=(0.35, 0.9),
                    bbox_transform=plt.gcf().transFigure)
            # ax.legend(handles=handles,labels=labels)

        if scattersw:
            fname = 'SAL_trajSAL'
        else:
            fname = 'SAL_trajSA'
        fig.tight_layout()
        fig.savefig(os.path.join(outdir,'{0}_{1}.png'.format(fname,mst)))
        plt.close(fig)

        fig = plt.figure(1,figsize=(6,3))
        ax = fig.add_subplot(111)
        xx = N.arange(12,36,1)
        for th,facecolor,linecolor in zip((5,15,30),
                    ('#E69F00','#56B4E9','#009E73'),
                    ('#b37b00','#1d9be2','#00664a')):
            taSAL = []
            for S,A,L in zip(TRAJ[th]['S_med'],TRAJ[th]['A_med'],TRAJ[th]['L_med']):
                taSAL.append(abs(S) + abs(A) + abs(L))
            ax.plot(xx,taSAL,color=linecolor,lw=2.0)
            ax.fill_between(xx,1.0,taSAL,facecolor=facecolor)

        ax.legend(handles=legpatch,
                bbox_to_anchor=(0.85, 0.9),
                bbox_transform=plt.gcf().transFigure)

        ax.set_xlim([12,35])
        ax.set_ylim([1.0,3.0])

        ax.set_xticks(N.arange(12,36,3))

        plt.axvline(24, color='k')
       
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')


        # ax.set_axis_bgcolor('lightgrey')
        ax.set_xlabel("Forecast hour")
        ax.set_ylabel("Total absolute SAL error")

        fig.tight_layout()
        fname = 'taSAL_{0}'.format(mst)
        fig.savefig(os.path.join(outdir,fname))

if countsize:
    COUNT = {}
    for th in (5,15,30,40,'acc'):
        if th == 'acc':
            picklefname = 'SAL_{0}_{1}f_{2}fp_{3}.pickle'.format('accum_precip',7,200,mst)
        else:
            picklefname = 'SAL_{0}_{1}dBZ_{2}fp_{3}.pickle'.format('cref',th,200,mst)
        picklef = os.path.join(pickledir,picklefname)
        DATA = pickle.load(open(picklef, 'rb'))
        COUNT[th] = {}
        COUNT[th]['OK'] = 0
        COUNT[th]['missing'] = 0
        COUNT[th]['SAL0'] = 0
        COUNT[th]['SAL2'] = 0

        for k,day in list(DATA.items()):
            for hr in N.arange(12,36,1):
                # Compute median S,A,L for each time/thold
                if th == 'acc':
                    S = day['S']
                    A = day['A']
                    L = day['L']
                else:
                    S = day[hr]['S']
                    A = day[hr]['A']
                    L = day[hr]['L']

                if S == -9999:
                    COUNT[th]['missing'] += 1
                elif (abs(S)<2.0) and (abs(A)<2.0) and (abs(L)<2.0) and  (abs(S)>0.0) and (abs(A)>0.0) and (abs(L)>0.0):
                    COUNT[th]['OK'] += 1
                elif (abs(S)==0.0) or (abs(A)==0.0) or (abs(L)==0.0): 
                    COUNT[th]['SAL0'] += 1
                elif (abs(S)==2.0) or (abs(A)==2.0) or (abs(L)==2.0): 
                    COUNT[th]['SAL2'] += 1

if active_pixels:
    # Count number of active pixels in obs and NAM cref per hour
    # Save to pickle file
    # Generate plot similar to Fig 5 (taSAL) to add underneath

    legpatch = [0,]*6
    ll = {5:('',''),30:('',''),15:('Obs','NAM')}
    for thresh,color,ln in zip((5,15,30),(('#E69F00','#f2cf7f'),
                            ('#56B4E9','#aad9f4'),('#009E73','#7fceb9')),list(range(3))):
        # for th,color in zip((5,15,30,40),('r','k','b','g')):
        legpatch[ln] = M.patches.Patch(color=color[0],label=ll[thresh][0])
        legpatch[ln+3] = M.patches.Patch(color=color[1],label=ll[thresh][1])
    # for thresh in plotthresh:
        # import pdb; pdb.set_trace()
        if thresh == 40:
            continue
        pfname = '/chinook2/jrlawson/bowecho/paper4_pickles/SAL_cref_{0}dBZ_200fp_gdata_px_rand2.pickle'.format(thresh)
        randomdata = pickle.load(open(pfname,'rb'))
        # import pdb; pdb.set_trace()

        scatter_px = 0
        if scatter_px:
            fig,ax = plt.subplots(1,figsize=(5,5))
            for d in list(randomdata.keys()):
                for h in list(randomdata[d].keys()):
                    try:
                        ax.scatter(randomdata[d][h]['active_px_obs'],
                            randomdata[d][h]['active_px_NAM'],
                            alpha=1.0,edgecolor='k',linewidth=0.15)
                    except KeyError:
                        pass
            ax.set_aspect('equal',adjustable='box')
            fig.tight_layout()
            plt.savefig(os.path.join(outdir,'scatter_{0}dBZ_active_px.png'.format(thresh)))
            plt.close(fig)

        fig = plt.figure(1,figsize=(6,3))
        ax = fig.add_subplot(111)
        xx = N.arange(12,36,1)

        obpx_AVE = []
        NAMpx_AVE = []
        for h in N.arange(12,36,1):
            hr_obpx_l = []
            hr_NAMpx_l = []
            for d in list(randomdata.keys()):
                try:
                    hr_obpx_l.append(randomdata[d][h]['active_px_obs'])
                except KeyError:
                    pass
                else:
                    hr_NAMpx_l.append(randomdata[d][h]['active_px_NAM'])
            obpx_AVE.append(N.mean(hr_obpx_l))
            NAMpx_AVE.append(N.mean(hr_NAMpx_l))

        # import pdb; pdb.set_trace()
        w = 0.3
        ax.grid(zorder=1)
        b1 = ax.bar(xx-w/2.0,obpx_AVE,width=w,color=color[0],alpha=1.0,align='center',label='Obs',zorder=100)
        b2 = ax.bar(xx+w/2.0,NAMpx_AVE,width=w,color=color[1],alpha=1.0,align='center',label='NAM',zorder=101)

        # legpatch = [M.patches.Patch(color='black',label='Obs'),
                     # M.patches.Patch(color='lightgrey',label='NAM')]
    # import pdb; pdb.set_trace()
    ax.legend(handles=legpatch,ncol=2,loc=5,
            bbox_to_anchor=(0.4, 0.1),
            bbox_transform=plt.gcf().transFigure,
            borderaxespad=0.0)

    ax.set_xlim([11,36])
    # ax.set_ylim([1.0,3.0])

    ax.set_xticks(N.arange(12,36,3))

    # plt.axvline(24, color='k')
   
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('top')

    ax.invert_yaxis()


    # ax.set_axis_bgcolor('lightgrey')
    # ax.set_xlabel("Forecast hour")
    ax.set_ylabel("Average active pixels (%)")

    fig.tight_layout()
    fname = 'activepx_bar_{0}'.format(mst)
    fig.savefig(os.path.join(outdir,fname))
