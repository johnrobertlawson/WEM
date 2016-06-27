import os
import datetime
import numpy as N
import numpy.lib.recfunctions as RF
from mpl_toolkits.basemap import Basemap
import mpl_toolkits.basemap as bmap
import copy
import matplotlib.pyplot as plt
import netCDF4
import glob
import matplotlib as M
from scipy.interpolate import griddata

from ecmwfapi import ECMWFDataServer

# Load all cases
# Skill rating and MCS types
# Further analysis to:
# (a) serial vs prog bow echo
# (b) see if BE mode was produced at all in WRF

GEFSdir = '/chinook2/jrlawson/bowecho/paper3_GEFSdata'
NAMdir = '/chinook2/jrlawson/bowecho/paper3_NAMdata'
ECMWFdir = '/chinook2/jrlawson/bowecho/paper3_ECMWFdata'
outdir = '/home/jrlawson/public_html/bowecho/paper3/'

baseurl = 'http://nomads.ncdc.noaa.gov/data/meso-eta-hi'
# gefslv = 100

download_GEFS_data = 0
compute_GEFS_std = 0
# compute_GEFS_stats = 1
download_ECMWF_data = 0
compute_ECMWF_Lyapunov = 0
skillplot = 0
# heat = False
# heat = 'spread'
# heat = 'skill'
heat = 'NAMskill'
download_NAM = 0
convert_NAM = 0

# FUNCTIONS #
def convert_to_nc(fpath_old):
    fpath_base, ext = os.path.splitext(fpath_old)
    fpath_new = fpath_base + '.nc'
    cmnd = 'ncl_convert2nc {0} -o {1}'.format(fpath_old,os.path.dirname(fpath_old))
    os.system(cmnd)
    return fpath_new

def ncks(typ,fpath_old,vrbl=False,Nlim=False,Elim=False,Slim=False,Wlim=False,
                t=False,lv=False):

    if typ == 'GEFSR2':
        latn = 'lat_0'
        lonn = 'lon_0'
        lvn = 'lv_ISBL0'
        tn = 'forecast_time0'

        if Wlim < 0:
            Wlim += 360.0
        if Elim < 0:
            Elim += 360.0

    fpath_base, ext = os.path.splitext(fpath_old)
    # fpath_new = fpath_base + '_ncks_TEST1.nc'
    fpath_new = fpath_base + '_ncks.nc'

    try:
        os.stat(fpath_new)
    except OSError:
        pass
    else:
        return 

    if isinstance(vrbl,(list,tuple)):
        vrblarg = '-v {0}'.format(','.join(vrbl))
    elif isinstance(vrbl,str):
        vrblarg = '-v {0}'.format(vrbl)
    else:
        vrblarg = ''

    if Nlim is not False:
        latarg = '-d {0},{1},{2}'.format(latn,Slim,Nlim)
    else:
        latarg = ''

    if Elim is not False:
        lonarg = '-d {0},{1},{2}'.format(lonn,Wlim,Elim)
    else:
        lonarg = ''

    if t is not False:
        if isinstance(t,(list,tuple)):
            targ = '-d {0},{1},{2}'.format(tn,t[0],t[1])
        elif isinstance(t,(int,float)):
            targ = '-d {0},{1}'.format(tn,t)
    else:
        targ = ''

    if lv is not False:
        if isinstance(lv,(list,tuple)):
            lvarg = '-d {0},{1},{2}'.format(lvn,lv[0],lv[1])
        if isinstance(lv,(float,int)):
            lvarg = '-d {0},{1}'.format(lvn,lv)
    else:
        lvarg = ''


    cmd = 'ncks {0} {1} {2} {3} {4} {5} {6}'.format(
        vrblarg,targ,lonarg,latarg,lvarg,fpath_old,fpath_new)
    os.system(cmd)
    return fpath_new

def get_nc(typ,date,member=False,vrbl=False,fcsthr=False):
    if typ=='GEFSR2':
        if vrbl is 'Z':
            vrblstr = 'hgt_pres'
        fname = '{0}_{1:04d}{2:02d}{3:02d}{4:02d}_{5}_ncks.nc'.format(
                vrblstr,date.year,date.month,date.day,date.hour,member)
        fpath = os.path.join(GEFSdir,fname)
    elif typ is 'NAM':
        if int(date.year) < 2008:
            fstr = 'f{0:03d}'.format(fcsthr)
        else:
            fstr = '{0:03d}'.format(fcsthr)

        fname = 'nam_218_{0:04d}{1:02d}{2:02d}_0000_{3}.nc'.format(
                date.year,date.month,date.day,fstr)
        fpath = os.path.join(NAMdir,fname)
    nc = netCDF4.Dataset(fpath)
    return nc

def lv_lookup(nctype,nc,lv):
    if nctype is 'GEFSR2':
        lvs = list(nc.variables['lv_ISBL0'][:])
        LV = lv * 100.0
    elif nctype is 'NAM':
        try:
            lvs = list(nc.variables['lv_ISBL2'][:])
        except KeyError:
            lvs = list(nc.variables['lv_ISBL0'][:])

        # lvs = [float(l) for l in lvs]
        # lv = float(l)

        if lvs[-1] > 5000.0:
            LV = lv * 100.0
        else:
            LV = lv

    lvidx = lvs.index(LV)
    # print(lvs,lv,lvidx)
    # import pdb; pdb.set_trace()    
    return lvidx

# def t_lookup(nc,t):
    # ts = list(


def delete_files(*args):
    for fpath in args:
        # answer = raw_input("Do you want to delete {0}? (Y/n)".format(fpath))
        # if answer == 'Y':
        os.system('rm -f {0}'.format(fpath))
    return

def download_GEFS(vrbl,date,member,coord='latlon'):
    baseurl = 'ftp://ftp.cdc.noaa.gov/Projects/Reforecast2'
    subdir = '{0:04d}/{0:04d}{1:02d}/{0:04d}{1:02d}{2:02d}{3:02d}/{4}/{5}'.format(
                        date.year,date.month,date.day,date.hour,member,coord)
    fname = '{0}_{1:04d}{2:02d}{3:02d}{4:02d}_{5}.grib2'.format(
                vrbl,date.year,date.month,date.day,date.hour,member)
    fpath = os.path.join(GEFSdir,fname)

    url = os.path.join(baseurl,subdir,fname)
    cmd = 'wget {0} -P {1}'.format(url,GEFSdir)
    os.system(cmd)
    return fpath

def download_ECMWF(date,days=2):
    date2 = date + datetime.timedelta(days=days-1)
    datestr = '{0:04d}-{1:02d}-{2:02d}/to/{3:04d}-{4:02d}-{5:02d}'.format(
                        date.year,date.month,date.day,
                        date2.year,date2.month,date2.day)
    ncstr = 'ECMWF_{0:04d}{1:02d}{2:02d}.nc'.format(
                        date.year,date.month,date.day)
    REQ = {}
    REQ['stream'] = 'oper'
    REQ['levtype'] = 'pl'
    REQ['param'] = 'z/u/v'
    REQ['dataset'] = 'interim'
    REQ['step'] = '0'
    REQ['grid'] = '1/1'
    REQ['time'] = '00/06/12/18'
    REQ['date'] = datestr
    REQ['type'] = 'an'
    REQ['class'] = 'ei'
    REQ['target'] = os.path.join(ECMWFdir,ncstr)
    REQ['format'] = 'netcdf'

    server = ECMWFDataServer()
    server.retrieve(REQ)

def get_ECMWF_nc(date):
    fname = 'ECMWF_{0:04d}{1:02d}{2:02d}.nc'.format(
            date.year,date.month,date.day)
    fpath = os.path.join(ECMWFdir,fname)
    return netCDF4.Dataset(fpath)

def get_ECMWF_vrbl(nc,hr=False,vrbl=False,lv=False,
                lats=False,lons=False):
    if vrbl == 'Z':
        ECv = 'z'

    if isinstance(lons,N.ndarray):
        if N.any(lons<180):
            lons = lons + 360.0

    lvidx = N.where(nc.variables['level'][:]==int(lv))[0][0]
    tidx = int(hr/6.0)

    minlonidx = N.where(nc.variables['longitude'][:]==lons.min())[0][0]
    maxlonidx = N.where(nc.variables['longitude'][:]==lons.max())[0][0]
    minlatidx = N.where(nc.variables['latitude'][:]==lats.min())[0][0]
    maxlatidx = N.where(nc.variables['latitude'][:]==lats.max())[0][0]
    # latidx = slice(

    # Longitude is 0 to 359
    # latitude is 90 to -90
    # data is time, level, lat, lon

    data = nc.variables[ECv][tidx,lvidx,maxlatidx:minlatidx+1,minlonidx:maxlonidx+1]
    # import pdb; pdb.set_trace()

    # 55,000 average Z
    # flip latitudes
    # return N.flipud(data)
    return data

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
    ext = fpath.split('.')[-1]
    fpath_new = fpath.replace(ext,'nc')
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

def regrid_NAM_data(nc,vrbl,h=False,lv=False):
    lvidx = lv_lookup('NAM',nc,lv)
    # import pdb; pdb.set_trace()

    if vrbl is 'Z':
        try:
            data = nc.variables['HGT_P0_L100_GLC0'][18,:,:]
        except:
            data = nc.variables['HGT_218_ISBL'][lvidx,:,:]

    dshp = (len(GEFSlons),len(GEFSlats))
    # dshp = (len(GEFSlats),len(GEFSlons))
    # regrid to GEFS/ECMWF 1 deg grid
    GEFSmlat,GEFSmlon = N.meshgrid(GEFSlats,GEFSlons)
    NAMmap, NAMlons, NAMlats, NAM_native_xx, NAM_native_yy = get_NAM_native_grid()
    NAMmx, NAMmy = N.meshgrid(NAM_native_xx,NAM_native_yy)

    xpts, ypts = NAMmap(GEFSmlon,GEFSmlat)

    # import pdb; pdb.set_trace()
    NAMdata_regrid = griddata((NAMmx.flat,NAMmy.flat),
                    data.flat,(xpts.flat,ypts.flat)).reshape(dshp[0],dshp[1])
                    # N.swapaxes(data,1,0).flat,(xpts.flat,ypts.flat)).reshape(dshp[0],dshp[1])
    return N.flipud(N.swapaxes(NAMdata_regrid,1,0))
    
def get_NAM_native_grid():
    m = Basemap(projection='lcc',llcrnrlon=-133.459,llcrnrlat=12.190,
            urcrnrlon=-49.420,urcrnrlat=57.328,lat_1=25.0,
            lon_0=-95.0)
    lons, lats, xx, yy = m.makegrid(614,428,returnxy=True)
    return m, lons, lats, xx[0,:], yy[:,0]


#### CASE LIST ####
fpath = 'paper3_climo.csv'
cases = N.genfromtxt(fpath,dtype=None,names=True,delimiter=',',)

# Generate list of dates here
datetimes = []
for y,m,d in zip(cases['Year'],cases['Month'],cases['Day']):
    datetimes.append(datetime.datetime(int(y),int(m),int(d),0,0,0))
cases = RF.append_fields(cases,'datetimes',N.array(datetimes))


# Define subdomain for CONUS
# 50 km spacing to interpolate NAM and ECMWF
# 3500 km N-S
# 4500 km W-E
# LL lat lon = (28.0,-130.0)
xx = N.arange(500,5050,50)*1000.0
yy = N.arange(500,3550,50)*1000.0
m = bmap.Basemap(projection='merc',llcrnrlat=20.0,llcrnrlon=-140.0,
                urcrnrlat=58.0,urcrnrlon=-70.0,lat_ts=40.0,
                resolution='h',area_thresh=1000)
mx, my = N.meshgrid(xx,yy)
lons, lats = m(mx,my,inverse=True)
Slim = N.floor(lats.min())
Nlim = N.ceil(lats.max())
Wlim = N.floor(lons.min())
Elim = N.ceil(lons.max())

GEFSlats = N.arange(Slim,Nlim+1.0,1.0)
GEFSlons = N.arange(Wlim,Elim+1.0,1.0)



mm = copy.copy(m)
mm.drawcountries()
mm.drawstates()
mm.drawcoastlines()
mm.drawmapboundary(fill_color='royalblue')
mm.fillcontinents(color='navajowhite',lake_color='royalblue')
# mm.drawparallels(N.arange(0,90,5))
# mm.plot([xx[0],]*len(yy),yy)
# mm.plot([xx[-1],]*len(yy),yy)
# mm.scatter(xx[0],yy[0])
# m.scatter(xx[-1],yy[0])
# m.scatter(xx[-1],yy[-1])
# m.scatter(xx[0],yy[-1])
# mm.drawgreatcircle(GEFSlons.min(),GEFSlats.min(),GEFSlons.min(),GEFSlats.max(),color='k')
# mm.drawgreatcircle(GEFSlons.min(),GEFSlats.min(),GEFSlons.max(),GEFSlats.min(),color='k')
# mm.drawgreatcircle(GEFSlons.max(),GEFSlats.max(),GEFSlons.min(),GEFSlats.max(),color='k')
# mm.drawgreatcircle(GEFSlons.max(),GEFSlats.max(),GEFSlons.max(),GEFSlats.min(),color='k')

pc = 'darkred'
lw = 2
mm.plot([GEFSlons.min()]*len(GEFSlats),GEFSlats,latlon=True,color=pc,lw=lw)
mm.plot([GEFSlons.max()]*len(GEFSlats),GEFSlats,latlon=True,color=pc,lw=lw)
mm.plot(GEFSlons,[GEFSlats.min(),]*len(GEFSlons),latlon=True,color=pc,lw=lw)
mm.plot(GEFSlons,[GEFSlats.max(),]*len(GEFSlons),latlon=True,color=pc,lw=lw)
plt.tight_layout()
plt.savefig(os.path.join(outdir,'paper3_grid.png'))
import pdb; pdb.set_trace()
# lvs = (100,300,500,700,850)
# lvlist = [l*100.0 for l in lvs]

# bmap.interp()

#### GEFS ####
gefsmembers = ['c00',] + ['p{0:02d}'.format(n) for n in range(1,11)]
vrbls = ['hgt_pres',]#'ugrd_pres','vgrd_pres']

if download_GEFS_data:
    for date in cases['datetimes']:
        print(("Date = {0}".format(date)))
        for member in gefsmembers:
            print(("Member = {0}".format(member)))
            for vrbl in vrbls:
                ncks_fname = '{0}_{1:04d}{2:02d}{3:02d}{4:02d}_{5}_ncks.nc'.format(
                        vrbl,date.year,date.month,date.day,date.hour,member)

                flist = glob.glob(os.path.join(GEFSdir,'*'))
                if os.path.join(GEFSdir,ncks_fname) in flist:
                    print("Already downloaded/ncksed")
                else:
                    print(("Variable = {0}".format(vrbl)))
                    grb_fpath = download_GEFS(vrbl,date,member,coord='latlon')
                    nc_fpath = convert_to_nc(grb_fpath)

                    # Kitchen sink extract
                    ncks_fpath = ncks('GEFSR2',nc_fpath,vrbl=False,Wlim=Wlim,Nlim=Nlim,Elim=Elim,
                                        Slim=Slim,t=(0,14),lv=False)

                    # Delete old file
                    # import pdb; pdb.set_trace()
                    delete_files(grb_fpath,nc_fpath)


#### ECMWF ####
# Get ECMWF Era as 'truth' at 0.75 deg
# For 00, 12, 24, 36 times

if download_ECMWF_data:
    for date in cases['datetimes']:
        download_ECMWF(date)
        # import pdb; pdb.set_trace()

# Compute 500 hPa differences between
# GEFS mean forecast and ERA Interim
# for each case

zoom = 0 
gefslv = 500
spr_list = {}
skl_list = {}
NAM_list = {}
list_t = (12,24,36)
for t in list_t:
    spr_list[t] = []
    skl_list[t] = []
    NAM_list[t] = []

if heat is not False:
    for heat in ('spread','skill','NAMskill'):
    # for heat in ('NAMskill',):
        HEAT = {}
        for gefslv in (500,): 
            if heat == 'spread':
                HEAT['times'] = N.arange(0,39,3)
                nctype = 'GEFSR2'
            elif heat == 'skill':
                HEAT['times'] = N.arange(0,42,6)
                nctype = 'GEFSR2'
            elif heat == 'NAMskill':
                HEAT['times'] = N.arange(0,48,12)
                nctype = 'NAM'

            if nctype is 'GEFSR2':
                pmembers = [m for m in gefsmembers if m is not 'c00']
            else:
                pmembers = ('det',)

            # count = 0
            for date in cases['datetimes']: 
                # count += 1
                # if count == 6:
                    # import pdb; pdb.set_trace()
                # if int(date.year) < 2008:
                    # continue
                print(("Computing {1} for {0}".format(date,heat)))
                HEAT[date] = {heat:[]}
                for h,fcsthr in enumerate(HEAT['times']): 
                    print(("Forecast hour {0}.".format(fcsthr)))
                    for member in pmembers:
                        nc = get_nc(nctype,date,member=member,vrbl='Z',fcsthr=False)
                        if nctype is 'GEFSR2':
                            lvidx = lv_lookup(nctype,nc,gefslv)

                        # if heat == 'spread':
                        if heat.endswith('skill'):
                            ECnc = get_ECMWF_nc(date)
                            ECz = get_ECMWF_vrbl(ECnc,vrbl='Z',hr=fcsthr,lv=gefslv,
                                lats=GEFSlats,lons=GEFSlons)/9.807

                        # if zoom:
                            # data = nc.variables['HGT_P1_L100_GLL0'][h,lvidx,5:-5,10:-10]
                        # else:
                        zstr = ''
                        if nctype is 'GEFSR2':
                            data = nc.variables['HGT_P1_L100_GLL0'][h,lvidx,:,:]

                            if member == 'p01':
                                stack = data
                            else:
                                stack = N.dstack((stack,data))

                        elif nctype is 'NAM':
                            data = regrid_NAM_data(nc,'Z',lv=gefslv)


                    if heat.endswith('skill'):
                        if heat is 'skill':
                            data = N.mean(stack,axis=2)
                        RMSE = N.sqrt(((ECz - data)**2).mean())
                        HEAT[date][heat].append(RMSE)
                        if fcsthr in list_t:
                            if heat is 'skill':
                                skl_list[fcsthr].append(RMSE)
                            else:
                                NAM_list[fcsthr].append(RMSE)

                        sanitycheck = False
                        if sanitycheck and date.year > 2007:
                            fig, ax = plt.subplots(2,1)
                            ax[0].pcolor(ECz)
                            ax[1].pcolor(data)
                            fig.savefig(os.path.join(outdir,'NAM_EC_gridcheck.png'))
                            import pdb; pdb.set_trace()
                    elif heat == 'spread':
                        std = N.std(stack,axis=2)
                        spread = (std.sum())/len(std)
                        HEAT[date]['spread'].append(spread)
                        if fcsthr in list_t:
                            spr_list[fcsthr].append(spread)

                if heat == 'spread':
                    ll = HEAT[date]['spread']
                    sprdiff = [j-i for i,j in zip(ll[:-1],ll[1:])] 
                    HEAT[date]['sprdiff'] = [s if s > 0.0 else 0 for s in sprdiff] 
                    del ll
                    HEAT[date]['sprdiff'].insert(0,0)

            relative = N.zeros([len(HEAT['times']),len(cases['datetimes'])])
            casesort = N.sort(cases.data,order='Score')
            for coln,date in enumerate(casesort['datetimes']):
                relative[:,coln] = HEAT[date][heat]

            rel_norm = (relative-N.mean(relative))/(relative.max()-relative.min())
            fig,ax = plt.subplots()
            # ax.plot()
            heatmap = ax.pcolor(N.swapaxes(rel_norm,1,0), cmap=M.cm.PuRd, alpha=0.8)

            timelabels = HEAT['times']
            # caselabels already done.
            OKb = ('S','P') 
            BElabels = [b if b in OKb else '' for b in casesort['BEtype']]
            ylabels = ["({2}) {0:%Y%m%d} = {1:.2f}".format(c,s,b) for c,s,b in 
                        zip(casesort['datetimes'],casesort['Score'],BElabels)]
            ax.set_xticks(N.arange(len(timelabels))+0.5)
            ax.set_xticklabels(timelabels, minor=False)
            ax.set_yticks(N.arange(len(casesort))+0.5)
            ax.set_yticklabels(ylabels,minor=False,fontsize=7)

            ax.set_xlim([0,len(timelabels)])
            ax.set_ylim([0,len(casesort)])

            # ax.set_yticks(N.arange(rel_norm.shape[0]) + 0.5, minor=False)
            # ax.set_xticks(N.arange(rel_norm.shape[1]) + 0.5, minor=False)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir,'{0}_heatmap_{1}hPa_{2}.png'.format(heat,gefslv,zstr)))
            plt.close(fig)

    # Plot skill vs spread at 30 h
    for whichskill in ('GEFSvEC','NAMvEC','GEFSvNAM'):
        fig, ax = plt.subplots()
        legpatch = []
        for t,c in zip(list_t,('b','g','r')): 
            if whichskill is 'GEFSvEC':
                ax.scatter(spr_list[t],skl_list[t],color=c,
                            edgecolor='k',linewidth=0.15,)
                ax.set_xlabel("GEFS/R2 Spread (m)")
                ax.set_ylabel("ECMWF-GEFS Error (m)")
            elif whichskill is 'NAMvEC':
                ax.scatter(NAM_list[t],skl_list[t],color=c,
                            edgecolor='k',linewidth=0.15,)
                ax.set_xlabel("ECMWF-NAM Error (m)")
                ax.set_ylabel("ECMWF-GEFS Error (m)")
            elif whichskill is 'GEFSvNAM':
                ax.scatter(spr_list[t],NAM_list[t],color=c,
                            edgecolor='k',linewidth=0.15,)
                ax.set_xlabel("GEFS/R2 Spread (m)")
                ax.set_ylabel("ECMWF-NAM Error (m)")

            legpatch.append(M.patches.Patch(color=c,label='{0} h'.format(t)))

        ax.set_axis_bgcolor('lightgrey')

        # ax.set_xlim([0,550])
        # ax.set_ylim([0,70])

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        ax.legend(handles=legpatch,
                bbox_to_anchor=(0.35, 0.9),
                bbox_transform=plt.gcf().transFigure)

        fig.tight_layout()
        fig.savefig(os.path.join(outdir,'spr_v_skl_{0}hPa_{1}.png'.format(gefslv,whichskill)))
        plt.close(fig)


#### NAM ####
# Get NAM archived forecasts for Snively
dirs = ['HAS010721020','HAS010721029','HAS010721031',
        'HAS010721033','HAS010721035']
NAMdir = '/chinook2/jrlawson/bowecho/paper3_NAMdata/'
baseurl = 'http://www1.ncdc.noaa.gov/pub/has/model'
import glob

if download_NAM:
    for n,d in enumerate(dirs):
        tarfs = []
        url = os.path.join(baseurl,d)
        cmd = 'wget -r -nH -P {0} --cut-dirs=4  --no-parent --reject "index.html*" {1}/'.format(NAMdir,url)
        # import pdb; pdb.set_trace()
        os.system(cmd)

if convert_NAM:
    tarfiles = glob.glob(os.path.join(NAMdir,'*.tar'))
    for tar in tarfiles:
        cmd = 'tar -xvf {0} -C {1}'.format(tar,NAMdir)
        os.system(cmd)
        delcmd = 'rm {0} -f'.format(tar)
        os.system(delcmd)

        fs2 = glob.glob(os.path.join(NAMdir,'*.grib'))
        ext = '.grib'
        if len(fs2) == 0:
            fs2 = glob.glob(os.path.join(NAMdir,'*.grb2'))
            ext = '.grb2'

        gribs = []
        for hr in ('00','12','24','36'):
            gribs.append([f for f in fs2 if f.endswith('{0}{1}'.format(hr,ext))][0])

        for fi in gribs:
            cmd = convert_nam(fi)

        # delfs = [f for f in fs2 if f not in gribs]
        for d in fs2:
            if d.endswith(ext):
                delcmd = 'rm {0} -f'.format(d)
                os.system(delcmd)


# Extract 500-hPa heights

# Delete old files


# Interpolate to grid for comparison with EC


#### STATS ####
# Compute ECMWF-NAM forecast error


# How do this correlate with MCS skill?



# How does GEFS spread correlate with MCS skill?



# How does GEFS spread correlate with Lyapunov?
# Mean, max, distribution... EOF?
# What do missing values mean?


# How does Lyapunov correlate with MCS skill?



# How do stats change with MCS mode?


# Spatial EOF with skill for synoptic regime



# Serial vs prog bow echoes and rising motion etc


# Plot domain(s)?
