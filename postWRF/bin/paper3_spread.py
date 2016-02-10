import os
import datetime
import numpy as N
import numpy.lib.recfunctions as RF
import mpl_toolkits.basemap as bmap
import copy
import matplotlib.pyplot as plt
import netCDF4
import glob
import matplotlib as M
from ecmwfapi import ECMWFDataServer

# Load all cases
# Skill rating and MCS types
# Further analysis to:
# (a) serial vs prog bow echo
# (b) see if BE mode was produced at all in WRF

GEFSdir = '/chinook2/jrlawson/bowecho/paper3_GEFSdata'
NAMdir = '/chinook2/jrlawson/bowecho/paper3_NAMdata'
ECMWFdir = '/chinook2/jrlawson/bowecho/paper3_ECMWFdata'
outdir = '/home/jrlawson/public_html/bowecho/'

download_GEFS_data = 0
compute_GEFS_std = 1
# compute_GEFS_stats = 1
download_ECMWF_data = 0
compute_ECMWF_Lyapunov = 0
skillplot = 0

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

def get_nc(typ,date,member=False,vrbl=False):
    if typ=='GEFSR2':
        fname = '{0}_{1:04d}{2:02d}{3:02d}{4:02d}_{5}_ncks.nc'.format(
                vrbl,date.year,date.month,date.day,date.hour,member)
        fpath = os.path.join(GEFSdir,fname)
    nc = netCDF4.Dataset(fpath)
    return nc


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
                urcrnrlat=58.0,urcrnrlon=-70.0,lat_ts=40.0)
mx, my = N.meshgrid(xx,yy)
lons, lats = m(mx,my,inverse=True)
Slim = N.floor(lats.min())
Nlim = N.ceil(lats.max())
Wlim = N.floor(lons.min())
Elim = N.ceil(lons.max())

GEFSlats = N.arange(Slim,Nlim+1.0,1.0)
GEFSlons = N.arange(Wlim,Elim+1.0,1.0)

# mymap = copy.copy(m)
# m.drawcountries()
# m.drawstates()
# m.drawcoastlines()
# m.plot([xx[0],]*len(yy),yy)
# m.plot([xx[-1],]*len(yy),yy)
# m.scatter(xx[0],yy[0])
# m.scatter(xx[-1],yy[0])
# m.scatter(xx[-1],yy[-1])
# m.scatter(xx[0],yy[-1])
# plt.savefig(os.path.join(outdir,'paper3_grid.png'))
# import pdb; pdb.set_trace()

# bmap.interp()

#### GEFS ####
gefsmembers = ['c00',] + ['p{0:02d}'.format(n) for n in range(1,11)]
vrbls = ['hgt_pres',]#'ugrd_pres','vgrd_pres']

if download_GEFS_data:
    for date in cases['datetimes']:
        print("Date = {0}".format(date))
        for member in gefsmembers:
            print("Member = {0}".format(member))
            for vrbl in vrbls:
                ncks_fname = '{0}_{1:04d}{2:02d}{3:02d}{4:02d}_{5}_ncks.nc'.format(
                        vrbl,date.year,date.month,date.day,date.hour,member)

                flist = glob.glob(os.path.join(GEFSdir,'*'))
                if os.path.join(GEFSdir,ncks_fname) in flist:
                    print("Already downloaded/ncksed")
                else:
                    print("Variable = {0}".format(vrbl))
                    grb_fpath = download_GEFS(vrbl,date,member,coord='latlon')
                    nc_fpath = convert_to_nc(grb_fpath)

                    # Kitchen sink extract
                    ncks_fpath = ncks('GEFSR2',nc_fpath,vrbl=False,Wlim=Wlim,Nlim=Nlim,Elim=Elim,
                                        Slim=Slim,t=(0,14),lv=50000.0)

                    # Delete old file
                    # import pdb; pdb.set_trace()
                    delete_files(grb_fpath,nc_fpath)

# Compute RM differences control vs perts
if compute_GEFS_std:
    SPREAD = {}
    SPREAD['times'] = N.arange(0,39,3)
    for date in cases['datetimes']:
        print("Computing spread for {0}".format(date))
        SPREAD[date] = {'spread':[]}
        for h,fcsthr in enumerate(SPREAD['times']):
            for member in gefsmembers:
                if member == 'c00':
                    continue
                nc = get_nc('GEFSR2',date,member=member,vrbl='hgt_pres')
                data = nc.variables['HGT_P1_L100_GLL0'][h,0,:,:]
                # data = nc.variables['HGT_P1_L100_GLL0'][h,0,5:-5,10:-10]
                if member == 'p01':
                    stack = data
                else:
                    stack = N.dstack((stack,data))
            std = N.std(stack,axis=2)
            spread = (std.sum())/len(std)
            # SPREAD[date][fcsthr] = spread
            SPREAD[date]['spread'].append(spread)


    relative = N.zeros([len(SPREAD['times']),len(cases['datetimes'])])

    # scoreidx =  cases.dtype.names.index('Score')
    casesort = N.flipud(N.sort(cases,order='Score'))
    # import pdb; pdb.set_trace()
    for coln,date in enumerate(casesort['datetimes']):
        relative[:,coln] = SPREAD[date]['spread']

    # SOMETHING WRONG WITH SORTING HERE?
    rel_norm = (relative-N.mean(relative))/(relative.max()-relative.min())
    fig,ax = plt.subplots()
    # ax.plot()
    heatmap = ax.pcolor(N.swapaxes(rel_norm,1,0), cmap=M.cm.PuRd, alpha=0.8)

    timelabels = SPREAD['times']
    # caselabels already done.
    ylabels = ["{0:%Y%m%d} = {1:.2f}".format(c,s) for c,s in zip(casesort['datetimes'],casesort['Score'])]
    ax.set_xticks(N.arange(len(timelabels))+0.5)
    ax.set_xticklabels(timelabels, minor=False)
    ax.set_yticks(N.arange(len(casesort))+0.5)
    ax.set_yticklabels(ylabels,minor=False,fontsize=7)

    ax.set_xlim([0,len(timelabels)])
    ax.set_ylim([0,len(casesort)])

    # ax.set_yticks(N.arange(rel_norm.shape[0]) + 0.5, minor=False)
    # ax.set_xticks(N.arange(rel_norm.shape[1]) + 0.5, minor=False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir,'spread_heatmap.png'))
    # fig.savefig(os.path.join(outdir,'spread_heatmap_zoom1.png'))

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

if skillplot:
    SKILL = {}
    SKILL['times'] = N.arange(0,42,6)
    for date in cases['datetimes']:
        print("Computing skill for {0}".format(date))
        SKILL[date] = {'skill':[]}
        ECnc = get_ECMWF_nc(date)
        for h,fcsthr in enumerate(SKILL['times']):
            ECz = get_ECMWF_vrbl(ECnc,vrbl='Z',hr=fcsthr,lv=500,
                            lats=GEFSlats,lons=GEFSlons)/9.807
            for member in gefsmembers:
                if member == 'c00':
                    continue
                nc = get_nc('GEFSR2',date,member=member,vrbl='hgt_pres')
                data = nc.variables['HGT_P1_L100_GLL0'][h,0,:,:]
                # data = nc.variables['HGT_P1_L100_GLL0'][h,0,5:-5,10:-10]

                if fcsthr == 30 and False==True:
                    fig,ax = plt.subplots(1,2)
                    ax[0].pcolor(ECz)
                    ax[1].pcolor(data)
                    fig.savefig(os.path.join(outdir,'GEFSvECMWF_test.png'))
                    import pdb; pdb.set_trace()

                if member == 'p01':
                    stack = data
                else:
                    stack = N.dstack((stack,data))
            mean = N.mean(stack,axis=2)

            RMSE = N.sqrt(((ECz - data)**2).mean())

            SKILL[date]['skill'].append(RMSE)


    relative = N.zeros([len(SKILL['times']),len(cases['datetimes'])])

    # scoreidx =  cases.dtype.names.index('Score')
    casesort = N.flipud(N.sort(cases,order='Score'))
    # import pdb; pdb.set_trace()
    for coln,date in enumerate(casesort['datetimes']):
        relative[:,coln] = SKILL[date]['skill']

    # SOMETHING WRONG WITH SORTING HERE?
    rel_norm = (relative-N.mean(relative))/(relative.max()-relative.min())
    fig,ax = plt.subplots()
    # ax.plot()
    heatmap = ax.pcolor(N.swapaxes(rel_norm,1,0), cmap=M.cm.YlOrRd, alpha=0.8)

    timelabels = SKILL['times']
    # caselabels already done.
    ylabels = ["{0:%Y%m%d} = {1:.2f}".format(c,s) for c,s in zip(casesort['datetimes'],casesort['Score'])]
    ax.set_xticks(N.arange(len(timelabels))+0.5)
    ax.set_xticklabels(timelabels, minor=False)
    ax.set_yticks(N.arange(len(casesort))+0.5)
    ax.set_yticklabels(ylabels,minor=False,fontsize=7)

    ax.set_xlim([0,len(timelabels)])
    ax.set_ylim([0,len(casesort)])

    # ax.set_yticks(N.arange(rel_norm.shape[0]) + 0.5, minor=False)
    # ax.set_xticks(N.arange(rel_norm.shape[1]) + 0.5, minor=False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir,'skill_heatmap.png'))

#### NAM ####
# Get NAM archived forecasts for Snively

# Convert to netCDF

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
