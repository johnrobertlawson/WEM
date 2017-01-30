#import scipy.ndimage as nd
from netCDF4 import Dataset
import calendar
import collections
import fnmatch
import math
import matplotlib as M
import numpy as N
import os
import pdb
import sys
import time
import glob
import pickle as pickle
from . import unix_tools as utils
import datetime
import heapq

from . import getdata

def decompose_wind(wspd,wdir,convert=0):
    # Split wind speed/wind direction into u,v
    if (type(wspd) == N.array) & (type(wdir) == N.array):
        uwind = N.array([-s * N.sin(N.radians(d)) if ((s>-1)&(d>-1)) else -9999
                    for s,d in zip(wspd,wdir)])
        vwind = N.array([-s * N.cos(N.radians(d)) if ((s>-1)&(d>-1)) else -9999
                    for s,d in zip(wspd,wdir)])
    else:
        uwind = -wspd * N.sin(N.radians(wdir))
        vwind = -wspd * N.cos(N.radians(wdir))
    if convert == 'ms_kt':
        uwind *= 1.94384449
        vwind *= 1.94384449
    elif convert == 'ms_mph':
        uwind *= 2.23694
        vwind *= 2.23694
    elif convert == 'kt_ms':
        uwind *= 0.51444444
        vwind *= 0.51444444
    else:
        pass
    return uwind, vwind

def combine_wind_components(u,v):
    wdir = N.degrees(N.arctan2(u,v)) + 180
    wspd = N.sqrt(u**2 + v**2)
    return wspd, wdir

def convert_kt2ms(wspd):
    wspd_ms = wspd*0.51444
    return wspd_ms

def dewpoint(T,RH): # Lawrence 2005 BAMS?
    #T in C
    #RH in 0-100 format
    es = 6.11 * (10**((7.5*T)/(237.7+T)))
    e = es * RH/100.0
    alog = 0.43429*N.log(e) - 0.43429*N.log(6.11)
    Td = (237.7 * alog)/(7.5-alog)
    #pdb.set_trace()
    return Td

def csvprocess(data,names,convert=0):
    # Get stations
    stationlist = N.unique(data['stid']) # Stations in the record
    stnum = len(stationlist) # Number of them


    # Create dictionary from data
    D = {} # Initialise dictionary of data
    for s in stationlist:
        D[s] = {} # Initialise dictionary of station/s
        print('Loading station data for ' + s)
        those = N.where(data['stid']==s) # Indices of obs
        obNum = len(those[0]) # Number of obs

        # Entries from MesoWest data
        for n in names:
            D[s][n] = data[n][those]

        # Sort times
        year = [int(t[0:4]) for t in D[s]['tutc']]
        month = [int(t[4:6]) for t in D[s]['tutc']]
        day = [int(t[6:8]) for t in D[s]['tutc']]
        hour = [int(t[9:11]) for t in D[s]['tutc']]
        minute = [int(t[11:13]) for t in D[s]['tutc']]

        # Create python time
        D[s]['pytime'] = N.empty(obNum)
        for i in range(0,obNum-1):
            D[s]['pytime'][i] = cal.timegm([year[i],month[i],day[i],hour[i],minute[i],0])

        # Convert to S.I. units
        if convert == 1:
            D[s]['tmpc'] = (5/9.0) * (D[s]['tmpf']-32)
            D[s]['dptc'] = (5/9.0) * (D[s]['dptf']-32)
            D[s]['wsms'] = D[s]['wskn']*0.51444
            D[s]['wgms'] = D[s]['wgkn']*0.51444

        ### DATA PROCESSING
        # Find time of biggest wind increase (DSWS initiation)

        D[s]['dVdt'] = N.zeros(obNum) # This is our array of d(wind)/d(time)
        for i in range(0,obNum-1):
            if i == 0:
                pass # First record is zero as there's no gradient yet
            else:
                D[s]['dVdt'][i] = ((D[s]['wsms'][i] - D[s]['wsms'][i-1])/
                                    (D[s]['pytime'][i] - D[s]['pytime'][i-1]))
        if any(D[s]['dVdt']) == False:
            # Data were absent (-9999) or rubbish (0)
            D[s]['mwi'] = -9999
            D[s]['mwi_t'] = -9999
        else:
            D[s]['mwi'] = D[s]['dVdt'].max() # max wind increase
            loc = N.where(D[s]['dVdt']==D[s]['mwi'])
            D[s]['mwi_t'] = D[s]['pytime'][loc][0] # time of max wind increase

        # Find time of maximum wind gust
        if any(D[s]['wgms']) == False:
            D[s]['mg'] = -9999
            D[s]['mg_t'] = -9999
        else:
            D[s]['mg'] = D[s]['wgms'].max() # Maximum gust at the station
            loc = N.where(D[s]['wgms']==D[s]['mg'])
            D[s]['mg_t'] = D[s]['pytime'][loc][0] # time of max gust

        # Find time of maximum wind speed
        if any(D[s]['wsms']) == False:
            D[s]['ms'] = -9999
            D[s]['ms_t'] = -9999
        else:
            D[s]['ms'] = D[s]['wsms'].max() # Maximum gust at the station
            loc = N.where(D[s]['wsms']==D[s]['ms'])
            D[s]['ms_t'] = D[s]['pytime'][loc][0] # time of max gust


        # Frequency of observation in minutes
        try:
            D[s]['dt'] = (D[s]['pytime'][1] - D[s]['pytime'][0]) / 60
        except IndexError:
            D[s]['dt'] = -9999

        # Find lowest pressures
        try:
            D[s]['lowp'] = D[s]['alti'].min()
        except IndexError:
            D[s]['lowp'] = -9999
            D[s]['lowp_t'] = -9999
        finally:
            if D[s]['lowp'] != -9999:
                D[s]['lowp_t'] = N.where(D[s]['alti']==D[s]['lowp'])
            else:
                pass

    return D

# WRF terrain in height above sea level
def WRFterrain(fileselection='control',dom=1):
    if fileselection=='nouintah':
        fname = '/uufs/chpc.utah.edu/common/home/horel-group/lawson/wrfout/1/NAM/2011112906_nouintah/wrfout_d0'+str(dom)+'_2011-11-29_06:00:00'
    elif fileselection=='withuintah':
        fname = '/uufs/chpc.utah.edu/common/home/horel-group/lawson/wrfout/1/NAM/2011112906_withuintah/wrfout_d0'+str(dom)+'_2011-11-29_06:00:00'
    else:
        fname = '/uufs/chpc.utah.edu/common/home/horel-group/lawson/WRFV3/test/em_real/wrfout_1.3_wasatch/wrfout_d0'+str(dom)+'_2011-12-01_00:00:00'
    nc = Dataset(fname,'r')
    terrain = nc.variables['HGT'][0,:,:]
    xlong = nc.variables['XLONG'][0]
    xlat = nc.variables['XLAT'][0]
    return terrain, xlong, xlat

# WRF terrain in pressure coords...from first time stamp (won't vary THAT much for a tropospheric plot!)
# SLICE keyword, when true, gives 1-D vector through middle of square.

def WRFterrain_P(fileselection='control',dom=1,slice=0):
    if fileselection=='nouintah':
        fname = '/uufs/chpc.utah.edu/common/home/horel-group/lawson/wrfout/1/NAM/2011112906_nouintah/wrfout_d0'+str(dom)+'_2011-11-29_06:00:00'
    elif fileselection=='withuintah':
        fname = '/uufs/chpc.utah.edu/common/home/horel-group/lawson/wrfout/1/NAM/2011112906_withuintah/wrfout_d0'+str(dom)+'_2011-11-29_06:00:00'
    else:
        fname = '/uufs/chpc.utah.edu/common/home/horel-group/lawson/WRFV3/test/em_real/wrfout_1.3_wasatch/wrfout_d0'+str(dom)+'_2011-12-01_00:00:00'
    nc = Dataset(fname,'r')
    terrain = nc.variables['PSFC'][0,:,:]
    xlong = nc.variables['XLONG'][0]
    xlat = nc.variables['XLAT'][0]
    if slice==1:
        Nx = nc.getncattr('WEST-EAST_GRID_DIMENSION')-1
        Ny = nc.getncattr('SOUTH-NORTH_GRID_DIMENSION')-1
        xlong = xlong[Ny/2,:]
        xlat = xlat[:,Nx/2]
    return terrain, xlong, xlat

# Constants
rE = 6378100 # radius of Earth in metres

# Settings
# plt.rc('text',usetex=True)
# fonts = {'family':'Computer Modern','size':16}
# plt.rc('font',**fonts)
# height, width = (9,17)

# outdir = '/uufs/chpc.utah.edu/common/home/u0737349/public_html/thesis/topoxs/'

# Functions
# First, if topography data is not in memory, load it
# try:
    # dataloaded
# except NameError:
    # topodata, lats, lons = gridded_data.gettopo()
    # dataloaded = 1

def get_map(Nlim,Elim,Slim,Wlim):
    ymax,xmax = gridded_data.getXY(lats,lons,Nlim,Elim) #Here, x is lat, y is lon
    ymin,xmin = gridded_data.getXY(lats,lons,Slim,Wlim) # Not sure why!
    terrain = topodata[xmin:xmax,ymin:ymax]
    xlat = lats[xmin:xmax]
    xlon = lons[ymin:ymax]
    #pdb.set_trace()
    return terrain,xlat,xlon

def get_cross_section(Alat, Alon, Blat, Blon):
    # Now find cross-sections
    Ax, Ay = gridded_data.getXY(lats,lons,Alat,Alon)
    Bx, By = gridded_data.getXY(lats,lons,Blat,Blon)

    # Number of points along cross-section
    xspt = int(N.hypot(Bx-Ax,By-Ay))
    xx = N.linspace(Ay,By,xspt).astype(int)
    yy = N.linspace(Ax,Bx,xspt).astype(int)

    # Get terrain heights along this transect
    heights = topodata[xx,yy]

    # Work out distance along this cross-section in m
    xsdistance = gridded_data.xs_distance(Alat,Alon,Blat,Blon)

    xvals = N.linspace(0,xsdistance,xspt)
    xlabels = ['%3.1f' %(x/1000) for x in xvals]
    # Now plot cross-sections
    fig = plt.figure(figsize=(width,height))
    plt.plot(xvals,heights)
    delta = xspt/10
    plt.xticks(xvals[::delta],xlabels[::delta])
    plt.xlabel('Distance along cross-section (km)')
    fname = 'test1.png'
    plt.savefig(outdir+fname,bbox_inches='tight',pad_inches=0.3)
    plt.clf()

# datestring needs to be format yyyymmddhh
def wrf_nc_load(dom,var,ncfolder,datestr,thin,Nlim=0,Elim=0,Slim=0,Wlim=0):
    datestr2 = datestr[0:4]+'-'+datestr[4:6]+'-'+datestr[6:8]+'_'+datestr[8:10]+':00:00'
    fname = ncfolder+'wrfout_'+dom+'_'+datestr2
    try:
        nc = Dataset(fname,'r')
    except RuntimeError:
        fname += '.nc'
    finally:
        nc = Dataset(fname,'r')
    # Load in variables here - customise or automate perhaps?
    u10 = nc.variables['U10']
    v10 = nc.variables['V10']
    terrain = nc.variables['HGT'][0,:,:]
    times = nc.variables['Times']

    ### PROCESSING
    # x_dim and y_dim are the x and y dimensions of the model
    # domain in gridpoints
    x_dim = len(nc.dimensions['west_east'])
    y_dim = len(nc.dimensions['south_north'])

    # Get the grid spacing
    dx = float(nc.DX)
    dy = float(nc.DY)

    width_meters = dx * (x_dim - 1)
    height_meters = dy * (y_dim - 1)

    cen_lat = float(nc.CEN_LAT)
    cen_lon = float(nc.CEN_LON)
    truelat1 = float(nc.TRUELAT1)
    truelat2 = float(nc.TRUELAT2)
    standlon = float(nc.STAND_LON)

    # Draw the base map behind it with the lats and
    # lons calculated earlier

    if Nlim:
        m = Basemap(projection='lcc',lon_0=cen_lon,lat_0=cen_lat,
                    llcrnrlat=Slim,urcrnrlat=Nlim,llcrnrlon=Wlim,
                    urcrnrlon=Elim,rsphere=6371200.,resolution='h',
                    area_thresh=100)
    else:
        m = Basemap(resolution='i',projection='lcc',
            width=width_meters,height=height_meters,
            lat_0=cen_lat,lon_0=cen_lon,lat_1=truelat1,
            lat_2=truelat2)
    xlong = nc.variables['XLONG'][0]
    xlat = nc.variables['XLAT'][0]
    #xlong = xlongtot[N.where((xlongtot < Elim) & (xlongtot > Wlim))]
    #xlat = xlattot[N.where((xlattot < Nlim) & (xlattot > Slim))]
    #x,y = m(nc.variables['XLONG'][0],nc.variables['XLAT'][0])
    #px,py = m(xlongtot,xlattot)
    #x,y = N.meshgrid(px,py)
    x,y = m(xlong,xlat)
    # This sets a thinned out grid point structure for plotting
    # wind barbs at the interval specified in "thin"
    #x_th,y_th = m(nc.variables['XLONG'][0,::thin,::thin],\
    #        nc.variables['XLAT'][0,::thin,::thin])
    x_th, y_th = m(xlong[::thin, ::thin],xlat[::thin,::thin])
    data = (times,terrain,u10,v10)
    return m, x, y, x_th, y_th, data, nc

def mkloop(dom='d03',fpath='./'):
    print("Creating a pretty loop...")
    os.system('convert -delay 50 '+fpath+dom+'*.png > -loop '+fpath+dom+'_windloop.gif')
    print("Finished!")

def p_interpol(dom,var,ncfolder,datestr):
    datestr2 = datestr[0:4]+'-'+datestr[4:6]+'-'+datestr[6:8]+'_'+datestr[8:10]+':00:00'
    fname = ncfolder+'wrfout_'+dom+'_'+datestr2
    nc = Dataset(fname,'r')
    var_data = nc.variables[var][:]
    if var == 'T': # pert. theta
        var_data += 300 # Add base state
    pert_pressure = nc.variables['P'][:] #This is perturbation pressure
    base_pressure = nc.variables['PB'][:] # This is base pressure
    pressure = pert_pressure + base_pressure # Get into absolute pressure in Pa.
    p_levels = N.arange(10000,100000,1000)
    var_interp = N.zeros((pressure.shape[0],len(p_levels),pressure.shape[2],pressure.shape[3]))
    #pdb.set_trace()
    for (t,y,x),v in N.ndenumerate(var_interp[:,0,:,:]):
        var_interp[t,:,y,x] = N.interp(p_levels,pressure[t,::-1,y,x],var_data[t,::-1,y,x])
    return var_interpi

def hgt_from_sigma(nc):
    vert_coord = (nc.variables['PH'][:] + nc.variables['PHB'][:]) / 9.81
    return vert_coord

# This converts wrf's time to python and human time
def find_time_index(wrftime,reqtimetuple,tupleformat=1):
    # wrftime = WRF Times in array
    # reqtime = desired time in six-tuple

    # Convert required time to Python time if required
    if tupleformat:
        reqtime = calendar.timegm(reqtimetuple)
    else:
        reqtime = reqtimetuple
    nt = wrftime.shape[0]
    pytime = N.zeros([nt,1])
    t = wrftime
    # Now convert WRF time to Python time
    for i in range(nt):
        yr = int(''.join(t[i,0:4]))
        mth = int(''.join(t[i,5:7]))
        day = int(''.join(t[i,8:10]))
        hr = int(''.join(t[i,11:13]))
        min = int(''.join(t[i,14:16]))
        sec = int(''.join(t[i,17:19]))
        pytime[i] = calendar.timegm([yr,mth,day,hr,min,sec])

    # Now find closest WRF time
    timeInd = N.where(abs(pytime-reqtime) == abs(pytime-reqtime).min())[0][0]
    return timeInd

# Find data for certain time, locations, level, variables
# For surface data
def TimeSfcLatLon(nc,varlist,times,latslons='all'):
    # Time (DateTime in string)
    if times == 'all':
        timeInds = list(range(nc.variables['Times'].shape[0]))
    elif len(times)==1: # If only one time is desired
        # Time is in 6-tuple format
        timeInds = find_time_index(nc.variables['Times'],times) # This function is from this module
    elif len(times)==2: # Find all times between A and B
        timeIndA = find_time_index(nc.variables['Times'],times[0])
        timeIndB = find_time_index(nc.variables['Times'],times[1])
        timeInds = list(range(timeIndA,timeIndB))
    # Lat/lon of interest and their grid pointd
    lats = nc.variables['XLAT'][:]
    lons = nc.variables['XLONG'][:]
    if latslons == 'all':
        latInds = list(range(lats.shape[-2]))
        lonInds = list(range(lons.shape[-1]))
    else:
        xmin,ymax = gridded_data.getXY(lats,lons,Nlim,Wlim)
        xmax,ymin = gridded_data.getXY(lats,lons,Slim,Elim)
        latInds = list(range(ymin,ymax))
        lonInds = list(range(xmin,xmax))

    # Return sliced data
    data = {}
    for v in varlist:
        data[v] = nc.variables[v][timeInds,latInds,lonInds]
        # Reshape if only one time
        if len(times)==1:
            data[v] = N.reshape(data[v],(len(latInds),len(lonInds)))
    return data

# Get lat/lon as 1D arrays from WRF
def latlon_1D(nc):
    Nx = nc.getncattr('WEST-EAST_GRID_DIMENSION')-1
    Ny = nc.getncattr('SOUTH-NORTH_GRID_DIMENSION')-1
    lats = nc.variables['XLAT'][0,:,Nx/2]
    lons = nc.variables['XLONG'][0,Ny/2,:]
    return lats, lons

def netcdf_files_in(folder,dom=1,init_time=0,model='auto',return_model=False):
    """
    Hunt through given folder to find the right netcdf file for data.


    Inputs:
    folder      :   Absolute path to directory
    dom         :   specify domain. None specified if zero.
    init_time   :   initialisation time. Can be tuple or datenum.
                    If zero, then folder must contain one unambiguous file.
    model       :   Default: automatically detect the type of netcdf file
                    (RUC data, wrfout file, etc)
    Returns:
    ncpath      :   Absolute path to file
    model       :   Model (RUC, WRF) of netcdf file,
                    if return_model is True and model is 'auto'.
    """
    t = 'auto'
    if init_time:
        t = ensure_timetuple(init_time,fmt='single')

    # Set the model type to load.
    if model=='auto':
        # Get files, check prefix
        files = glob.glob(os.path.join(folder,'*'))
        matches = 0
        model_test = []
        for f in files:
            model_test.append(determine_model(f.split('/')[-1]))
            # print(model_test)
            model_set = set(model_test)
            # import pdb; pdb.set_trace()
            # if model_test:
                # matches += 1
                # model = model_test

        model_set.discard(False)
        matches = len(model_set)
        # import pdb; pdb.set_trace()
        if matches < 1:
            print("No netcdf files found.")
            raise Exception
        elif matches > 1 and isinstance(t,str):
            print("Ambiguous netcdf file selection. Specify model?")
            raise Exception
        else:
            model = list(model_set)[0]

    # import pdb; pdb.set_trace()
    if model=='wrfout':
        pfx = 'wrfout' # Assume the prefix
    elif model=='ruc':
        pfx = getdata.RUC_fname(t,filetype='netcdf')[:7]
        # TODO: logic that considers the four versions of RUC
    else:
        raise Exception

    # import pdb; pdb.set_trace()
    # Pick unambiguous
    if t=='auto':
        # We assume the user has wrfout files in different folders for different times
        f = glob.glob(os.path.join(folder,pfx+'*'))
        # import pdb; pdb.set_trace()
        if len(f) != 1:
            print("Ambiguous netCDF4 selection.")
            raise Exception
        else:
            if return_model:
                return f[0], model
            else:
                return f[0]
    else:
        if (dom > 8):
            print("Domain is out of range. Choose number between 1 and 8 inclusive.")
            raise IndexError

        fname = get_netcdf_naming(model,t,dom)
        f = glob.glob(os.path.join(folder,fname))

        if len(f) == 1:
            if return_model:
                return f[0], model
            else:
                return f[0]
        elif len(f) == 0:
            print("No netCDF4 file found.")
            raise Exception
        else:
            print("Ambiguous netCDF4 selection.")
            raise Exception


def wrfout_files_in(folders,dom=0,init_time='notset',descend=1,avoid=0,
                    unambiguous=0):
    """
    Hunt through given folder(s) to find all occurrences of wrfout
    files.

    Inputs:
    folders     :   list of absolute paths to directories
    dom         :   specify domain. None specified if zero.
    init_time   :   tuple of initialisation time
    descend     :   boolean: go into subfolders
    avoid       :   string of filenames. if a subfolder contains
                    the string, do not descend into this one.
    unambiguous :   only return a single absolute path, else throw
                    an Exception.

    Returns:
    wrfouts     :   list of absolute paths to wrfout files
    """

    folders = get_sequence(folders)
    avoids = []
    if 'avoid':
        # Avoid folder names with this string
        # or list of strings
        avoid = get_sequence(avoid)
        for a in avoid:
            avoids.append('/{0}/'.format(a))


    w = 'wrfout' # Assume the prefix
    if init_time=='notset':
        suffix = '*0'
        # We assume the user has wrfout files in different folders for different times
    else:
        try:
            it = utils.string_from_time('wrfout',init_time)
        except:
            print("Not a valid wrfout initialisation time; try again.")
            raise Error
        suffix = '*' + it

    if not dom:
    # Presume all domains are desired.
        prefix = w + '_d'
    elif (dom > 8):
        print("Domain is out of range. Choose number between 1 and 8 inclusive.")
        raise IndexError
    else:
        dom = 'd{0:02d}'.format(dom)
        prefix = w + '_' + dom

    wrfouts = []
    if descend:
        for folder in folders:
            for root,dirs,files in os.walk(folder):
                for fname in fnmatch.filter(files,prefix+suffix):
                    skip_me = 0
                    fpath = os.path.join(root,fname)
                    if avoids:
                        for a in avoids:
                            if a in fpath:
                                skip_me = 1
                    else:
                        pass
                    if not skip_me:
                        wrfouts.append(fpath)

    else:
        for folder in folders:
            findfile = os.path.join(folder,prefix+suffix)
            files = glob.glob(findfile)
            # pdb.set_trace()
            for f in files:
                wrfouts.append(os.path.join(folder,f))
    # pdb.set_trace()
    if unambiguous:
        if not len(wrfouts) == 1:
            print(("Found {0} wrfout files.".format(len(wrfouts))))
            raise Exception
        else:
            return wrfouts[0]
    else:
        return wrfouts

def getXY(lats,lons,ptlat,ptlon):
    """
    Output is lat, lon so y,x
    """
    # Find closest lat/lon in array
    minlat = abs(lats-ptlat).min()
    minlon = abs(lons-ptlon).min()
    # Find where these are in the grid
    wherelat = N.where(abs(lats-ptlat) == minlat)
    wherelon = N.where(abs(lons-ptlon) == minlon)
    # pdb.set_trace()
    lat_idx = N.where(lats==lats[wherelat])[0][0]
    lon_idx = N.where(lons==lons[wherelon])[0][0]
    exactlat = lats[wherelat]
    exactlon = lons[wherelon]
    return lat_idx,lon_idx, exactlat, exactlon

def gettopo():
    fname = '/uufs/chpc.utah.edu/common/home/u0737349/dsws/topodata/globe30.bin'
    f = open(fname,'r')
    fdata = N.fromfile(f,dtype='int16')
    # Transposes and reshapes to a lat-lon grid
    # Changes negative values to 0 (sea level)
    xnum = 43200.0
    ynum = 18000.0
    topodata = N.flipud(N.reshape(fdata,(ynum,xnum))).clip(0)
    #topodata = ((N.reshape(fdata,(xnum,ynum))).clip(0))
    f.close(); del fdata
    # Define size of pixels
    xpixel = 360/xnum
    ypixel = 150/ynum # Note only 150 degrees!
    # Create lat/lon grid
    lats = N.arange(-60,90,ypixel)#[::-1]
    lons = N.arange(-180,180,xpixel)#[::-1]
    print('Topographic data has been loaded. Everest is but a mere pixel.')
    return topodata, lats, lons

def xs_distance(Alat, Alon, Blat, Blon):
    phi1 = N.radians(90.0-Alat)
    phi2 = N.radians(90.0-Blat)
    theta1 = N.radians(Alon)
    theta2 = N.radians(Blon)
    arc = math.acos(math.sin(phi1)*math.sin(phi2)*math.cos(theta1-theta2) +
                    math.cos(phi1)*math.cos(phi2))
    xsdistance = rE * arc
    return xsdistance

# This dstacks arrays, unless it's the first time through, in which case it initialises the variable
def dstack_loop(data, Dict, Key):
    # Try evaluating dict[key]. If it doesn't exist, then initialise it
    # If it does exist, stack data
    try:
        Dict[Key]
    except KeyError:
        stack = data
        #Dict[Key] = data
    else:
        stack = N.dstack((Dict[Key],data))
    return stack
    pass

# Create thinned pressure levels for skew T barb plotting
def thinned_barbs(pres):
    levels = N.arange(20000.0,105000.0,5000.0)
    plocs = []
    for l in levels:
        ploc = N.where(abs(pres-l)==(abs(pres-l).min()))[0][0]
        plocs.append(ploc)
    thin_locs = N.array(plocs)
    return thin_locs # Locations of data at thinned levels

def trycreate(loc):
    try:
        os.stat(loc)
    except:
        os.makedirs(loc)

def padded_times(timeseq):
    padded = ['{0:04d}'.format(t) for t in timeseq]
    return padded

def string_from_time(usage,t,dom=0,strlen=0,conven=0,**kwargs):
    """
    conven  :   convection of MM/DD versus DD/MM
    """
    t = ensure_timetuple(t)

    if isinstance(t,str):
        if usage == 'output':
            usage = 'skip' # Time is already a string
        elif usage == 'title':
            pass
        #    if kwargs['itime']: # For averages or maxima over time
        #        itime = kwargs['itime']
        #        ftime = kwargs['ftime']
        #    else:
        #        pass
        else:
            raise Exception
    elif isinstance(t,float) or isinstance(t,int):
        # In this case, time is in datenum. Get it into tuple format.
        t = time.gmtime(t)
    else:
        pass

    if usage == 'title':
        # Generates string for titles
        if not 'itime' in kwargs: # i.e. for specific times
        #if not hasattr(kwargs,'itime'): # i.e. for specific times
            strg = '{3:02d}:{4:02d}Z on {2:02d}/{1:02d}/{0:04d}'.format(*t)
        else: # i.e. for ranges (average over time)
            s1 = '{3:02d}:{4:02d}Z to '.format(*kwargs['itime'])
            s2 = '{3:02d}:{4:02d}Z'.format(*kwargs['ftime'])
            strg = s1 + s2
    elif usage == 'wrfout':
        # Generates string for wrfout file finding
        # Needs dom
        if not dom:
            print("No domain specified; using domain #1.")
            dom = 1
        strg = ('wrfout_d0' + str(dom) +
               '{0:04d}-{1:02d}-{2:02d}_{3:02d}:{4:02d}:{5:02d}'.format(*t))
    elif usage == 'ruc':
        # This depends on the RUC version? Will break?
        strg = ('ruc2_252_{0:04d}{1:02d}{2:02d}_' +
                '{3:02d}{4:02d}_{5:02d}0.nc'.format(*t))
    elif usage == 'output':
        if not conven:
            # No convention set, assume DD/MM (I'm biased)
            conven = 'full'
        # Generates string for output file creation
        if conven == 'DM':
            strg = '{2:02d}{1:02d}_{3:02d}{4:02d}'.format(*t)
        elif conven == 'MD':
            strg = '{1:02d}{2:02d}_{3:02d}{4:02d}'.format(*t)
        elif conven == 'full':
            strg = '{0:04d}{1:02d}{2:02d}{3:02d}{4:02d}'.format(*t)
        else:
            print("Set convention for date format: DM or MD.")
    elif usage == 'dir':
        # Generates string for directory names
        # Needs strlen which sets smallest scope of time for string
        if not strlen:
             print("No timescope strlen set; using hour as minimum.")
             strlen = 'hour'
        n = lookup_time(strlen)
        strg = "{0:04d}".format(t[0]) + ''.join(
                ["{0:02d}".format(a) for a in t[1:n+1]])
    elif usage == 'skip':
        strg = t
    else:
        print("Usage for string not valid.")
        raise Exception
    return strg

def lookup_time(str):
    D = {'year':0, 'month':1, 'day':2, 'hour':3, 'minute':4, 'second':5}
    return D[str]

def get_level_naming(va,lv,**kwargs):
    #lv = kwargs['lv']

    if lv < 1500:
        return str(lv)+'hPa'
    elif lv == 2000:
        return 'sfc'
    elif lv.endswith('K'):
        return lv
    elif lv.endswith('PVU'):
        return lv
    elif lv.endswith('km'):
        return lv
    elif lv == 'all':
        if va == 'shear':
            name = '{0}to{1}'.format(kwargs['bottom'],kwargs['top'])
            return name
        else:
            return 'all_lev'


def check_vertical_coordinate(level):
    """ Check to see what type of level is requested by user.

    """
    if isinstance(level,(str,int,type(None))):
        lv = level
    elif isinstance(level,(list,tuple,N.ndarray)):
        lv = level[0]
    else:
        print(("What have you given me here? Level is"
                "{0}".format(type(level))))
        raise Exception

    # import pdb; pdb.set_trace()
    if isinstance(lv,int):
        if lv<100:
            return 'index'
        else:
            return 'isobaric'
    elif (lv is 'all') or (lv is None):
        return 'eta'

    elif lv.endswith('hPa'):
        # import pdb; pdb.set_trace()
        if lv[:4] == '2000':
            return 'surface'
        elif int(lv.split('h')[0]) < 2000:
            return 'isobaric'
        else:
            print("Pressure is in hPa. Requested value too large.")
            raise Exception

    elif lv.endswith('K'):
        return 'isentropic'

    elif lv.endswith('PVU'):
        return 'PV-surface'

    elif lv.endswith('km'):
        return 'geometric'

    else:
        print('Unknown vertical coordinate.')
        raise Exception

def closest(arr,val):
    """
    Find index of closest value.
    Only working on 1D array right now.

    Inputs:
    val     :   required value
    arr     :   array of values

    Output:

    idx     :   index of closest value

    """
    idx = N.argmin(N.abs(arr - val))
    return idx

def closest_datetime(times,t,round=False):
    """Find closest value in list of datetimes.
    Return index of closest.

    Arguments:
        times (list,tuple): collection of datetimes.
        t (datetime.datetime): required time
        round (bool,str): If False, return closest index only.
            If 'afterinc', return index of first time after t.
                (If closest time is identical to t, return that index)
            If 'afterexc', same, but if closest time = t, return one after.
            If 'beforeinc', return index of last time before t.
                (If closest time is identical to t, return that index)
            If 'beforeexc', same, but if closest time = t, return one before.

    Returns:
        idx (int): Index of times requests
        dtss[idx] (int): Number of seconds difference between the two.
    """
    stimes = N.array(sorted(times))
    dts = stimes-t
    dtss = [(d.days*86400)+d.seconds for d in dts]

    # Closest index
    cidx = N.argmin(N.abs(dtss))

    if round is False:
        idx = cidx
    else:
        if dtss[cidx] == 0:
            bidx_inc = cidx
            aidx_inc = cidx
            bidx_exc = cidx-1
            aidx_exc = cidx+1
        elif times[cidx] < t:
            bidx_inc = cidx
            bidx_exc = cidx
            aidx_inc = cidx+1
            aidx_exc = cidx+1
        else:
            bidx_exc = cidx-1
            bidx_inc = cidx-1
            aidx_exc = cidx
            aidx_nxc = cidx

        if round is 'afterinc':
            idx = aidx_inc
        elif round is 'afterexc':
            idx = aidx_exc
        elif round is 'beforeinc':
            idx = bidx_inc
        elif round is 'beforeexc':
            idx = bidx_exc
        else:
            raise Exception("Enter valid value for round.")
    
    return idx, dtss[idx]

def dstack_loop(data, obj):
    """
    Tries to stack numpy array (data) into 'stack' object (obj).
    If obj doesn't exist, then initialise it
    If obj does exist, stack data.
    """
    if isinstance(obj,N.ndarray):
        stack = N.dstack((obj,data))
    else:
        stack = data

    return stack

def dstack_loop_v2(data, obj):
    """
    Need to set obj = 0 at start of loop in master script

    Tries to stack numpy array (data) into 'stack' object (obj).
    If obj doesn't exist, then initialise it
    If obj does exist, stack data.
    """
    try:
        print(obj)
    except NameError:
        stack = data
    else:
        stack = N.dstack((obj,data))

    return stack

def vstack_loop(data, obj):
    """
    Need to set obj = 0 at start of loop in master script

    Tries to stack numpy array (data) into 'stack' object (obj).
    If obj doesn't exist, then initialise it
    If obj does exist, stack data.
    """

    if isinstance(obj,N.ndarray):
        stack = N.vstack((obj,data))
    else:
        stack = data

    return stack


def generate_times(idate,fdate,interval,fmt='timetuple',inclusive=False):
    """
    :param itime:       Start date/time. Format is
                        YYYY,MM,DD,HH,MM,SS (calendar.timegm).
    :type itime:        list,tuple
    :param ftime:       End date/time. Same format as itime.
    :type ftime:        list,tuple
    :param interval:    interval between output times, in seconds.
    :type interval:     int
    :returns:           list of times in datenum format.

    """
    if isinstance(idate,datetime.datetime):
        # idate = (idate.year,idate.month,idate,day,idate.hour,
                    # idate.minute,idate.second)
        # fdate = (fdate.year,fdate.month,fdate,day,fdate.hour,
                    # fdate.minute,fdate.second)
        idate = datetime_to_timetuple(idate)
        fdate = datetime_to_timetuple(fdate)
    it = calendar.timegm(idate)
    ft = calendar.timegm(fdate)
    if inclusive:
        ft = ft + interval
    times = N.arange(it,ft,interval,dtype=int)
    tttimes = [ensure_timetuple(t) for t in times]
    if fmt=='datetime':
        dttimes = [timetuple_to_datetime(t) for t in tttimes]
        return dttimes
    else:
        return times

def generate_colours(M,n):
    """
    M       :   Matplotlib instance
    n       :   number of colours you want

    Returns

    Usage: when cycling over n plots, the colour should
    be colourlist[n].
    """

    colourlist = [M.cm.spectral(i) for i in N.linspace(0.08,0.97,n)]
    return colourlist

def get_sequence(x,sos=0):
    """ Returns a sequence (tuple or list) for iteration.
    Avoids an error for strings/integers.
    SoS = 1 enables the check for a sequence of sequences (list of dates)
    """
    # If sos is True, then use its first element as an example.
    if sos:
        y = x[0]
    else:
        y = x

    if isinstance(y, collections.Sequence) and not isinstance(y, str):
        # i.e., if y is a list or tuple
        return x
    else:
        # make a one-element list
        return [x,]

def convert_tuple_to_dntimes(times):
    """
    Convert tuple or tuple of tuples to datenum date format.
    """
    timelist = get_sequence(times,sos=1)

    dntimes = []
    for t in timelist:
        dntimes.append(calendar.timegm(t))

    return dntimes

def ensure_datenum(times,fmt='int'):
    """
    Make sure times are in list-of-datenums format.
    If not, convert them.

    Possibilities:
    times = 123456                                      #1
    times = (123456,)                                   #2
    times = (123456,234567)                             #3
    times = (2011,12,1,18,0,0)                          #4
    times = ((2011,12,1,18,0,0),(2011,12,2,6,0,0))      #5

    fmt     :   whether to return list of integers or an integer
                'int' or 'list'

    Output:
    dntimes = (123456,) or (123456,234567)
    """
    if isinstance(times,int):
        dntimes = [times,] #1
    elif isinstance(times,str):
        print("Don't give me strings...")
        raise Exception
    elif isinstance(times,datetime.datetime):
        dntimes = convert_tuple_to_dntimes(datetime_to_timetuple(times))
    elif isinstance(times,(list,tuple)): #2,3,4,5
        if not isinstance(times[0],int): #5
            dntimes = convert_tuple_to_dntimes(times)
        elif times[0]<3000: #4
            dntimes = convert_tuple_to_dntimes(times)
        elif isinstance(times[0],datetime.datetime):
            dntimes = [convert_tuple_to_dntimes(datetime_to_timetuple(t)) 
                        for t in times]
        else: #2,3
            dntimes = times

    if (fmt == 'list') or (len(dntimes)>1):
        return dntimes
    elif (fmt == 'int') or (len(dntimes)==1):
        return dntimes[0]
    else:
        print("Nonsense format choice.")
        raise Exception

    import pdb; pdb.set_trace()

def ensure_timetuple(times,fmt='single'):
    """
    MAke sure time(s) are in six-item tuple format
    (YYYY,MM,DD,HH,MM,SS)

    fmt     :   whether to return a list of tuples or single tuple.
                'list' or 'single'

    Possibilities:
    times = 123456                                      #1
    times = (123456,)                                   #2
    times = (123456,234567)                             #3
    times = (2011,12,1,18,0,0)                          #4
    times = ((2011,12,1,18,0,0),(2011,12,2,6,0,0))      #5
    """
    if isinstance(times,(int,N.int64)):
        tttimes = [list(time.gmtime(times)),] #1
    elif isinstance(times,str):
        print("Don't give me strings...")
        raise Exception
    elif isinstance(times,datetime.datetime):
        tttimes = datetime_to_timetuple(times)
    # elif isinstance(times[0],datetime.datetime):
        # tttimes = datetime_to_timetuple(times)
    elif isinstance(times,(list,tuple)): #2,3,4,5
        if not isinstance(times[0],int): #5
            tttimes = times
        elif times[0]<3000: #4
            tttimes = [times,]
        elif isinstance(times[0],datetime.datetime):
            tttimes = [datetime_to_timetuple(t) for t in times]
        elif isinstance(times[0]>3000): #2,3
            tttimes = [list(time.gmtime(t)) for t in times]

    # import pdb; pdb.set_trace()
    
    if (fmt == 'list') or (len(tttimes)>1):
        return tttimes
    elif (fmt == 'single') or (len(tttimes)==1):
        return tttimes[0]
    else:
        print("Nonsense format choice.")
        raise Exception

def ensure_datetime(t):
    """
    Possibilities:
    times = 123456                                      #1
    times = (123456,)                                   #2
    times = (123456,234567)                             #3
    times = (2011,12,1,18,0,0)                          #4
    times = ((2011,12,1,18,0,0),(2011,12,2,6,0,0))      #5
    times = datetime.datetime                           #6
    times = (datetime.datetime, datetime.datetime)      #7
    """
    if isinstance(t,(int,N.int64)):  #1
        utc = timetuple_to_datetime(list(time.gmtime(t)))
    elif isinstance(t,datetime.datetime): #6
        utc = t
    elif isinstance(t,(list,tuple)):
        if isinstance(t[0],datetime.datetime): #7
            utc = t
        elif isinstance(t[0],(int,N.int64)):
            if t[0] < 3000: # 4
                utc = timetuple_to_datetime(t)
            else: #2, 3
                utc = [timetuple_to_datetime(list(time.gmtime(t))) for
                        t in t]
        elif isinstance(t[0],(list,tuple)): # 5
            utc = [timetuple_to_datetime(t) for t in t]
    else:
        raise Exception("Unidentified format.")
    return utc

def datetime_to_timetuple(utc):
    tttime = (utc.year,utc.month,utc.day,
                utc.hour,utc.minute,utc.second)
    return tttime

def timetuple_to_datetime(utc):
    dttime = datetime.datetime(*utc[:6])
    return dttime

def dt_from_fnames(f1,f2,model):
    """Work out time difference between two data files
    from their naming scheme.


    Arguments:
        f1,f2 (str): filename, with or without extension
        model (str): model used to generate data
    Returns:
        Difference between files, in seconds.
    """
    if f1.endswith('.nc'):
        f1 = f1.split('.')[0]
        f2 = f2.split('.')[0]
    if (model=='wrfout') or (model=='wrf'):
        # We assume default naming
        t = []
        for f in (f1,f2):
            _1, _2, tstr = f.split('_',2)
            fmt = '%Y-%m-%d_%H:%M:%S'
            t.append(datetime.datetime.strptime(tstr,fmt))
        dt = t[1] - t[0]
        return dt.seconds

def get_netcdf_naming(model,t,dom=0):
    """
    By default:
    wrfout files don't have an extension
    other files have .nc extension (convert first)
    """
    
    t = ensure_datetime(t)
    # import pdb; pdb.set_trace()
    if (model=='wrfout') or (model=='wrf'):
        if not dom:
            print("No domain specified; using domain #1.")
            dom = 1
        # fname = ('wrfout_d{0:02d}_{1:04d}-{2:02d}-{3:02d}_{4:02d}:{5:02d}:{6:02d}'.format(dom,*t))
        fname = ('wrfout_d{0:02d}_{1:04d}-{2:02d}-{3:02d}_{4:02d}:{5:02d}:{6:02d}'.format(dom,
                                     t.year,t.month,t.day,t.hour,t.minute,t.second))
    elif model == 'ruc':
        # This depends on the RUC version? Will break?


        # prefix = ruc_naming_prefix(t)

        # fname = (prefix+'{0:04d}{1:r2d}{2:02d}_{3:02d}{4:02d}_{5:02d}0.nc'.format(*t))
        fname = getdata.RUC_fname(t,filetype='netcdf')
        # import pdb; pdb.set_trace()
    else:
        print("Model format not supported yet.")
        raise Exception
    return fname

def determine_model(fname):
    """
    Return model depending on naming convention.

    If no model exists, return false.
    """

    models = {'wrfout_d':'wrfout','ruc':'ruc','rap':'ruc'}

    for k,v in models.items():
        if k in fname[:10]:
            return v

    return False

def save_data(data,folder,fname,format='pickle'):
    """
    Save array to file.
    """

    # Strip file extension given
    fname_base = os.path.splitext(fname)[0]
    # Check for folder, create if necessary
    trycreate(folder)
    # Create absolute path
    fpath = os.path.join(folder,fname_base)

    if format=='pickle':
        with open(fpath+'.pickle','wb') as f:
            pickle.dump(data,f)
    elif format=='numpy':
        N.save(fpath,data)
    elif format=='json':
        j = json.dumps(data)
        with open(fpath+'.json','w') as f:
            print(j, file=f)
    else:
        print("Give suitable saving format.")
        raise Exception

    print(("Saved file {0} to {1}.".format(fname,folder)))

def load_data(folder,fname,format='pickle'):
    """
    Load array from file.
    """

    fname2 = os.path.splitext(fname)[0]
    fpath = os.path.join(folder,fname2)
    if format=='pickle':
        with open(fpath+'.pickle','rb') as f:
            data = pickle.load(f)
    elif format=='numpy':
        data = N.load(fpath+'.npy')
    elif format=='json':
        print("JSON stuff not coded yet.")
        raise Exception
    else:
        print("Give suitable loading format.")
        raise Exception

    print(("Loaded file {0} from {1}.".format(fname,folder)))
    return data


def return_subdomain(data,lats,lons,Nlim,Elim,Slim,Wlim,
                        fmt='latlon'):
    """
    Returns smaller domain of data and lats/lons based
    on specified limits.
    """
    # import pdb; pdb.set_trace()
    Nidx = closest(lats,Nlim) 
    Eidx = closest(lons,Elim)
    Sidx = closest(lats,Slim)
    Widx = closest(lons,Wlim)

    # Assuming [lats,lons]
    if fmt=='latlon':
        if Nidx<Sidx:
            xmin,xmax = Nidx,Sidx
        else:
            xmin,xmax = Sidx,Nidx

        if Widx<Eidx:
            ymin,ymax = Widx,Eidx
        else:
            ymin,ymax = Eidx,Widx
    elif fmt=='lonlat':
        if Nidx<Sidx:
            ymin,ymax = Nidx,Sidx
        else:
            ymin,ymax = Sidx,Nidx

        if Widx<Eidx:
            xmin,xmax = Widx,Eidx
        else:
            xmin,xmax = Eidx,Widx
    else:
        print("Need right format")
        raise Exception
    # Radar data: latlon, N<S, W<E
    # data = data[Nidx:Sidx+1,Widx:Eidx+1]

    data = data[xmin:xmax+1,ymin:ymax+1]

    if Nidx<Sidx:
        lats = lats[Nidx:Sidx+1]
    else:
        # flipud for RUC data - does this break WRF?
        lats = lats[Sidx:Nidx+1]
        # lats = N.flipud(lats[Sidx:Nidx+1])
    if Widx<Eidx: 
        lons = lons[Widx:Eidx+1]
    else:
        lons = lons[Eidx:Widx+1]

    # pdb.set_trace()
    return data,lats,lons

def interp_latlon(data,lat,lon,lats,lons):
    ntimes = data.shape[0]
    nlvs = data.shape[1]
    dataout = N.zeros([ntimes,nlvs,1,1])
    for lv in range(nlvs):
    # for t in range(ntimes):
        dataout[:,lv,0,0] = interp2point(data[:,lv:lv+1,:,:],lat,lon,lats,lons)
    return dataout


def interp2point(data,lat_loc,lon_loc,lat,lon,lvidx=0,xyidx=False):
        er = 6370000
        if xyidx:
            # Don't need data, ignore
            field = None
            data = None
        else:
            field = data[:,lvidx,:,:]
            # field = self.make_4D(data)[:,lvidx,:,:]
        templat = lat.ravel()
        templon = lon.ravel()

        #calculates the great circle distance from the inquired point
        delta = haversine_baker(lon_loc,lat_loc,templon,templat,earth_rad=er)
        #grid distance
        dxdy = haversine_baker(templon[0],templat[0],templon[1],templat[0],earth_rad=er)

        #9 smallest values to find the index of
        smallest = heapq.nsmallest(9,delta.ravel())

        wtf = N.in1d(delta.ravel(),smallest)
        tf2d = N.in1d(delta.ravel(),smallest).reshape(lat.shape)
        ix,iy = N.where(tf2d == True)
        if xyidx:
            xidx = N.median(ix)
            yidx = N.median(iy)
            return xidx, yidx

        weights =  1.0 - delta.ravel()[wtf]/(dxdy.ravel() * 2)
        weighted_mean = N.average(field[:,ix,iy],axis=1,weights=weights.ravel())

        return weighted_mean

def haversine_baker(lon1, lat1, lon2, lat2, radians=False, earth_rad=6371.227):
    """
    Allows to calculate geographical distance
    using the haversine formula.
    :param lon1: longitude of the first set of locations
    :type lon1: numpy.ndarray
    :param lat1: latitude of the frist set of locations
    :type lat1: numpy.ndarray
    :param lon2: longitude of the second set of locations
    :type lon2: numpy.float64
    :param lat2: latitude of the second set of locations
    :type lat2: numpy.float64
    :keyword radians: states if locations are given in terms of radians
    :type radians: bool
    :keyword earth_rad: radius of the earth in km
    :type earth_rad: float
    :returns: geographical distance in km
    :rtype: numpy.ndarray
    """

    if radians == False:
        cfact = N.pi / 180.0
        lon1 = cfact * lon1
        lat1 = cfact * lat1
        lon2 = cfact * lon2
        lat2 = cfact * lat2

    # Number of locations in each set of points
    if not N.shape(lon1):
        nlocs1 = 1
        lon1 = N.array([lon1])
        lat1 = N.array([lat1])
    else:
        nlocs1 = N.max(N.shape(lon1))
    if not N.shape(lon2):
        nlocs2 = 1
        lon2 = N.array([lon2])
        lat2 = N.array([lat2])
    else:
        nlocs2 = N.max(N.shape(lon2))
    # Pre-allocate array
    distance = N.zeros((nlocs1, nlocs2))
    i = 0
    while i < nlocs2:
        # Perform distance calculation
        dlat = lat1 - lat2[i]
        dlon = lon1 - lon2[i]
        aval = (N.sin(dlat / 2.) ** 2.) + (N.cos(lat1) * N.cos(lat2[i]) * (N.sin(dlon / 2.) ** 2.))
        distance[:, i] = (2. * earth_rad * N.arctan2(N.sqrt(aval), N.sqrt(1 - aval))).T
        i += 1
    return distance.ravel()

def get_latlon_idx(lats,lons,lat,lon):
    coords = N.unravel_index(N.argmin((lat-lats)**2+
                (lon-lons)**2),lons.shape)
    # lon, lat
    return [int(c) for c in coords]

def make_subplot_label(ax,label):
    if not label.endswith(')'):
        label = label + ')'
    ax.text(0.1,0.15,label,transform=ax.transAxes,
        bbox={'facecolor':'white'},fontsize=13,zorder=1000)
    return ax
