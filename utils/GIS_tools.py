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

import unix_tools as utils

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
        print 'Loading station data for ' + s
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
    print "Creating a pretty loop..."
    os.system('convert -delay 50 '+fpath+dom+'*.png > -loop '+fpath+dom+'_windloop.gif')
    print "Finished!"

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
        timeInds = range(nc.variables['Times'].shape[0])
    elif len(times)==1: # If only one time is desired
        # Time is in 6-tuple format
        timeInds = find_time_index(nc.variables['Times'],times) # This function is from this module
    elif len(times)==2: # Find all times between A and B
        timeIndA = find_time_index(nc.variables['Times'],times[0])
        timeIndB = find_time_index(nc.variables['Times'],times[1])
        timeInds = range(timeIndA,timeIndB)
    # Lat/lon of interest and their grid pointd
    lats = nc.variables['XLAT'][:]
    lons = nc.variables['XLONG'][:]
    if latslons == 'all':
        latInds = range(lats.shape[-2])
        lonInds = range(lons.shape[-1])
    else:
        xmin,ymax = gridded_data.getXY(lats,lons,Nlim,Wlim)
        xmax,ymin = gridded_data.getXY(lats,lons,Slim,Elim)
        latInds = range(ymin,ymax)
        lonInds = range(xmin,xmax)

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
        for f in files:
            model_test = determine_model(f.split('/')[-1])
            if model_test:
                matches += 1
                model = model_test

        if matches < 1:
            print("No netcdf files found.")
            raise Exception
        elif matches > 1 and isinstance(t,str):
            print("Ambiguous netcdf file selection. Specify model?")
            raise Exception

    # import pdb; pdb.set_trace()
    if model=='wrfout':
        pfx = 'wrfout' # Assume the prefix
    elif model=='ruc':
        pfx = 'ruc'
        # TODO: logic that considers the four versions of RUC
    else:
        raise Exception

    # Pick unambiguous
    if t=='auto':
        # We assume the user has wrfout files in different folders for different times
        f = glob.glob(os.path.join(folder,pfx+'*'))
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
            print("Found {0} wrfout files.".format(len(wrfouts)))
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
    print 'Topographic data has been loaded. Everest is but a mere pixel.'
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
    if isinstance(level,(basestring,int)):
        lv = level
    elif isinstance(level,(list,tuple,N.ndarray)):
        lv = level[0]
    else:
        print("What have you given me here? Level is"
                "{0}".format(type(level)))
        raise Exception

    # import pdb; pdb.set_trace()
    if isinstance(lv,int):
        return 'index'
    if lv.endswith('hPa'):
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

    elif lv == 'all':
        return 'eta'

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
        print obj
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


def generate_times(itime,ftime,interval):
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
    it = calendar.timegm(idate)
    ft = calendar.timegm(fdate)
    times = range(it,ft,interval)
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

    if isinstance(y, collections.Sequence) and not isinstance(y, basestring):
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

def ensure_datenum(times,fmt='list'):
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
    elif isinstance(times,basestring):
        print("Don't give me strings...")
        raise Exception
    elif isinstance(times,collections.Sequence): #2,3,4,5
        if not isinstance(times[0],int): #5
            dntimes = convert_tuple_to_dntimes(times)
        elif times[0]<3000: #4
            dntimes = convert_tuple_to_dntimes(times)
        else: #2,3
            dntimes = times

    if (fmt == 'list') or (len(dntimes)>1):
        return dntimes
    elif (fmt == 'int') or (len(dntimes)==1):
        return dntimes[0]
    else:
        print("Nonsense format choice.")
        raise Exception

def ensure_timetuple(times,fmt='list'):
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

    if isinstance(times,int):
        tttimes = [calendar.timegm(times),] #1
    elif isinstance(times,basestring):
        print("Don't give me strings...")
        raise Exception
    elif isinstance(times,collections.Sequence): #2,3,4,5
        if not isinstance(times[0],int): #5
            tttimes = times
        elif times[0]<3000: #4
            tttimes = [times,]
        else: #2,3
            tttimes = [calendar.timegm(t) for t in times]

    if (fmt == 'list') or (len(tttimes)>1):
        return tttimes
    elif (fmt == 'single') or (len(tttimes)==1):
        return tttimes[0]
    else:
        print("Nonsense format choice.")
        raise Exception

def get_netcdf_naming(model,t,dom=0):
    """
    By default:
    wrfout files don't have an extension
    other files have .nc extension (convert first)
    """
    if model=='wrfout':
        if not dom:
            print("No domain specified; using domain #1.")
            dom = 1
        fname = ('wrfout_d{0:02d}_{1:04d}-{2:02d}-{3:02d}_{4:02d}:{5:02d}:{6:02d}'.format(dom,*t))
    elif model == 'ruc':
        # This depends on the RUC version? Will break?
        fname = ('ruc2_252_{0:04d}{1:02d}{2:02d}_{3:02d}{4:02d}_{5:02d}0.nc'.format(*t))
    else:
        print("Model format not supported yet.")
        raise Exception
    return fname

def determine_model(fname):
    """
    Return model depending on naming convention.

    If no model exists, return false.
    """

    models = {'wrfout_d':'wrfout','ruc2':'ruc'}

    for k,v in models.iteritems():
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
            print >> f,j
    else:
        print("Give suitable saving format.")
        raise Exception

    print("Saved file {0} to {1}.".format(fname,folder))

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

    print("Loaded file {0} from {1}.".format(fname,folder))
    return data
