"""
Utility scripts related to WRF files.
"""

# Some functions to import WRF data etc
from netCDF4 import Dataset
import pdb
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as N
#import scipy.ndimage as nd
import os
import calendar

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
    # domain in gridpoints
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
    
def wrfout_files_in(self,folders,dom='notset',init_time='notset',descend=1,**kwargs):
    """
    Hunt through given folder(s) to find all occurrences of wrfout
    files.

    Optional:
    avoid   :   string. if folder name includes this
                term, ignore the folder.
    
    Returns:
    wrfouts :   list of absolute paths to wrfout files
    """

    folders = self.get_sequence(folders)
    avoids = []
    if 'avoid' in kwargs:
        # Avoid folder names with this string
        # or list of strings
        avoid = self.get_sequence(kwargs['avoid'])
        for a in avoid:
            avoids.append('/{0}/'.format(a))


    w = 'wrfout' # Assume the prefix
    if init_time=='notset':
        suffix = '*0'
        # We assume the user has wrfout files in different folders for different times
    else:
        try:
            it = self.string_from_time('wrfout',init_time)
        except:
            print("Not a valid wrfout initialisation time; try again.")
            raise Error
        suffix = '*' + it

    if dom=='notset':
        # Presume all domains are desired.
        prefix = w + '_d'
    elif (dom == 0) | (dom > 8):
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
    return wrfouts
