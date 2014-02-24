# Functions useful for mesowest data

import numpy as N
import calendar as cal
import pdb
from netCDF4 import Dataset

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
