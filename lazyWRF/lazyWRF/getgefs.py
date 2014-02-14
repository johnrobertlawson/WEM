# This script downloads all variables for GEFS R2 reforecasts
# John Lawson, University of Utah
import os
import calendar
import time
 
# Dates in YYYYMMDD format - all GEFS runs are 00z
# Need to be strings
# This is a list of dates; one date can be entered too.
 
#dates = ['20111128','20111129','20111130']
dates = ['20110419']

# Download data
download = 0
# Split data
split = 1
 
# This switch allows GEFS data after T190 to be downloaded (lower resolution)
lowres = 0
 
# This selected all 10 perturbation ensemble members. Change ens for desired member (or mean/sprd)
ens = ['p' + '%02u' %p for p in range(1,11)]
 
# This switch allows downloading of control member too.
control = 1
if control:
    ens.append('c00')
 
# Choose gaussian or latlon coordinates.
coord = 'latlon'
 
# Root directory of FTP site
FTP = 'ftp://ftp.cdc.noaa.gov/Projects/Reforecast2/'
 
# -nc does not download a renamed multiple copy of file
# --output-document=CATNAME concatenates all files together for the big grib file
# -nd makes sure hierachy isn't downloaded too

if download: 
    for d in dates:
        for e in ens:
            url = os.path.join(FTP, d[0:4], d[0:6], d+'00', e, coord)
            fname = '/*' + e + '.grib2'
            CATNAME = d + '_' + e + '.grib2'
            cmnd = "wget -nc -nd --output-document=" + CATNAME + ' ' + url + fname
            os.system(cmnd)
            print d, e, " Downloaded."
     
# This section will split the data into forecast times for WRF to read
# Using WGRIB2
# fin : grib2 input file
# fout : smaller grib2 output file with just one forecast time
# timestr : search pattern to find the forecast time

if split: 
    for d in dates:
        # Convert this date to python time for later conversion
        pytime_anl = calendar.timegm((int(d[:4]),int(d[4:6]),int(d[6:8]),0,0,0))
        for e in ens:
            fin = ''.join((d,'_',e,'.grib2'))
            fprefix = '_'.join((d,e,'f'))
            for t in range(0,198,6):
                ts = "%03d" %t # Gets files into chron order with padded zeroes
                if t==0:
                    timestr = '":anl:"'    
                else:
                    timestr = ''.join(('":(',str(t),' hour fcst):"'))
                fout = fprefix + ts + '.grib2'
                str1 = ' '.join(('wgrib2',fin,'-match',timestr,'-grib',fout)) 
                os.system(str1)
