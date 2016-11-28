import os
import calendar 
import time
import pdb
import sys
import glob

from . import GIS_tools as utils
from . import GIS_tools

# date format: YYYYMMDD (string)

def getgefs(dates,download=1,split=1,lowres=0,custom_ens=0,control=1,
            coord='latlon'):
    """This script downloads all variables for GEFS R2 reforecasts.
    All runs are initialised at 0000 UTC.

    Inputs (all optional unless stated):
    dates       :   YYYYMMDD, list of strings (mandatory)
    download    :   whether to download the data
    split       :   whether to split up the data
    lowres      :   whether to download times after T+190
    custom_ens  :   a custom list of perturbation ensemble members
    control     :   whether to download the control member
    coord       :   latlon/gaussian grid
    """

    # This selected all 10 perturbation ensemble members. Change ens for desired member (or mean/sprd)
    if not custom_ens:
        ens = ['p' + '%02u' %p for p in range(1,11)]
    else:
        ens = custom_ens
     
    if control:
        ens.append('c00')
         
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
                print(d, e, " Downloaded.")
         
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

def getgfs(dates,hours):
    """ Downloads GFS analysis data.

    Inputs:
    dates       :   List of strings, YYYYMMDD
    hours       :   List of strings, HH 
    """

    # If date is before 2007, download grib1.

    for d in dates:
        yr_int = int(d[:4])
        for h in hours:
            if yr_int > 2006:
                os.system('wget "http://nomads.ncdc.noaa.gov/data/gfsanl/'+d[:6]+'/'+ d+'/gfsanl_4_'+d+'_'+h+'00_000.grb2"')
            else:
                os.system('wget "http://nomads.ncdc.noaa.gov/data/gfsanl/'+d[:6]+'/'+ d+'/gfsanl_3_'+d+'_'+h+'00_000.grb"')

def getnam(dates,hours,datatype,**kwargs):
    """ Downloads NAM analysis and forecast data.

    Inputs:
    dates       :   List of strings, YYYYMMDD
    hours       :   List of strings, HH 
    datatype    :   analysis or forecast.

    Optional arguments for forecasts via kwargs:
    tmax        :   maximum forecast time to download (inclusive)
    tint        :   internal (hr) between fetched forecasts

    """

    # If date is before ####, download grib1, use this:
    age = 'old'

    def get_anl(dates,hours,*args):
        for d in dates:
            for h in hours:
                #if age=='new':
                #    command = ('wget "http://nomads.ncdc.noaa.gov/data/namanl/'+
                #                d[:6]+'/'+ d+'/namanl_4_'+d+'_'+h+'00_000.grb2"')
                #elif age=='old':
                command = ('wget "http://nomads.ncdc.noaa.gov/data/namanl/'+
                            d[:6]+'/'+ d+'/namanl_218_'+d+'_'+h+'00_000.grb"')
                os.system(command)
         
    # Where are these forecast archives?                
    def get_218fcst(dates,hours,Tmax,Tint):
        for d in dates:
            for h in hours:
                fhs = list(range(0,Tmax+Tint,Tint))
                for fh in fhs:
                    fpad = "%03d" %fh
                    if age == 'old':
                        command = ('wget "http://nomads.ncdc.noaa.gov/data/nam/'+
                                    d[:6]+'/'+ d+'/nam_218_'+d+'_'+h+'00_'+fpad+'.grb"')
                    elif age == 'new': # doesn't seem to work
                        command = ('wget "http://nomads.ncdc.noaa.gov/data/nam/'+
                                    d[:6]+'/'+ d+'/nam_4_'+d+'_'+h+'00_'+fpad+'.grb2"')
                       
                    os.system(command)

    CMND = {'forecast':get_218fcst, 'analysis':get_anl}
    CMND[data](dates,hours,Tmax,Tint)

def getruc(utc,ncpath='./',convert2nc=False,duplicate=False):

    URL = RUC_URL(utc)
    fname = RUC_fname(utc)
    URLpath = os.path.join(URL,fname)
    fpath = os.path.join(ncpath,fname)
    if not duplicate:
        fexist = glob.glob(fpath)

    # import pdb; pdb.set_trace()
    if not len(fexist):
        command = 'wget {0} -P {1}'.format(URLpath,ncpath)
        os.system(command)
        if convert2nc:
            command2 = 'ncl_convert2nc {0} -o {1}'.format(fpath,ncpath)
            os.system(command2)
    return

def RUC_fname(utc,filetype='grib'):
    """
    Returns RUC filename for date.
    """
    version = RUC_version(utc)

    t = GIS_tools.ensure_datenum(utc)
    yr = time.gmtime(t).tm_year
    mth = time.gmtime(t).tm_mon
    day = time.gmtime(t).tm_mday
    hr = time.gmtime(t).tm_hour

    if version == 3:
        prefix = 'rap_130'
        suffix = 'grb2'
    elif version == 2:
        prefix = 'ruc2anl_130'
        suffix = 'grb2'
    elif version == 1:
        prefix = 'ruc2anl_252'
        suffix = 'grb'
    else:
        prefix = 'ruc2_252'
        suffix = 'grb'

    if filetype=='netcdf':
        suffix = 'nc'
    fname = '{0}_{1:04d}{2:02d}{3:02d}_{4:02d}00_000.{5}'.format(prefix,yr,mth,day,hr,suffix)
    return fname

def RUC_URL(utc):
    """
    Returns URL to download RUC file from nomads.
    """
    t = GIS_tools.ensure_datenum(utc)
    yr = time.gmtime(t).tm_year
    mth = time.gmtime(t).tm_mon
    day = time.gmtime(t).tm_mday

    URL_base = "http://nomads.ncdc.noaa.gov/data/rucanl"
    URL = '{0}/{1:04d}{2:02d}/{1:02d}{2:02d}{3:02d}/'.format(URL_base,yr,mth,day)

    return URL

def RUC_version(utc,fname=False,URL=False):
    """Returns the version/fname of RUC file
    """
    t = GIS_tools.ensure_datenum(utc)
    date0 = utils.ensure_datenum((2004,1,1,0,0,0))
    date1 = utils.ensure_datenum((2007,1,1,0,0,0))
    date2 = utils.ensure_datenum((2008,1,1,0,0,0))
    date3 = utils.ensure_datenum((2009,1,1,0,0,0))
    date4 = utils.ensure_datenum((2012,5,9,0,0,0))

    if t >= date4:
        version = 3
    elif t >= date3:
        version = 2
    elif t >= date2:
        version = 0
    elif t >= date1:
        version = 1
    elif t >= date0:
        version = 0
    else:
        print("No RUC data for this date exists.")
        raise Exception

    # print("This RUC file is Version {0}.".format(version))
    return version

