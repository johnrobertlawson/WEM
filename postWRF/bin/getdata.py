import os
import calendar 
import time
import pdb

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

def getruc(dates,hours):

    def right_url(d,h):
        """To work out what URL to use for RUC/RAP data.
        """
        yr = int(d[:4])
        mth = int(d[4:6])


        if (yr > 2012) and (mth > 3): # With a massive gap for RAP
            URL_base = "http://nomads.ncdc.noaa.gov/data/rap130/"
            URL = '/'.join((URL_base,d[:6],d,'rap_130_'+d+'_'+h+'00_000.grb2'))
        elif (yr > 2007) and (mth > 10): # Massive gap after 2012/05 (transition to RAP).
            URL_base = "http://nomads.ncdc.noaa.gov/data/rucanl/"
            URL = '/'.join((URL_base,d[:6],d,'ruc2anl_130_'+d+'_'+h+'00_000.grb2'))
        elif (yr>2006):
            URL_base = "http://nomads.ncdc.noaa.gov/data/rucanl/"
            URL = '/'.join((URL_base,d[:6],d,'ruc2anl_252_'+d+'_'+h+'00_000.grb'))
        elif (yr>2004):
            URL_base = "http://nomads.ncdc.noaa.gov/data/rucanl/"
            URL = '/'.join((URL_base,d[:6],d,'ruc2_252_'+d+'_'+h+'00_000.grb'))

        return URL


    for d in dates:
        for h in hours:
            URL = right_url(d,h)
            command = 'wget ' + URL
            os.system(command)
