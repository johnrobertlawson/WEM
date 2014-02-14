import os
import pdb

# Specify type of file required
data = 'fcst'
#data = 'anl'


# Dates of initialisation YYYYMMDD, list of strings
#dates = ['201104{0:02d}'.format(n) for n in (19,20)]
dates = ['20110419']
# Hours of initialisation HH, list of strings
hours = ['00','12']

# For forecasts only
# Maximum forecast time to download (inclusive)
Tmax = 48
# Interval (hr) to fetch forecasts
Tint = 3
# If date is before ####, download grib1, use this:
age = 'old'

def get_anl(dates,hours,*args):
    for d in dates:
        for h in hours:
            if age=='new':
                command = ('wget "http://nomads.ncdc.noaa.gov/data/namanl/'+
                            d[:6]+'/'+ d+'/namanl_4_'+d+'_'+h+'00_000.grb2"')
            elif age=='old':
                command = ('wget "http://nomads.ncdc.noaa.gov/data/namanl/'+
                            d[:6]+'/'+ d+'/namanl_218_'+d+'_'+h+'00_000.grb"')
            os.system(command)
                
def get_218fcst(dates,hours,Tmax,Tint):
    for d in dates:
        for h in hours:
            fhs = range(0,Tmax+Tint,Tint)
            for fh in fhs:
                fpad = "%03d" %fh
                if age == 'old':
                    command = ('wget "http://nomads.ncdc.noaa.gov/data/nam/'+
                                d[:6]+'/'+ d+'/nam_218_'+d+'_'+h+'00_'+fpad+'.grb"')
                elif age == 'new': # doesn't seem to work
                    command = ('wget "http://nomads.ncdc.noaa.gov/data/nam/'+
                                d[:6]+'/'+ d+'/nam_4_'+d+'_'+h+'00_'+fpad+'.grb2"')
                   
                os.system(command)

CMND = {'fcst':get_218fcst, 'anl':get_anl}

CMND[data](dates,hours,Tmax,Tint)

