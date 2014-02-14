import os
import calendar

dates = ['20110419','20110420']
hours = ['00','06','12','18']

# If date is before ####, download grib1, use this:
#newold = 'old'
newold = 'new'

for d in dates:
    for h in hours:
        if newold=='new':
            os.system('wget "http://nomads.ncdc.noaa.gov/data/gfsanl/'+d[:6]+'/'+ d+'/gfsanl_4_'+d+'_'+h+'00_000.grb2"')
        else:
            os.system('wget "http://nomads.ncdc.noaa.gov/data/gfsanl/'+d[:6]+'/'+ d+'/gfsanl_3_'+d+'_'+h+'00_000.grb"')
