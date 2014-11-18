import sys
import numpy as N
sys.path.append('/home/jrlawson/gitprojects/WEM')

from WEM.postWRF.postWRF import WRFEnviron

# Initialise the environment
p = WRFEnviron()

# output directory
outdir = '/home/jrlawson/public_html/bowecho'
# netcdf directory.
ncdir = '/chinook2/jrlawson/bowecho/20130815/GEFSR2/c00/ICBC'

# initial and final times.
itime = (2013,8,15,3,0,0)
ftime = (2013,8,15,3,0,0)
hourly = 6
times = p.generate_times(itime,ftime,hourly*60*60)

# Loop over all times.
# clvs lets you change the contour values
# The second argument is accumulation time, in hours
for t in times:
    p.plot_accum_rain(t,hourly,ncdir,outdir,clvs=N.arange(5,85,1))
