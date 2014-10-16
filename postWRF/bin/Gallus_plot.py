import os
import pdb
import sys
import matplotlib as M
M.use('gtkagg')
import matplotlib.pyplot as plt
import numpy as N

sys.path.append('/path/to/WEM/')

from WEM.postWRF import WRFEnviron

p = WRFEnviron()

itime = (2006,5,10,12,0,0)
ftime = (2006,5,11,12,0,0)
hourly = 3
times = p.generate_times(itime,ftime,hourly*60*60)
level = 2000

outdir = '/absolute/path/to/figures/'
ncdir = '/absolute/path/to/wrfoutdata/'
ncf = = 'wrfout_d01...'

for t in times:
    p.plot2D('cref',t,level,outdir=outdir,ncdir=ncdir,ncf=ncf,
                legend=True)
