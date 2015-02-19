import os
import pdb
import sys
import matplotlib as M
M.use('gtkagg')
import matplotlib.pyplot as plt
import numpy as N

sys.path.append('/home/jrlawson/gitprojects/')

from WEM.postWRF.postWRF import WRFEnviron
import WEM.utils as utils
#from WEM.postWRF.postWRF.rucplot import RUCPlot

outroot = '/home/jrlawson/public_html/bolt/'
ncroot = '/chinook2/jrlawson/bolttest/'

p = WRFEnviron()

meteogram = 0
plot2D = 1
nens = 9

itime = (2015,2,7,0,0,0)
ftime = (2015,2,8,1,0,0)
hourly = 1
times = utils.generate_times(itime,ftime,hourly*60*60)

ncfiles = [os.path.join(ncroot,'SGp{0:02d}'.format(n)) for n in range(1,11)]
ncfiles.append(os.path.join(ncroot,'Gm4km'))

vrbls = ['T2','wind10']

loclist = [{'Stockton':(54.57,-1.32)},
            {'Norwich':(52.63,1.30)},
            {'FortWill':(56.82,-5.11)},
            {'Wick':(58.45,-3.09)},
            ]
            
if meteogram:
    for loc in loclist:
        for vrbl in vrbls:
            p.meteogram(vrbl,loc,ncfiles,outdir=outroot)

if plot2D:
    for t in times:
        p.plot2D('T2',t,ncdir=ncfiles[nens],outdir=outroot,cb=True,clvs=N.arange(263,284,1),f_suffix='{0}'.format(nens))
