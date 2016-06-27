
import pdb
import os
import numpy as N
import sys
import matplotlib.pyplot as plt
sys.path.append('/home/jrlawson/gitprojects/')
import time
import datetime
import string

from WEM.postWRF.postWRF import WRFEnviron
import WEM.utils as utils
from WEM.postWRF.postWRF.wrfout import WRFOut

ncdir = {'SINGLE':'/chinook2/jrlawson/bowecho/20130815/GEFSR2_paper2/p09/ICBC/',
        'NESTED':'/chinook2/jrlawson/bowecho/20130815_hires/'}
ncfile = 'wrfout_d01_2013-08-15_00:00:00'
nct = (2013,8,15,0,0,0)
# itime = (2013,8,15,20,0,0)
# ftime = (2013,8,15,23,0,0)
# interval = 1*60*60
# times = utils.generate_times(itime,ftime,interval)
outdir = '/home/jrlawson/public_html/bowecho/paper2'
# lims = {'Nlim':41.0,'Elim':-97.4,'Slim':38.3,'Wlim':-101.0}
latA = 40.3
lonA = -99.3
latB = 39.0
lonB = -98.2

# latA = 39.6
# latB = 39.7

vrbl = 'Q_pert'; clvs=N.arange(-0.007,0.0071,0.0005);cftix=N.arange(-0.007,0.0105,0.0035);cmap='BrBG';ctvrbl='parawind';ctclvs=N.arange(-25,40,5)
utc = (2013,8,15,21,0,0)
dom = 1

p = WRFEnviron()
for nest in list(ncdir.keys()):
    p.plot_xs(vrbl,utc,ncdir[nest],outdir,latA=latA,lonA=lonA,
        latB=latB,lonB=lonB,nct=nct,dom=dom,f_suffix=nest,
        clvs=clvs,cmap=cmap,contour_vrbl=ctvrbl,
        contour_clvs=ctclvs,cftix=cftix,avepts=2,
        cflabel='Water mixing ratio perturbation ($kg\,kg^{-1}$)')

