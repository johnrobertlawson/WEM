"""
Compare WRF initialisations for GEFS/NAM 
data sets.
"""
import os
import pdb
import sys
import matplotlib as M
M.use('agg')
import matplotlib.pyplot as plt
import numpy as N

sys.path.append('/home/jrlawson/gitprojects/')

from WEM.postWRF.postWRF.wrfout import WRFOut
from WEM.postWRF.postWRF import WRFEnviron
import WEM.utils as utils

outroot = '/home/jrlawson/public_html/bowecho/'

NC = {'NEOK06':{
        'Gncdir':'/chinook2/jrlawson/bowecho/20060526/GEFSR2/c00/ICBC/',
        'Rncdir':'/chinook2/jrlawson/bowecho/20060526/RUC/anl/VERIF',
        'Nnc':'wrfout_d01_2006-05-26_00:00:00',
        'Gnc':'wrfout_d01_2006-05-26_00:00:00',
        'Rnc':'rap_130_20060526_0000_000.nc',
        'nct':(2006,5,26,0,0,0),
        'itime':(2006,5,26,0,0,0),
        'ftime':(2006,5,27,12,0,0),
        'Nncdir':'/chinook2/jrlawson/bowecho/20060526/NAM/anl/ICBC/'},
      'KSOK13':{
          'Gncdir':'/chinook2/jrlawson/bowecho/20130815/GEFSR2_paper1/c00/ICBC/',
          'Nncdir':'/chinook2/jrlawson/bowecho/20130815/NAM/anl/ICBC/',
          'Rncdir':'/chinook2/jrlawson/bowecho/20130815/RUC/anl/VERIF',
          'Nnc':'wrfout_d01_2013-08-15_00:00:00',
          'Gnc':'wrfout_d01_2013-08-15_00:00:00',
          'Rnc':'rap_130_20130815_0000_000.nc',
          'itime':(2013,8,15,0,0,0),
          'ftime':(2013,8,16,12,0,0),
          'nct':(2013,8,15,0,0,0)}}

import collections
PLOT = {'Z':collections.OrderedDict()}
PLOT['Z'][300] = N.arange(7800,9900,30)
PLOT['Z'][500] = N.arange(5000,6000,30)
PLOT['Z'][700] = N.arange(2800,3430,30)
nrow = 3
ncol = 3
pt = 'contour'
t = 0
clvs = False
sm = 5
hourly = 6
fname = 'GEFS_NAM_compare'

p = WRFEnviron()

for case in NC:
    times = utils.generate_times(NC[case]['itime'],NC[case]['ftime'],hourly*60*60)
    for tn, t in enumerate(times):
        mnc = os.path.join(NC[case]['Gncdir'],NC[case]['Gnc'])
        fig,axes = plt.subplots(nrow,ncol,figsize=(6,6))
        axn = 0
        for vrbl in list(PLOT.keys()):
            for lv,clvs in list(PLOT[vrbl].items()):
                for model,ncdir in zip(['RUC','GEFSR2','NAM'],(NC[case]['Rncdir'],NC[case]['Gncdir'],NC[case]['Nncdir'])):
                    if model=='RUC':
                        nct = t
                    else:
                        nct = NC[case]['nct']
                    cb = p.plot2D(vrbl,utc=t,level=lv,ncdir=ncdir,fig=fig,ax=axes.flat[axn],cb=False,clvs=clvs,nct=nct,smooth=sm,plottype=pt,save=False,match_nc=mnc)
                    axes.flat[axn].set_title('{0} {1}'.format(model,lv))
                    axn += 1
        fnameout = fname+'_{0:04d}_{1:02d}h.png'.format(NC[case]['nct'][0],tn*hourly)
        fpath = os.path.join(outroot,fnameout)
        fig.tight_layout()
        fig.savefig(fpath)
        print(('Saved to {0}'.format(fpath)))






