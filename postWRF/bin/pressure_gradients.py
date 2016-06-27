"""
Compare pressure gradients in SINGLE and NESTED
"""

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

# vrbl = 'T2_gradient'
# vrbl = 'PMSL_gradient'
# vrbl = 'cref'
# vrbl = 'RH'
# vrbl = 'Q_pert'
# vrbl = 'shear'
# vrbl = 'wind'
vrbl = 'wind10'

# ncroot = '/chinook2/jrlawson/bowecho/20130815/GEFSR2_paper1/p09'
ncdir = {'SINGLE':'/chinook2/jrlawson/bowecho/20130815/GEFSR2_paper2/p09/ICBC/s06',
        'NESTED':'/chinook2/jrlawson/bowecho/20130815_hires/s22'}
ncfile = 'wrfout_d01_2013-08-15_00:00:00'
nct = (2013,8,15,0,0,0)
itime = (2013,8,15,19,0,0)
ftime = (2013,8,15,23,0,0)
# ftime = (2013,8,16,4,0,0)
interval = 60*60*1
times = utils.generate_times(itime,ftime,interval)
outdir = '/home/jrlawson/public_html/bowecho/paper2'
lims = {'Nlim':41.0,'Elim':-96.0,'Slim':36.0,'Wlim':-102.0}

PLOTS = {
        'PMSL_gradient':{'lv':2000,'clvs':N.arange(0.02,0.18,0.02),'cbl':'PMSL gradient (Pa/m)','cmap':'cubehelix_r','extend':'max'},
        'T2_gradient':{'lv':2000,'clvs':N.arange(0.0002,0.0024,0.0002),'cbl':'2-m pot. temp gradient (K/m)','cmap':'cubehelix_r','extend':'max'},
        'cref':{'lv':False,'clvs':False,'cbl':'Simulated composite reflectivity (dBZ)','cmap':False,'extend':False},
        'RH':{'lv':600,'clvs':N.arange(0,105,5),'cbl':'Relative Humidity (%)','cmap':'terrain_r','extend':'max'},
        'Q_pert':{'lv':700,'clvs':N.arange(-0.002,0.0021,0.0001),'cbl':'Vapor mixing ratio perturbation (kg/kg)','cmap':'BrBG','extend':'both'},
        'shear':{'lv':False,'clvs':N.arange(5,32.5,2.5),'cbl':'0--6 km vertical wind shear (m/s)','cmap':'YlGnBu','extend':'max'},
        'wind':{'lv':500,'clvs':N.arange(5,32.5,2.5),'cbl':'Wind speed (m/s)','cmap':'YlGnBu','extend':'max'},
        'wind10':{'lv':False,'clvs':N.arange(2,20.1,0.1),'cbl':'Wind speed (m/s)','cmap':'YlGnBu','extend':'max'},
        }

cb_opt = False
p = WRFEnviron()
ncols = len(times)
nrows = 2
fig, axes = plt.subplots(nrows,ncols,figsize=(6,4))
axit = iter(axes.flat)
labels = iter(list(string.ascii_lowercase))

def make_subplot_label(ax,label):
    if not label.endswith(')'):
        label = label + ')'
    ax.text(0.05,0.15,label,transform=ax.transAxes,
        bbox={'facecolor':'white'},fontsize=15,zorder=1000)
    return

for nest in ('SINGLE','NESTED'):
    for tn,t in enumerate(times):

        dt = datetime.datetime(*time.gmtime(t)[:-2])
        nicetime = '{0:%H:%M UTC %Y/%m/%d}'.format(dt)
        # Verification
        ax = next(axit)
        if nest == 'SINGLE':
            ax.set_title(nicetime)

        # if vrbl is not 'wind':
        cb = p.plot2D(vrbl,utc=t,level=PLOTS[vrbl]['lv'],ncdir=ncdir[nest],outdir=False,fig=fig,ax=ax,cb=cb_opt,
                clvs=PLOTS[vrbl]['clvs'],nct=nct,plottype='contourf',cmap=PLOTS[vrbl]['cmap'],
                extend=PLOTS[vrbl]['extend'],save=False,dom=1,**lims)
        # else:
            # cb = p.plot2D(vrbl,utc=t,level=PLOTS[vrbl]['lv'],ncdir=ncdir[nest],outdir=False,fig=fig,ax=ax,cb=cb_opt,
                    # clvs=PLOTS[vrbl]['clvs'],nct=nct,plottype='quiver',cmap=PLOTS[vrbl]['cmap'],
                    # extend=PLOTS[vrbl]['extend'],save=False,dom=1,**lims)
            # lims = False
            # cb = p.plot_streamlines(t,PLOTS[vrbl]['lv'],ncdir[nest],outdir=False,fig=fig,ax=ax,nct=nct,
                       # bounding=lims,dom=1,density=0.8)
        make_subplot_label(ax,next(labels))
        if tn == len(times)-1:
            ax.text(1.1,0.5,nest,transform=ax.transAxes)

fig.tight_layout()
fig.subplots_adjust(bottom=0.144,right=0.89)
# if vrbl is not 'wind':
cbar_ax = fig.add_axes([0.15,0.097,0.7,0.025])
cbx = plt.colorbar(cb,cax=cbar_ax,orientation='horizontal')#,extend='both')
cbx.set_label(PLOTS[vrbl]['cbl'])

fname = '{0}_KSOK13.png'.format(vrbl)
fpath = os.path.join(outdir,fname)
utils.trycreate(outdir)
fig.savefig(fpath)
plt.close(fig)

