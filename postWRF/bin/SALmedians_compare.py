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

PLOTS = {
        'cref':{'lv':False,'clvs':False,'cbl':'Simulated composite reflectivity (dBZ)','cmap':False,'extend':False},
        # 'WSPD10MAX':{'lv':False,'clvs':N.arange(10,32.5,2.5),'extend':'max','cbl':'Maximum 10-m wind speed ($m s^{-1}$)'},
        # 'wind10':{'lv':False,'clvs':N.arange(10,32.5,2.5),'extend':'max'},
        'wind':{'lv':850,'clvs':N.arange(15,47.5,2.5),'extend':'max','cbl':'850-hPa wind speed ($m s^{-1}$)'},
        # 'T2p':{'lv':2000,'clvs':N.arange(-6,0.5,0.5),'cmap':'ocean','extend':'min'},
        'dptp':{'lv':2000,'clvs':N.arange(-18,1,1),'cmap':'terrain','extend':'min','cbl':'Density potential temperature perturbation (K)'}}
ncroot = '/chinook2/jrlawson/bowecho/20130815/GEFSR2_paper1/p09'
ncdir = {'SINGLE':'/chinook2/jrlawson/bowecho/20130815/GEFSR2_paper2/p09/ICBC/s06',
        'NESTED':'/chinook2/jrlawson/bowecho/20130815_hires/s22'}
plotnames = ['NEXRAD',] + ['s06 (SINGLE)','s12 (NESTED)']*2
ncfile = 'wrfout_d01_2013-08-15_00:00:00'
radar_datadir = os.path.join('/chinook2/jrlawson/bowecho/20130815/VERIF')
nct = (2013,8,15,0,0,0)
itime = (2013,8,15,21,0,0)
ftime = (2013,8,16,6,0,0)
interval = 60*60*3
times = utils.generate_times(itime,ftime,interval)
outdir = '/home/jrlawson/public_html/bowecho/paper2'

lims = {'Nlim':42.0,'Elim':-95.0,'Slim':34.0,'Wlim':-104.0}

def make_subplot_label(ax,label):
    if not label.endswith(')'):
        label = label + ')'
    ax.text(0.05,0.15,label,transform=ax.transAxes,
        bbox={'facecolor':'white'},fontsize=15,zorder=1000)
    return

# cb_saved = {}
# for vrbl in PLOTS.keys():
    # cb_saved[vrbl] = False
p = WRFEnviron()
# Compare old and new runs
labels = iter(list(string.ascii_lowercase))
pltitle = iter(plotnames)

nrows = len(times)
ncols = len(plotnames)
fig, axes = plt.subplots(nrows,ncols,figsize=(9,6))
axit = iter(axes.flat)
cb = {}
nicetimes = iter(['21 h','24 h','27 h'])

for tn,t in enumerate(times):
    print(("Creating row for time #{0} of {1}.".format(tn+1,len(times))))

    # Verification
    ax = next(axit)
    p.plot_radar(t,radar_datadir,ncdir=ncdir['SINGLE'],fig=fig,ax=ax,cb=False,nct=nct,dom=1,**lims)
    make_subplot_label(ax,next(labels))
    if tn == 0:
        ax.set_title(next(pltitle))

    # Sim cref
    for vrbl, nest in zip(('cref','cref','dptp','dptp'),('SINGLE','NESTED','SINGLE','NESTED')):
        print((vrbl, nest))
        ax = next(axit)
        cb[vrbl] = p.plot2D(vrbl,utc=t,level=PLOTS[vrbl]['lv'],ncdir=ncdir[nest],outdir=False,fig=fig,ax=ax,cb=False,
                clvs=PLOTS[vrbl]['clvs'],nct=nct,plottype='contourf',cmap=PLOTS[vrbl]['cmap'],
                extend=PLOTS[vrbl]['extend'],save=False,**lims)
        if tn == 0:
            ax.set_title(next(pltitle))
        make_subplot_label(ax,next(labels))
    ax.text(1.1,0.5,next(nicetimes),transform=ax.transAxes)

fig.tight_layout()
fig.subplots_adjust(bottom=0.13,right=0.92)
# cbar_ax = fig.add_axes([0.15,0.075,0.7,0.025])
cbar_ax1 = fig.add_axes([0.245,0.08,0.27,0.02])
cbar_ax2 = fig.add_axes([0.615,0.08,0.27,0.02])
for cbar_ax, vrbl in zip((cbar_ax1,cbar_ax2),('cref','dptp')):
    cbx = plt.colorbar(cb[vrbl],cax=cbar_ax,orientation='horizontal')#,extend='both')
    cbx.set_label(PLOTS[vrbl]['cbl'])

# fig.suptitle('{0:%H:%M UTC %Y/%m/%d} (Day {1})'.format(nicetime,int(nicetime.day)-14))

# fname = 'MAX_runs_compare_{0}_{1:%Y%m%d%H}.png'.format(vrbl,nicetime)
fname = 'SAL_medians_compare_KSOK13.png'
fpath = os.path.join(outdir,fname)
utils.trycreate(outdir)
fig.savefig(fpath)
plt.close(fig)

