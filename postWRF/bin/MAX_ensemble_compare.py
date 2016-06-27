"""
Generate figure for publication showing
SKEB vs no SKEB and hail/graupel differences
for bow echoes.

Reflectivity, cold pool, max surface wind,
wind above surface.
"""
import pdb
import os
import numpy as N
import sys
import matplotlib.pyplot as plt
sys.path.append('/home/jrlawson/gitprojects/')
import time
import datetime

from WEM.postWRF.postWRF import WRFEnviron
import WEM.utils as utils
from WEM.postWRF.postWRF.wrfout import WRFOut

PLOTS = {
        'cref':{'lv':False,'clvs':False,'cbl':'Simulated composite reflectivity (dBZ)'},
        'WSPD10MAX':{'lv':False,'clvs':N.arange(10,32.5,2.5),'extend':'max','cbl':'Maximum 10-m wind speed ($m s^{-1}$)'},
        # 'wind10':{'lv':False,'clvs':N.arange(10,32.5,2.5),'extend':'max'},
        'wind':{'lv':850,'clvs':N.arange(15,47.5,2.5),'extend':'max','cbl':'850-hPa wind speed ($m s^{-1}$)'},
        # 'T2p':{'lv':2000,'clvs':N.arange(-6,0.5,0.5),'cmap':'ocean','extend':'min'},
        'dptp':{'lv':2000,'clvs':N.arange(-20,1,1),'cmap':'terrain','extend':'min','cbl':'Density potential temperature perturbation (K)'}}
ncroot = '/chinook2/jrlawson/bowecho/20130815/GEFSR2_paper1/p09'
members = ['WSM6_Grau','WSM6_Grau_MAX','WSM6_Grau_STCH','WSM6_Grau_STCH_MAX',
            'WSM6_Hail','WSM6_Hail_MAX','WSM6_Hail_STCH','WSM6_Hail_STCH_MAX',
            'WDM5','WDM5_MAX','WDM5_STCH','WDM5_STCH_MAX']
ncfile = 'wrfout_d01_2013-08-15_00:00:00'
nct = (2013,8,15,0,0,0)
# itime = (2013,8,15,17,0,0)
itime = (2013,8,16,3,0,0)
# ftime = (2013,8,16,8,0,0)
ftime = (2013,8,16,4,0,0)
# interval = 60*20
interval = 60*60

times = utils.generate_times(itime,ftime,interval)
outdir = '/home/jrlawson/public_html/bowecho/paper1/all_MAX/'
# outdir = '/home/jrlawson/public_html/bowecho/paper1/all_MAX/animate'

MAXmembers = [x for x in members if x.endswith('MAX')]

# lims = {'Nlim':42.0,'Elim':-95.0,'Slim':34.0,'Wlim':-104.0}
lims = {'Nlim':39.5,'Elim':-95.6,'Slim':35.8,'Wlim':-101.5}

def make_subplot_label(ax,label):
    ax.text(0.05,0.15,label,transform=ax.transAxes,
        bbox={'facecolor':'white'},fontsize=15,zorder=1000)
    return

cb_saved = {}
for vrbl in list(PLOTS.keys()):
    cb_saved[vrbl] = False
p = WRFEnviron()
# Compare old and new runs
labels = ['a)','b)','c)','d)','e)','f)']

for tn,t in enumerate(times):
    nicetime = datetime.datetime(*time.gmtime(t)[:-2])
    for vrbl,PL in PLOTS.items():
        fig, axes = plt.subplots(3,2,figsize=(6,9))
        for n,ens in enumerate(MAXmembers):
            ncdir = os.path.join(ncroot,ens)
            ax = axes.flat[n]
            title = ens[:-4].replace('_',' ')
            ax.set_title(title)
            make_subplot_label(ax,labels[n])

            try:
                cm = PL['cmap']
            except KeyError:
                cm = False

            try:
                extend = PL['extend']
            except KeyError:
                extend = False

            cb = p.plot2D(vrbl,utc=t,level=PL['lv'],ncdir=ncdir,outdir=False,fig=fig,ax=ax,cb=False,
                    clvs=PL['clvs'],nct=nct,plottype='contourf',cmap=cm,extend=extend,save=False,
                    **lims)

            if not cb_saved[vrbl]:
                p.plot2D(vrbl,utc=t,level=PL['lv'],ncdir=ncdir,outdir=outdir,
                        f_suffix=False,cb='only',
                        clvs=PL['clvs'],nct=nct,plottype='contourf',cmap=cm,extend=extend,
                        **lims)
                # if vrbl is not 'dptp':
                cb_saved[vrbl] = True

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.12,top=0.94)
        # cbar_ax = fig.add_axes([0.15,0.075,0.7,0.025])
        cbar_ax = fig.add_axes([0.15,0.06,0.7,0.02])
        cb1 = plt.colorbar(cb,cax=cbar_ax,orientation='horizontal')#,extend='both')
        cb1.set_label(PL['cbl'])

        fig.suptitle('{0:%H:%M UTC %Y/%m/%d} (Day {1})'.format(nicetime,int(nicetime.day)-14))

        # fname = 'MAX_runs_compare_{0}_{1:%Y%m%d%H}.png'.format(vrbl,nicetime)
        fname = 'MAX_runs_compare_{0}_{1:03d}.png'.format(vrbl,tn)
        fpath = os.path.join(outdir,fname)
        utils.trycreate(outdir)
        fig.savefig(fpath)
        plt.close(fig)
        print(("Saved figure for {0} at {1:%Y%m%d %H:%M}.".format(vrbl,nicetime)))

