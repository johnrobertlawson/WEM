import matplotlib as M
M.use('agg')
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
from WEM.postWRF.postWRF.birdseye import BirdsEye

# vrbl = 'T2_gradient'
# vrbl = 'PMSL_gradient'
# vrbl = 'cref'
# vrbl = 'RH'
# vrbl = 'Q_pert'
# vrbl = 'shear'
# vrbl = 'wind'
# vrbl = 'wind10'

lv = 800

# ncroot = '/chinook2/jrlawson/bowecho/20130815/GEFSR2_paper1/p09'
ncdir = {'SINGLE':'/chinook2/jrlawson/bowecho/20130815/GEFSR2_paper2/p09/ICBC/',
        'NESTED':'/chinook2/jrlawson/bowecho/20130815_hires/'}
ncfile = 'wrfout_d01_2013-08-15_00:00:00'
nct = (2013,8,15,0,0,0)
# itime = (2013,8,15,20,0,0)
# ftime = (2013,8,15,23,0,0)
# interval = 1*60*60
# times = utils.generate_times(itime,ftime,interval)
utc = (2013,8,15,21,0,0)
outdir = '/home/jrlawson/public_html/bowecho/paper2'
lims = {'Nlim':41.0,'Elim':-97.4,'Slim':38.3,'Wlim':-101.0}

DATA = {}

# cross-section labels
latA = 40.3
lonA = -99.3
latB = 39.0
lonB = -98.2

for nest in ("SINGLE","NESTED"):
    DATA[nest] = {}
    DATA[nest]['W'] = WRFOut(os.path.join(ncdir[nest],ncfile))
    # import pdb; pdb.set_trace()
    DATA[nest]['U'] = DATA[nest]['W'].get('U',utc=utc,level=lv)[0,0,:,:]
    DATA[nest]['U'],lats,lons = utils.return_subdomain(DATA[nest]['U'],
                    DATA[nest]['W'].lats1D,DATA[nest]['W'].lons1D,
                    fmt='latlon',**lims)
    DATA[nest]['V'] = DATA[nest]['W'].get('V',utc=utc,level=lv)[0,0,:,:]
    DATA[nest]['V'],lats,lons = utils.return_subdomain(DATA[nest]['V'],
                    DATA[nest]['W'].lats1D,DATA[nest]['W'].lons1D,
                    fmt='latlon',**lims)

U_diff = DATA['SINGLE']['U'] - DATA['NESTED']['U']
V_diff = DATA['SINGLE']['V'] - DATA['NESTED']['V']

# difference plot
diffplot = 0
if diffplot:
    fig,ax = plt.subplots(1,figsize=(4,4))
    F = BirdsEye(ax=ax,fig=fig)
    F.bmap,F.x,F.y = F.basemap_setup(lats=lats,lons=lons,)
    n = 3
    U_diff2 = -1* U_diff[::n,::n]
    V_diff2 = -1* V_diff[::n,::n]
    x = F.x[::n,::n]
    y = F.y[::n,::n]
    F.bmap.quiver(x,y,U_diff2,V_diff2,scale=500)
    windmag = N.sqrt(U_diff**2 + V_diff**2)
    clvs = N.arange(5,32.5,2.5)
    cb = F.bmap.contourf(F.x,F.y,windmag,alpha=0.5,cmap=M.cm.copper_r,levels=clvs,extend='max')

    # F.bmap.plot((lonA,lonB),(latA,latB),latlon=True,color='k')

    # fig.subplots_adjust(bottom=0.12,right=0.9)
    # cbar_ax = fig.add_axes([0.15,0.087,0.7,0.025])
    # cbar_ax = False
    # cbx = plt.colorbar(cb,ticks=False,cax=cbar_ax,orientation='horizontal')#,extend='both')
    cbx = plt.colorbar(cb,orientation='horizontal',shrink=0.5)
    cbx.set_label('Wind difference (m/s)')
    fig.tight_layout()

    fname = 'diff_wind.png'
    fpath = os.path.join(outdir,fname)
    utils.trycreate(outdir)
    fig.savefig(fpath)
    print(("Saved at {0}.".format(fpath)))
    plt.close(fig)

# Two together
for nest in ('SINGLE','NESTED'):
    fig,ax = plt.subplots(1,figsize=(4,4))
    F = BirdsEye(ax=ax,fig=fig)
    F.bmap,F.x,F.y = F.basemap_setup(lats=lats,lons=lons,)
    xpts, ypts = F.bmap([lonB,lonA],[latB,latA])
    F.bmap.plot(xpts,ypts,latlon=False,color='k')
    n = 3
    U = DATA[nest]['U']
    U2 = U[::n,::n]
    V = DATA[nest]['V']
    V2 = V[::n,::n]
    x2 = F.x[::n,::n]
    y2 = F.y[::n,::n]
    F.bmap.quiver(x2,y2,U2,V2,scale=500)
    windmag = N.sqrt(U**2 + V*2)
    clvs = N.arange(5,32.5,2.5)
    cb = F.bmap.contourf(F.x,F.y,windmag,alpha=0.5,cmap=M.cm.copper_r,levels=clvs,extend='max')


    # fig.subplots_adjust(bottom=0.12,right=0.9)
    # cbar_ax = fig.add_axes([0.15,0.087,0.7,0.025])
    # cbar_ax = False
    # cbx = plt.colorbar(cb,ticks=False,cax=cbar_ax,orientation='horizontal')#,extend='both')
    cbx = plt.colorbar(cb,orientation='horizontal',shrink=0.5)
    cbx.set_label('Wind speed (m/s)')
    fig.tight_layout()

    fname = '{0}_wind.png'.format(nest)
    fpath = os.path.join(outdir,fname)
    utils.trycreate(outdir)
    fig.savefig(fpath)
    print(("Saved at {0}.".format(fpath)))
    plt.close(fig)

