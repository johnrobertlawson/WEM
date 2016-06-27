import os
import pdb
import sys
import matplotlib as M
M.use('gtkagg')
import matplotlib.pyplot as plt
import numpy as N
from mpl_toolkits.basemap import Basemap

sys.path.append('/home/jrlawson/gitprojects/')

from WEM.postWRF.postWRF import WRFEnviron
import WEM.utils as utils
import WEM.postWRF.postWRF.stats as stats
from WEM.postWRF.postWRF.wrfout import WRFOut
#from WEM.postWRF.postWRF.rucplot import RUCPlot

case = '20130815'
BOW = {}
BOW['hires3'] = {'dom':1,'ncf':'wrfout_d01_2013-08-15_00:00:00',
                'ensnames':['s{0:02d}'.format(e) for e in range(21,31)],
                'ncroot':'/chinook2/jrlawson/bowecho/20130815_hires/'}
BOW['hires1'] = {'dom':2,'ncf':'wrfout_d02_2013-08-15_00:00:00',
                'ensnames':['s{0:02d}'.format(e) for e in range(21,31)],
                'ncroot':'/chinook2/jrlawson/bowecho/20130815_hires/'}
BOW['normal'] = {'dom':1,'ncf':'wrfout_d01_2013-08-15_00:00:00',
                'ensnames':['s{0:02d}'.format(e) for e in range(1,11)],
                'ncroot':'/chinook2/jrlawson/bowecho/20130815/GEFSR2/p09/ICBC/'}


outroot = '/home/jrlawson/public_html/bowecho/hires/{0}'.format(case)
outdir = outroot

p = WRFEnviron()

nct = (2013,8,15,0,0,0)
itime = (2013,8,15,0,0,0)
ftime = (2013,8,16,12,0,0)

hourly = 1
level = 2000
times = utils.generate_times(itime,ftime,hourly*60*60)

# Four subplots of std at 6h, 12h, 18h, 24h
# For original 3km, hires 3km, hires 1km.

titlets = ['6h','12h','18h','24h']
# vrbl = 'wind10'; clvs = N.arange(0.25,5.25,0.25); lv = False
# vrbl = 'T2'; clvs = N.arange(0.25,5.25,0.25); lv = False
vrbl = 'cref'; lv= False
# vrbl = 'Q2'; lv = False
# vrbl = 'U10'; lv = False

# Dictionary for line plots of std over time
std_line = {}

fourpanel = 0
lines = 1
npts = 250

choicedone = 0
for pn,plot in enumerate(BOW.keys()):
    fig, axes = plt.subplots(2,2)
    ncfiles = [os.path.join(BOW[plot]['ncroot'],e,BOW[plot]['ncf']) 
                    for e in BOW[plot]['ensnames']]
    if lines and not fourpanel:
        if plot is "hires1":
            W1k = WRFOut(ncfiles[0])
            continue
    # 2 experiments, four times, 250 random points
    std_line[plot] = N.zeros([len(times),npts])
    for nt,t in (list(zip(list(range(len(times))),times))): 
        
        print(("Time",nt))
        W = WRFOut(ncfiles[0])
        if fourpanel:
            titlet = titlets[nt]
            ax = axes.flat[nt]
            width_m = W.dx*(W.x_dim-1)
            height_m = W.dy*(W.y_dim-1)

            m = Basemap(
                projection='lcc',width=width_m,height=height_m,
                lon_0=W.cen_lon,lat_0=W.cen_lat,lat_1=W.truelat1,
                lat_2=W.truelat2,resolution='i',area_thresh=500,
                ax=ax)
            m.drawcoastlines()
            m.drawstates()
            m.drawcountries()
            x,y = m(W.lons,W.lats)
        std = stats.std(ncfiles,vrbl,utc=t,level=lv)[0,0,:,:]
        if fourpanel:
            pt = m.contourf(x,y,std,levels=clvs)
            m.colorbar(pt)
            ax.set_title(titlet)

        # Save some points for line plot

        if lines:
            if (nt == 0) and (choicedone==0):
                # pick 100 random latidx and lonidx
                latidx = N.random.choice(std.shape[-1],npts)
                lonidx = N.random.choice(std.shape[-1],npts)
                choicedone = 1
            std_line[plot][nt,...] = std[latidx,lonidx]

    if fourpanel:
        fname = 'std_{0}_{1}.png'.format(vrbl,plot)
        fpath = os.path.join(outdir,fname)
        fig.savefig(fpath)
        print(("Saved to {0}".format(fpath)))
        plt.close(fig)

if lines:
    fig,ax = plt.subplots(1)
    # for n in range(100):
    ax.plot(std_line['hires3'],color='lightcoral',lw=0.5,label='Nested')
    ax.plot(std_line['normal'],color='lightblue',lw=0.5,label='Single')
    ax.plot(N.mean(std_line['hires3'],axis=1),color='red',lw=3,label='Nested mean')
    ax.plot(N.mean(std_line['normal'],axis=1),color='blue',lw=3,label='Single mean')

    ax.set_xlabel("Forecast time (hr)")
    ax.set_ylabel("Standard deviation")
    handles, labels = ax.get_legend_handles_labels()
    labelpick = labels[0:1] + labels[-3:]
    handlepick = handles[0:1] + handles[-3:]
    ax.legend(handlepick, labelpick,loc=2)

    fname = 'std_lines_{0}.png'.format(vrbl)
    fpath = os.path.join(outdir,fname)
    fig.savefig(fpath)
    print(("Saved to {0}".format(fpath)))
    plt.close(fig)

    # Get lat/lon limits of 1-km domain
    Nl,El,Sl,Wl = W1k.get_limits()

    latpts = W.lats[latidx,lonidx]
    lonpts = W.lons[latidx,lonidx]

    aa = N.where(latpts < Nl)
    bb = N.where(latpts > Sl)
    cc = N.where(lonpts > Wl)
    dd = N.where(lonpts < El)

    ee = N.intersect1d(aa,bb)
    ff = N.intersect1d(cc,dd)
    # indices of those locs within box
    gg = N.intersect1d(ee,ff) 

    fig,ax = plt.subplots(1)
    ax.plot(std_line['hires3'][:,gg],color='lightcoral',lw=0.5,label='Nested')
    ax.plot(std_line['normal'][:,gg],color='lightblue',lw=0.5,label='Single')
    ax.plot(N.mean(std_line['hires3'][:,gg],axis=1),color='red',lw=3,label='Nested mean')
    ax.plot(N.mean(std_line['normal'][:,gg],axis=1),color='blue',lw=3,label='Single mean')

    ax.set_xlabel("Forecast time (hr)")
    ax.set_ylabel("Standard deviation")

    handles, labels = ax.get_legend_handles_labels()
    labelpick = labels[0:1] + labels[-3:]
    handlepick = handles[0:1] + handles[-3:]
    ax.legend(handlepick, labelpick,loc=2)

    fname = 'std_lines_{0}_1kmdomain.png'.format(vrbl)
    fpath = os.path.join(outdir,fname)
    fig.savefig(fpath)
    print(("Saved to {0}".format(fpath)))
    plt.close(fig)
