import os
import pdb
import sys
import matplotlib as M
M.use('agg')
import matplotlib.pyplot as plt
import numpy as N
from mpl_toolkits.basemap import Basemap

sys.path.append('/home/jrlawson/gitprojects/')

from WEM.postWRF.postWRF import WRFEnviron
import WEM.utils as utils
import WEM.postWRF.postWRF.stats as stats
from WEM.postWRF.postWRF.wrfout import WRFOut
#from WEM.postWRF.postWRF.rucplot import RUCPlot

# case = '20110419'
case = '20130815'

BOW = {}
if case[:4] == '2013':
    BOW['hires3'] = {'dom':1,'ncf':'wrfout_d01_2013-08-15_00:00:00',
                    'ensnames':['s{0:02d}'.format(e) for e in range(21,31)],
                    'ncroot':'/chinook2/jrlawson/bowecho/20130815_hires/'}
    BOW['normal'] = {'dom':1,'ncf':'wrfout_d01_2013-08-15_00:00:00',
                    'ensnames':['s{0:02d}'.format(e) for e in range(1,11)],
                    'ncroot':'/chinook2/jrlawson/bowecho/20130815/GEFSR2/p09/ICBC/'}
    BOW['hires1'] = {'dom':2,'ncf':'wrfout_d02_2013-08-15_00:00:00',
                    'ensnames':['s{0:02d}'.format(e) for e in range(21,31)],
                    'ncroot':'/chinook2/jrlawson/bowecho/20130815_hires/'}
    nct = (2013,8,15,0,0,0)
    itime = (2013,8,15,0,0,0)
    ftime = (2013,8,16,12,0,0)
else:
    BOW['hires3'] = {'dom':1,'ncf':'wrfout_d01_2011-04-19_00:00:00',
                    'ensnames':['s{0:02d}'.format(e) for e in range(21,31)],
                    'ncroot':'/chinook2/jrlawson/bowecho/20110419_hires/'}
    BOW['normal'] = {'dom':1,'ncf':'wrfout_d01_2011-04-19_00:00:00',
                    'ensnames':['s{0:02d}'.format(e) for e in range(1,11)],
                    'ncroot':'/chinook2/jrlawson/bowecho/20110419/NAM/anl/WSM5'}
    BOW['hires1'] = {'dom':2,'ncf':'wrfout_d02_2011-04-19_00:00:00',
                    'ensnames':['s{0:02d}'.format(e) for e in range(21,31)],
                    'ncroot':'/chinook2/jrlawson/bowecho/20110419_hires/'}
    nct = (2011,4,19,0,0,0)
    itime = (2011,4,19,0,0,0)
    ftime = (2011,4,20,12,0,0)


outdir = '/home/jrlawson/public_html/bowecho/hires/{0}'.format(case)


hourly = 1
level = 2000
times = utils.generate_times(itime,ftime,hourly*60*60)

# vrbl = 'wind10'; lv = False
vrbl = 'T2'; lv = False
# vrbl = 'cref'; lv= False
# vrbl = 'Q2'; lv = False
# vrbl = 'U10'; lv = False

th = 15

entire_domain = True
limited_domain = False

tstats = []
pvalues = []

if entire_domain:
    for plot in ['hires3','normal']:
        BOW[plot]['stdTS'] = []
    plot = 'normal'
    ncfiles1 = [os.path.join(BOW[plot]['ncroot'],e,BOW[plot]['ncf']) 
                    for e in BOW[plot]['ensnames']]
    plot = 'hires3'
    ncfiles2 = [os.path.join(BOW[plot]['ncroot'],e,BOW[plot]['ncf']) 
                    for e in BOW[plot]['ensnames']]
    del plot

    for nt,t in (list(zip(list(range(len(times))),times))): 
        
        print(("Time",nt))
    # t = False
        stdval1, stdval2, tstat, pvalue =  stats.std_ttest(ncfiles1,ncfiles2,vrbl,utc=t,level=lv,th=th)
        BOW['normal']['stdTS'].append(stdval1)
        BOW['hires3']['stdTS'].append(stdval2)
        # tstats.append(tstat)
        # pvalues.append(pvalue)
        # if nt == 9:
            # import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()

    fig,ax = plt.subplots(1)
    # for n in range(100):
    ax.plot(BOW['normal']['stdTS'],color='blue',lw=2,label='SINGLE')
    ax.plot(BOW['hires3']['stdTS'],color='red',lw=2,label='NESTED')

    # ax2 = ax.twinx()
    # ax2.plot(pvalues,color='black',lw=1,label='p Value')

    ax.set_xlabel("Forecast time (hr)")
    ax.set_ylabel("Standard deviation")
    # ax2.set_ylabel("p Value")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels,loc='lower center')

    fname = 'std_lines_entiredomain_{0}_{1}th.png'.format(vrbl,th)
    fpath = os.path.join(outdir,fname)
    fig.savefig(fpath)
    print(("Saved to {0}".format(fpath)))
    plt.close(fig)

if limited_domain:
    # Get lat/lon limits of 1-km domain
    W3fpath = os.path.join(BOW['hires3']['ncroot'],'s21',BOW[plot]['ncf'])
    W1fpath = os.path.join(BOW['hires1']['ncroot'],'s21',BOW[plot]['ncf']) 
    W1k = WRFOut(W1fpath)
    Nl,El,Sl,Wl = W1k.get_limits()
    # data, lats, lons = utils.return_subdomain(data,lats,lons,Nl,El,Sl,Wl)

    latpts = W3.lats[latidx,lonidx]
    lonpts = W3.lons[latidx,lonidx]
    import pdb; pdb.set_trace()

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
