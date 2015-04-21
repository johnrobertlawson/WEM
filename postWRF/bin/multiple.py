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

outroot = '/home/jrlawson/public_html/bowecho/'
ncroot = '/chinook2/jrlawson/bowecho/'

p = WRFEnviron()

# enstype = 'STCH5'
enstype = 'STCH'
# enstype = 'ICBC'
# enstype = 'MXMP'
# enstype = 'STMX'

# case = '2006052512'
# case = '20060526'
# case = '2006052612'
#case = '20090910'
# case = '20110419'
case = '20130815'

IC = 'GEFSR2'; ens = 'p04'
# IC = 'NAM'; ens = 'anl'
# IC = 'RUC'
# IC = 'GFS'; ens = 'anl'
# IC = 'RUC'

if enstype == 'STCH':
    experiments = ['ICBC',]+['s'+"%02d" %n for n in range(1,11)]
    ensnames = [ens,]
    MP = 'ICBC'
elif enstype == 'STCH5':
    experiments = ['ICBC',]+['ss'+"%02d" %n for n in range(1,11)]
    ensnames = [ens,]
    MP = 'WDM6_Grau'
elif enstype == 'STMX':
    experiments = ['ICBC','WSM6_Grau_STCH','WSM6_Hail_STCH','Kessler_STCH',
                    'Ferrier_STCH', 'WSM5_STCH','WDM5_STCH','Lin_STCH',
                    'WDM6_Grau_STCH','WDM6_Hail_STCH',
                    'Morrison_Grau_STCH','Morrison_Hail_STCH',]
    ensnames = [ens,]
elif enstype == 'MXMP':
    experiments = ['ICBC','WSM6_Grau','WSM6_Hail','Kessler','Ferrier',
                    'WSM5','WDM5','Lin','WDM6_Grau','WDM6_Hail',
                    'Morrison_Grau','Morrison_Hail']
    ensnames = [ens,]
elif enstype == 'ICBC':
    ensnames =  ['c00'] + ['p'+"%02d" %n for n in range(1,11)]
    experiments = ['ICBC',]
else:
    raise Exception

if case[:4] == '2006':
    inittime = (2006,5,26,0,0,0)
    itime = (2006,5,26,0,0,0)
    ftime = (2006,5,27,12,0,0)
    iwind = (2006,5,26,22,0,0)
    fwind = (2006,5,27,9,0,0)
    compt = [(2006,5,d,h,0,0) for d,h in zip((26,27,27),(23,3,6))]
    matchnc = '/chinook2/jrlawson/bowecho/20060526/GFS/anl/ICBC/wrfout_d01_2006-05-26_00:00:00'
    # times = [(2006,5,26,12,0,0),]
elif case[:4] == '2009':
    inittime = (2009,9,10,23,0,0)
    itime = (2009,9,10,18,0,0)
    ftime = (2009,9,11,12,0,0)
elif case[:4] == '2011':
    inittime = (2011,4,19,0,0,0)
    itime = (2011,4,20,1,0,0)
    ftime = (2011,4,20,9,0,0)
    iwind = (2011,4,19,21,0,0)
    fwind = (2011,4,20,9,0,0)
    matchnc = '/chinook2/jrlawson/bowecho/20110419/GEFSR2/c00/ICBC/wrfout_d01_2011-04-19_00:00:00'
elif case[:4] == '2013':
    inittime = (2013,8,15,0,0,0)
    itime = (2013,8,15,0,0,0)
    ftime = (2013,8,16,12,0,0)
    iwind = (2013,8,15,21,0,0)
    fwind = (2013,8,16,11,0,0)
    # times = [(2013,8,16,3,0,0),]
    compt = [(2013,8,d,h,0,0) for d,h in zip((15,16,16),(22,2,6))]
else:
    raise Exception

hourly = 1
level = 2000

def get_folders(en,ex,ic=IC):
    if enstype[:4] == 'STCH':
        if ic=='RUC':
            mp = ''
        else:
            mp = MP

        out_sd = os.path.join(outroot,case,ic,en,mp,ex)
        wrf_sd = os.path.join(ncroot,case,ic,en,mp,ex)
        sp_sd = os.path.join(outroot,case,ic,en,mp)
    elif enstype == 'ICBC':
        out_sd = os.path.join(outroot,case,ic,en,ex)
        wrf_sd = os.path.join(ncroot,case,ic,en,ex)
        sp_sd = os.path.join(outroot,case,ic)
    else:
        out_sd = os.path.join(outroot,case,ic,en,ex)
        wrf_sd = os.path.join(ncroot,case,ic,en,ex)
        sp_sd = os.path.join(outroot,case,ic,en)
    return out_sd, wrf_sd, sp_sd

def get_verif_dirs():
    outdir = os.path.join(outroot,case,'VERIF')
    datadir = os.path.join(ncroot,case,'VERIF')
    return outdir,datadir

times = utils.generate_times(itime,ftime,hourly*60*60)
# times = ((2013,8,16,3,0,0),)
# times = ((2006,5,27,6,0,0),)
# Multiple plots
# GEFS ICBC has 11 + verif = 12
# MXMP has 12 + verif = 13
# STMX has 12 + verif = 13
# STCH has 11 + verif = 12
if (enstype=='ICBC') or (enstype[:4]=='STCH'):
    nrow = 3
    ncol = 5
else:
    nrow = 3
    ncol = 5

if enstype=='ICBC':
    enslist = ['NEXRAD','RUC'] + ensnames
    exlist = experiments * len(enslist)
    plotlist = enslist
else:
    exlist = ['NEXRAD','RUC'] + experiments
    enslist = ensnames * len(exlist)
    plotlist = exlist

outdir,datadir = get_verif_dirs()

for t in times:
    fig,axes = plt.subplots(nrow,ncol,figsize=(9,6))
    for ax,plot,en,ex in zip(axes.flat,plotlist,enslist,exlist):
        cb = False
        if plot=='NEXRAD':
            # Verification
            outdir, ncdir, sp_outdir = get_folders(enslist[-1],exlist[-1])
            print("Looking for data in {0}".format(ncdir))
            p.plot_radar(t,datadir,outdir=outdir,ncdir=ncdir,fig=fig,ax=ax,cb=cb)
        else:
            if plot=='RUC':
                outdir, ncdir, sp_outdir = get_folders('anl','VERIF',ic='RUC')
                nct = t
                mnc = matchnc
                sm = 1
                cb_done = False
            else:
                if not cb_done:
                    # cb = fig.add_axes([.9,.1,.05,.1]) 
                    cb = axes.flat[-1]
                    # cb, kw = M.colorbar.make_axes(cb,shrink=0.2)
                else:
                    cb = False
                nct = inittime
                mnc = False
                sm = 7
                outdir, ncdir, sp_outdir = get_folders(en,ex)
                if plot=='ICBC' and enstype[:4] =='STCH':
                    ncdir = os.path.dirname(ncdir)
            print("Looking for data in {0}".format(ncdir))

            tstr = t
            # pdb.set_trace()
            ##### SET NAME #####
            other = False
            pt = 'contourf'
            # vrbl = 'PMSL';lv = False;clvs=N.arange(900,1100,4)*10**2;pt='contour';sm=sm*5
            vrbl = 'strongestwind';lv=False;clvs = N.arange(10,31,1); tstr=False
            # vrbl = 'RH';lv = 700;clvs=N.arange(0,105,5)
            # vrbl = 'Q2';lv = False;clvs=N.arange(1,20.5,0.5)*10**-3
            # vrbl = 'Z';lv = 300;clvs=N.arange(8400,9600,30); pt='contour';sm=sm*3
            # vrbl = 'Z';lv = 700;clvs=N.arange(2800,3430,30); pt='contour';sm=sm*3
            # vrbl='Z';lv=500;clvs=N.arange(5000,6000,50);pt='contour';sm=sm*3
            # vrbl = 'T2';lv = False;clvs=N.arange(280,316,1)
            # vrbl = 'shear';lv = False;clvs=N.arange(0,31,1); other = {'top':6,'bottom':0}
            # vrbl = 'frontogen';lv = 2000; clvs = N.arange(-1.5,1.6,0.1)*10**-7
            # vrbl = 'cref';lv = False;clvs=False;sm=False
            ######### COMMAND HERE #########
            # if plot!='RUC': cb = p.plot2D(vrbl,utc=t,level=lv,ncdir=ncdir,outdir=outdir,fig=fig,ax=ax,cb=cb,clvs=clvs,nct=nct,match_nc=mnc,other=other,smooth=sm,plottype=pt)
            # cb = p.plot2D(vrbl,utc=t,level=lv,ncdir=ncdir,outdir=outdir,fig=fig,ax=ax,cb=cb,clvs=clvs,nct=nct,match_nc=mnc,other=other,smooth=sm,plottype=pt)
            # p.frontogenesis(utc=t,level=lv,ncdir=ncdir,outdir=outdir,clvs=clvs,smooth=5,fig=fig,ax=ax,cb=False,match_nc=mnc,nct=nct)
            if plot!='RUC': cb = p.plot_strongest_wind(iwind,fwind,ncdir=ncdir,outdir=outdir,clvs=clvs,fig=fig,ax=ax,cb=cb)
            ################################
        print("Plotting {0} panel".format(plot))
        ax.set_title(plot)
    if len(plotlist)==13:
        axes.flat[-2].axis('off')
    
    # axes.flat[-1].axis('off')
    if vrbl=='cref' or 'strongestwind':
        axes.flat[1].axis('off')

    # axes.flat[-1].colorbar(cb)
    # plt.colorbar(cb,cax=cax)
    # plt.colorbar(cb,cax=axes.flat[-1],use_gridspec=True)
    fig.tight_layout()
    fig.subplots_adjust(wspace = 0.1, hspace = 0.1)
    sp_fname = p.create_fname(vrbl,utc=tstr,level=lv,f_prefix=IC,f_suffix=enstype)
    sp_fpath = os.path.join(sp_outdir,sp_fname)
    fig.savefig(sp_fpath)
    plt.close(fig)
    print("Saving figure to {0}".format(sp_fpath))
