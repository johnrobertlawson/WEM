import os
import pdb
import sys
import matplotlib as M
M.use('agg')
import matplotlib.pyplot as plt
import numpy as N

sys.path.append('/home/jrlawson/gitprojects/')

from WEM.postWRF.postWRF import WRFEnviron
import WEM.utils as utils
#from WEM.postWRF.postWRF.rucplot import RUCPlot

outroot = '/home/jrlawson/public_html/bowecho/'
ncroot = '/chinook2/jrlawson/bowecho/'

p = WRFEnviron()

# lims = {'Nlim':41.05,'Wlim':-89.65,'Slim':38.60,'Elim':-86.90}
lims = {'Nlim':42.0,'Wlim':-103.0,'Slim':35.0,'Elim':-95.0}
# lims = {}

# enstype = 'STCH5'
enstype = 'STCH'
# enstype = 'ICBC'
# enstype = 'MXMP'
# enstype = 'STMX'

# case = '20060526'
#case = '20090910'
# case = '20110419'
# case = '20110419_hires'
# case = '20130815_hires'
case = '20130815'
paper = 2
dom = 1

IC = 'GEFSR2'; ens = 'p09'; MP = 'ICBC'
# IC = 'NAM'; ens = 'anl'; MP = 'WSM5'
# IC = 'RUC'
# IC = 'GFS'; ens = 'anl'
# IC = 'RUC'

import string

def make_subplot_label(ax,label):
    if not label.endswith(')'):
        label = label + ')'
    ax.text(0.05,0.15,label,transform=ax.transAxes,
        bbox={'facecolor':'white'},fontsize=15,zorder=1000)
    return

if enstype == 'STCH':
    if case.endswith('hires'):
        experiments = ['ICBC',]+['s'+"%02d" %n for n in range(21,31)]
    else:
        experiments = ['ICBC',]+['s'+"%02d" %n for n in range(1,11)]
    ensnames = [ens,]
    # MP = 'ICBC'
    # MP = 'WSM5'
    labels = list(string.ascii_lowercase)[:12]
elif enstype == 'STCH5':
    experiments = ['ICBC',]+['ss'+"%02d" %n for n in range(1,11)]
    ensnames = [ens,]
    MP = 'WDM6_Grau'
    labels = list(string.ascii_lowercase)[:12]
elif enstype == 'STMX':
    experiments = ['ICBC','WSM6_Grau_STCH','WSM6_Hail_STCH','Kessler_STCH',
                    'Ferrier_STCH', 'WSM5_STCH','WDM5_STCH','Lin_STCH',
                    'WDM6_Grau_STCH','WDM6_Hail_STCH',
                    'Morrison_Grau_STCH','Morrison_Hail_STCH',]
    ensnames = [ens,]
    labels = list(string.ascii_lowercase)[:13]
elif enstype == 'MXMP':
    experiments = ['ICBC','WSM6_Grau','WSM6_Hail','Kessler','Ferrier',
                    'WSM5','WDM5','Lin','WDM6_Grau','WDM6_Hail',
                    'Morrison_Grau','Morrison_Hail']
    ensnames = [ens,]
    labels = list(string.ascii_lowercase)[:13]
elif enstype == 'ICBC':
    ensnames =  ['c00'] + ['p'+"%02d" %n for n in range(1,11)]
    experiments = ['ICBC',]
    labels = list(string.ascii_lowercase)[:12]
else:
    raise Exception

if case[:4] == '2006':
    inittime = (2006,5,26,0,0,0)
    itime = (2006,5,27,5,0,0)
    ftime = (2006,5,27,6,0,0)
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
    itime = (2011,4,20,2,0,0)
    ftime = (2011,4,20,3,0,0)
    iwind = (2011,4,19,20,0,0)
    fwind = (2011,4,20,10,0,0)
    matchnc = '/chinook2/jrlawson/bowecho/20110419/GEFSR2/c00/ICBC/wrfout_d01_2011-04-19_00:00:00'
elif case[:4] == '2013':
    inittime = (2013,8,15,0,0,0)
    itime = (2013,8,15,18,0,0)
    ftime = (2013,8,16,6,0,0)
    iwind = (2013,8,15,18,0,0)
    fwind = (2013,8,16,12,0,0)
    # times = [(2013,8,16,3,0,0),]
    compt = [(2013,8,d,h,0,0) for d,h in zip((15,16,16),(22,2,6))]
    matchnc = '/chinook2/jrlawson/bowecho/20130815/GEFSR2_paper{0}/p09/ICBC/wrfout_d0{1}_2013-08-15_00:00:00'.format(paper,dom)
else:
    raise Exception

hourly = 3
level = 2000

def get_folders(en,ex,ic=IC):
    if ic=='GEFSR2' and case[:4]=='2013' and not case.endswith('hires'):
        ic_data = 'GEFSR2_paper{0}'.format(paper)
    else:
        ic_data = ic
    print(ic)

    if case.endswith('hires'):
        out_sd = os.path.join(outroot,'hires',case[:8],'d0{0}'.format(dom),ex)
        wrf_sd = os.path.join(ncroot,case,ex)
        sp_sd = os.path.join(outroot,'hires',case[:8],'d0{0}'.format(dom))
    elif enstype[:4] == 'STCH':
        if ic=='RUC':
            mp = ''
        else:
            mp = MP


        out_sd = os.path.join(outroot,case,ic_data,en,mp,ex)
        wrf_sd = os.path.join(ncroot,case,ic_data,en,mp,ex)
        sp_sd = os.path.join(outroot,case,ic_data,en,mp)
    elif enstype == 'ICBC':
        out_sd = os.path.join(outroot,case,ic_data,en,ex)
        wrf_sd = os.path.join(ncroot,case,ic_data,en,ex)
        sp_sd = os.path.join(outroot,case,ic_data)
    else:
        out_sd = os.path.join(outroot,case,ic_data,en,ex)
        wrf_sd = os.path.join(ncroot,case,ic_data,en,ex)
        sp_sd = os.path.join(outroot,case,ic_data,en)
    return out_sd, wrf_sd, sp_sd

def get_verif_dirs():
    outdir = os.path.join(outroot,case,'VERIF')
    datadir = os.path.join(ncroot,case[:8],'VERIF')
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

# for lvl in (1,2,4,7,10,15,20,25,30):

for t in times:
    fig,axes = plt.subplots(nrow,ncol,figsize=(9,7))
    pn = 0
    for ax,plot,en,ex in zip(axes.flat,plotlist,enslist,exlist):
        # cb = False
        if plot=='NEXRAD':
            # Verification
            outdir, ncdir, sp_outdir = get_folders(enslist[3],exlist[3])
            # import pdb; pdb.set_trace()
            print(("Looking for data in {0}".format(ncdir)))
            p.plot_radar(t,datadir,outdir=False,ncdir=ncdir,fig=fig,ax=ax,cb=False,dom=dom,nct=inittime,**lims)
            make_subplot_label(ax,labels[pn])
            pn += 1
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
                    # cb = axes.flat[-1]
                    pass
                    # cb, kw = M.colorbar.make_axes(cb,shrink=0.2)
                else:
                    # cb = False
                    pass
                nct = inittime
                mnc = False
                sm = 7
                outdir, ncdir, sp_outdir = get_folders(en,ex)
                if plot=='ICBC' and enstype[:4] =='STCH':
                    ncdir = os.path.dirname(ncdir)
            print(("Looking for data in {0}".format(ncdir)))

            tstr = t
            # pdb.set_trace()
            ##### SET NAME #####
            other = False
            pt = 'contourf'
            # vrbl = 'PMSL';lv = False;clvs=N.arange(900,1100,4)*10**2;pt='contour';sm=sm*5
            # vrbl = 'strongestwind';lv=False;clvs = N.arange(10,31,1); tstr=False;extend='max';cmap='jet'
            # vrbl = 'RH';lv = 700;clvs=N.arange(0,105,5)
            # vrbl = 'Q2';lv = False;clvs=N.arange(1,20.5,0.5)*10**-3
            # vrbl = 'Z';lv = 300;clvs=N.arange(8400,9600,30); pt='contour';sm=sm*3
            # vrbl = 'Z';lv = 700;clvs=N.arange(2800,3430,30); pt='contour';sm=sm*3
            # vrbl='Z';lv=500;clvs=N.arange(5000,6000,50);pt='contour';sm=sm*3
            # vrbl = 'T2';lv = False;clvs=N.arange(280,316,1)
            # vrbl = 'shear';lv = False;clvs=N.arange(5,36,1); other = {'top':6,'bottom':0};extend='max';cmap='YlGnBu'
            # vrbl = 'Q_pert';lv=800;clvs=N.arange(-0.005,0.0051,0.0001);cmap='BrBG';extend='both'
            # vrbl = 'frontogen';lv = 2000; clvs = N.arange(-1.5,1.6,0.1)*10**-7
            vrbl = 'cref';lv = False;clvs=False;sm=False;extend=False;cmap=False
            # vrbl = 'dptp';lv=2000;clvs=N.arange(-20,1,1);cmap='terrain';extend='min'
            # vrbl = 'wind10';lv = 2000;clvs=N.arange(10,40,5);sm=False;extend='max'; cmap='jet';
            # vrbl = 'wind';lv = lvl;clvs=N.arange(10,40,5);False;sm=False;cmap='jet';extend='max'
            ######### COMMAND HERE #########
            if plot!='RUC': cb = p.plot2D(vrbl,utc=t,level=lv,ncdir=ncdir,outdir=outdir,fig=fig,ax=ax,cb=False,clvs=clvs,nct=nct,match_nc=mnc,other=other,smooth=sm,plottype=pt,save=False,dom=dom,extend=extend,cmap=cmap,**lims)
            # cb = p.plot2D(vrbl,utc=t,level=lv,ncdir=ncdir,outdir=outdir,fig=fig,ax=ax,cb=cb,clvs=clvs,nct=nct,match_nc=mnc,other=other,smooth=sm,plottype=pt)
            # p.frontogenesis(utc=t,level=lv,ncdir=ncdir,outdir=outdir,clvs=clvs,smooth=5,fig=fig,ax=ax,cb=False,match_nc=mnc,nct=nct)
            # if plot!='RUC': cb = p.plot_strongest_wind(iwind,fwind,ncdir=ncdir,outdir=outdir,clvs=clvs,fig=fig,ax=ax,cb=False,nct=nct,dom=dom,save=False)
            ################################
            if plot is not 'RUC':
                try:
                    make_subplot_label(ax,labels[pn])
                except IndexError:
                    pass
                else:
                    pn += 1
        print(("Plotting {0} panel".format(plot)))
        if plot=='ICBC':
            axtitle = 'Control'
        elif plot=='RUC':
            axtitle = ''
        else:
            axtitle = plot.replace('_',' ')
        ax.set_title(axtitle)
    if len(plotlist)==13:
        axes.flat[-2].axis('off')
    
    axes.flat[-1].axis('off')
    if vrbl=='cref' or 'strongestwind':
        axes.flat[1].axis('off')

    # axes.flat[-1].colorbar(cb)
    # plt.colorbar(cb,cax=cax)
    # plt.colorbar(cb,cax=axes.flat[-1],use_gridspec=True)
    fig.tight_layout(h_pad=0.01)
    fig.subplots_adjust(wspace = 0.1, hspace = 0.1)

    if vrbl is not 'sstrongestwind':
        cbar_ax = fig.add_axes([0.81,0.12,0.16,0.023])
        cb1 = plt.colorbar(cb,cax=cbar_ax,orientation='horizontal')#,extend='both')
        cb1.set_label('Comp. Reflectivity (dBZ)')

    sp_fname = p.create_fname(vrbl,utc=tstr,level=lv,f_prefix=IC,f_suffix=enstype)
    sp_fpath = os.path.join(sp_outdir,sp_fname)
    utils.trycreate(sp_outdir)
    fig.savefig(sp_fpath)
    plt.close(fig)
    print(("Saving figure to {0}".format(sp_fpath)))
