"""
Load/download RUC data.
Loop through dates in case study climo.
Plot certain fields with WEM.postWRF
"""


import os
import pdb
import sys
import matplotlib as M
M.use('agg')
import matplotlib.pyplot as plt
import numpy as N
import datetime

# import WEM.lazyWRF.lazyWRF as lazyWRF
from WEM.postWRF.postWRF import WRFEnviron
import WEM.utils as utils
#from WEM.postWRF.postWRF.rucplot import RUCPlot

RUCdir = '/chinook2/jrlawson/bowecho/RUCclimo/'

p = WRFEnviron()

# cases = {}
# cases['20091109'] = {'utc':(2009,11,10,0,0,0), 'datadir':os.path.join(RUCdir,'20060526/RUC/anl/VERIF/') }
# cases['2013'] = {'utc':(2013,8,16,0,0,0), 'datadir':os.path.join(RUCdir,'20130815/RUC/anl/VERIF/') }

download_data = 1
skipcase = 0

def download_RUC(utc,fpath):
    print(('Downloading {0} RUC file.'.format(utc)))
    utils.getruc(utc,ncpath=fpath,convert2nc=True,duplicate=False)

plot_Z = 0
plot_T_adv = 0
plot_omega = 0
plot_lyapunov = 0
plot_regime = 1

# TODO: edit WEM so a switch will automatically download the RUC data.

fpath = '/home/jrlawson/pythoncode/bowecho/snively.csv'
# names = ('casedate','caseno','casetime','casestate','bowdate','bowtime','bowstate','score','type','comments')
# formats = ['S16',]*10
names = ('casedate','caseno','casetime','casestate','bowdate',
        'bowlat','bowlon','bowtime','bowstate','score',
        'initlat','initlon','inittime','type','comments')
formats = ['S16',]*len(names)
cases = N.loadtxt(fpath,dtype={'names':names,'formats':formats},skiprows=1,delimiter=',')

plothrs = (0,12)
# plothrs = (0,)
# plothrs = (0,9,12,15,18)
Nlim, Elim, Slim, Wlim = [52.0,-78.0,25.0,-128.0]

for n, case in enumerate(cases):
    outroot ='/home/jrlawson/public_html/bowecho/climoplots/'
    outdir = os.path.join(outroot,case['casedate'])
    

    if n<skipcase:
        continue

    if case['bowdate'] == '99999999':
        continue
    utils.trycreate(outdir)

    t = case['casedate'] + case['casetime']
    # t = case['bowdate'] + case['bowtime']
    year = int(t[:4])
    # if year==2006 or year==2008:
        # continue
    mth = int(t[4:6])
    day = int(t[6:8])
    hr = int(t[8:10])
    timetuple = (year,mth,day,12,0,0)
    # Now progress through every six hours and plot variables
    # Use datetime
    dt = datetime.datetime(*timetuple)
    times = [dt + datetime.timedelta(hours=n) for n in plothrs]

    # import pdb; pdb.set_trace()

    for un,utc in enumerate(times):
        outstr = case['casedate'] + '_{0:02d}Z'.format(utc.hour)
        if download_data:
            download_RUC(utc,RUCdir)
             
        print(utc)
        # import pdb; pdb.set_trace()
        if plot_Z:
            fig, ax = plt.subplots()

            levels = {}
            # levels[300] = {'color':'blue','clvs':N.arange(8400,9600,120)}
            # levels[500] = {'color':'black','clvs':N.arange(4800,6000,60)}
            # levels[850] = {'color':'red','clvs':N.arange(900,2100,30)}
            levels[500] = {'color':'black','clvs':N.arange(5460,6000,60),'z':10}
            levels[925] = {'color':'#9966FF','clvs':N.arange(660,2100,30),'z':1}

            for lv in levels:
                try:
                    im = p.plot2D('Z',utc=utc,level=lv,ncdir=RUCdir,outdir=outdir,
                            fig=fig,ax=ax,cb=False,clvs=levels[lv]['clvs'],nct=utc,match_nc=False,
                            smooth=10,plottype='contour',color=levels[lv]['color'],inline=True,)
                            # Nlim=Nlim,Elim=Elim,Slim=Slim,Wlim=Wlim)
                except:
                    continue
            fpath = os.path.join(outdir,outstr+'_overview.png')
            fig.tight_layout()
            fig.savefig(fpath)
            plt.close(fig)

        if plot_T_adv:
            levels = {}
            levels[700] = {'clvs':N.arange(-5,5.51,0.5)*0.0001}
            levels[850] = {'clvs':N.arange(-5,5.5,0.5)*0.0001}

            for lv in levels:
                fig, ax = plt.subplots()
                try:
                    im = p.plot2D('temp_advection',utc=utc,level=lv,ncdir=RUCdir,outdir=outdir,
                            fig=fig,ax=ax,cb=True, clvs=levels[lv]['clvs'], nct=utc,)
                except:
                    continue
                fpath = os.path.join(outdir,outstr+'_tempadvection_{0}hPa.png'.format(lv))
                fig.tight_layout()
                fig.savefig(fpath)
                plt.close(fig)

        if plot_omega:
            levels = {}
            levels[500] = {'clvs':N.arange(-1,1,0.1)*0.001}

            for lv in levels:
                fig, ax = plt.subplots()
                try:
                    im = p.plot2D('omega',utc=utc,level=lv,ncdir=RUCdir,outdir=outdir,
                            fig=fig,ax=ax,cb=True, 
                            #clvs=levels[lv]['clvs'],
                            nct=utc,)
                except:
                    continue
                fpath = os.path.join(outdir,outstr+'_omega_{0}hPa.png'.format(lv))
                fig.tight_layout()
                fig.savefig(fpath)
                plt.close(fig)

        if plot_lyapunov:
            levels = {}
            levels[500] = {'clvs':N.arange(-5.0,5.25,0.25)*10**-4}

            for lv in levels:
                fig, ax = plt.subplots()
                try:
                    im = p.plot2D('lyapunov',utc=utc,level=lv,ncdir=RUCdir,outdir=outdir,
                            fig=fig,ax=ax,cb=True,
                            # #clvs=levels[lv]['clvs'], 
                            nct=utc,
                            #cmap='bwr'
                            )
                except:
                    # If file doesn't exist on server
                    print("MISSING")
                    continue


                fpath = os.path.join(outdir,outstr+'_lyapunov_{0}hPa.png'.format(lv))
                fig.tight_layout()
                fig.savefig(fpath)
                plt.close(fig)

        if plot_regime:
            utc = datetime.datetime(*timetuple)
            Nlim, Elim, Slim, Wlim = [52.0,-78.0,25.0,-128.0]
            lims = {'Nlim':Nlim, 'Elim':Elim, 'Slim':Slim, 'Wlim':Wlim}
            fig, ax = plt.subplots()
            try:
                LXclvs = N.arange(-3.0,3.125,0.125)*10**-4
                LXcf = p.plot2D('lyapunov',utc=utc,level=500,ncdir=RUCdir,outdir=outdir,
                        fig=fig,ax=ax,cb=True,clvs=LXclvs,smooth=1,
                        nct=utc,cmap='bwr',extend='both',**lims)
                # LXct = p.plot2D('lyapunov',utc=utc,level=500,ncdir=RUCdir,outdir=outdir,
                        # fig=fig,ax=ax,cb=False,clvs=[0.5*10**-4,],color='black',lw=0.5,
                        # nct=utc,plottype='contour',**lims)
                # Zclvs = N.arange(6000,12000,120)
                Zclvs = N.arange(3000,9000,60)
                Zct = p.plot2D('Z',utc=utc,level=500,ncdir=RUCdir,outdir=outdir,
                        fig=fig,ax=ax,cb=False,clvs=Zclvs,nct=utc,match_nc=False,
                        smooth=10,plottype='contour',color='darkblue',inline=True,**lims)

            except ValueError:
                # If file doesn't exist on server
                print("MISSING")
                continue

            fig.suptitle('{0} bow echo of score {1:.2f}'.format
                            (case['type'].upper(),float(case['score'])))
            fpath = os.path.join(outdir,'Day1_12Z_regime.png')
            fig.tight_layout()
            fig.savefig(fpath)
            plt.close(fig)
            print(("Saved figure to {0}".format(fpath)))

if plot_regime:
    os.system('python /home/jrlawson/public_html/bowecho/climoplots/day1_regimes/copy_regime.py')
