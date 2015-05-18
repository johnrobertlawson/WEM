"""
Load/download RUC data.
Loop through dates in case study climo.
Plot certain fields with WEM.postWRF
"""
from __future__ import division

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
    print('Downloading {0} RUC file.'.format(utc))
    utils.getruc(utc,ncpath=fpath,convert2nc=True,duplicate=False)

plot_Z = 0
plot_T_adv = 0
plot_omega = 0
plot_lyapunov = 1

# TODO: edit WEM so a switch will automatically download the RUC data.

fpath = '/home/jrlawson/pythoncode/bowecho/snively.csv'
names = ('casedate','caseno','casetime','casestate','bowdate','bowtime','bowstate','score','type','comments')
formats = ['S16',]*10
cases = N.loadtxt(fpath,dtype={'names':names,'formats':formats},skiprows=1,delimiter=',')

# plothrs = (0,12,18)
plothrs = (0,9,12,15,18)
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
    # if year==2007 or year==2008:
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
                im = p.plot2D('Z',utc=utc,level=lv,ncdir=RUCdir,outdir=outdir,
                            fig=fig,ax=ax,cb=False,clvs=levels[lv]['clvs'],nct=utc,match_nc=False,
                            smooth=10,plottype='contour',color=levels[lv]['color'],inline=True,)
                            # Nlim=Nlim,Elim=Elim,Slim=Slim,Wlim=Wlim)
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

                im = p.plot2D('temp_advection',utc=utc,level=lv,ncdir=RUCdir,outdir=outdir,
                            fig=fig,ax=ax,cb=True, clvs=levels[lv]['clvs'], nct=utc,)
                fpath = os.path.join(outdir,outstr+'_tempadvection_{0}hPa.png'.format(lv))
                fig.tight_layout()
                fig.savefig(fpath)
                plt.close(fig)

        if plot_omega:
            levels = {}
            levels[500] = {'clvs':N.arange(-1,1,0.1)*0.001}

            for lv in levels:
                fig, ax = plt.subplots()
                im = p.plot2D('omega',utc=utc,level=lv,ncdir=RUCdir,outdir=outdir,
                            fig=fig,ax=ax,cb=True, 
                            #clvs=levels[lv]['clvs'],
                            nct=utc,)
                fpath = os.path.join(outdir,outstr+'_omega_{0}hPa.png'.format(lv))
                fig.tight_layout()
                fig.savefig(fpath)
                plt.close(fig)

        if plot_lyapunov:
            levels = {}
            levels[500] = {'clvs':N.arange(-7.5,8.0,0.5)*10**-3}

            for lv in levels:
                fig, ax = plt.subplots()
                im = p.plot2D('lyapunov',utc=utc,level=lv,ncdir=RUCdir,outdir=outdir,
                            fig=fig,ax=ax,cb=True,
                            #clvs=levels[lv]['clvs'], 
                            nct=utc,
                            #cmap='bwr'
                            )
                fpath = os.path.join(outdir,outstr+'_lyapunov_{0}hPa.png'.format(lv))
                fig.tight_layout()
                fig.savefig(fpath)
                plt.close(fig)

