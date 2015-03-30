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

sys.path.append('/home/jrlawson/gitprojects/')

# import WEM.lazyWRF.lazyWRF as lazyWRF
from WEM.postWRF.postWRF import WRFEnviron
import WEM.utils as utils
#from WEM.postWRF.postWRF.rucplot import RUCPlot

outdir = '/home/jrlawson/public_html/bowecho/climoplots/'
RUCdir = '/chinook2/jrlawson/bowecho/RUCclimo/'

p = WRFEnviron()

# cases = {}
# cases['20091109'] = {'utc':(2009,11,10,0,0,0), 'datadir':os.path.join(RUCdir,'20060526/RUC/anl/VERIF/') }
# cases['2013'] = {'utc':(2013,8,16,0,0,0), 'datadir':os.path.join(RUCdir,'20130815/RUC/anl/VERIF/') }

download_data = 0
skipcase = 0

def download_RUC(utc,fpath):
    print('Downloading {0} RUC file.'.format(utc))
    utils.getruc(utc,ncpath=fpath,convert2nc=True,duplicate=False)

plot_Z =0
plot_T_adv = 0
plot_omega = 1
plot_lyapunov = 0

# TODO: edit WEM so a switch will automatically download the RUC data.

fpath = '/home/jrlawson/pythoncode/bowecho/snively.csv'
names = ('casedate','caseno','casetime','casestate','bowdate','bowtime','bowstate','score','type','comments')
formats = ['S16',]*10
cases = N.loadtxt(fpath,dtype={'names':names,'formats':formats},skiprows=1,delimiter=',')


for n, case in enumerate(cases):
    if n<skipcase:
        continue

    if case['bowdate'] == '99999999':
        continue


    t = case['bowdate'] + case['bowtime']
    year = int(t[:4])
    # if year==2007 or year==2008:
        # continue
    mth = int(t[4:6])
    day = int(t[6:8])
    hr = int(t[8:10])
    utc = (year,mth,day,hr,0,0)

    

    if download_data:
        download_RUC(utc,RUCdir)
         
    # import pdb; pdb.set_trace()
    if plot_Z:
        fig, ax = plt.subplots()

        levels = {}
        # levels[300] = {'color':'blue','clvs':N.arange(8400,9600,120)}
        levels[500] = {'color':'black','clvs':N.arange(4800,6000,60)}
        levels[850] = {'color':'red','clvs':N.arange(900,2100,30)}

        for lv in levels:
            im = p.plot2D('Z',utc=utc,level=lv,ncdir=RUCdir,outdir=outdir,
                        fig=fig,ax=ax,cb=False,clvs=levels[lv]['clvs'],nct=utc,match_nc=False,
                        smooth=10,plottype='contour',color=levels[lv]['color'])
        fpath = os.path.join(outdir,case['casedate']+'_overview.png')
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
            fpath = os.path.join(outdir,case['casedate']+'_tempadvection_{0}hPa.png'.format(lv))
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
            fpath = os.path.join(outdir,case['casedate']+'_omega_{0}hPa.png'.format(lv))
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
            fpath = os.path.join(outdir,case['casedate']+'_lyapunov_{0}hPa.png'.format(lv))
            fig.savefig(fpath)
            plt.close(fig)

