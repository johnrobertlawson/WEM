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

def download_RUC(utc,fpath):
    print(('Downloading {0} RUC file.'.format(utc)))
    utils.getruc(utc,ncpath=fpath,convert2nc=True,duplicate=False)

outdir = '/home/jrlawson/public_html/bowecho/'
RUCdir = '/chinook2/jrlawson/bowecho/'

p = WRFEnviron()

cases = {}
# cases['20060526'] = {'utc':(2006,5,26,12,0,0), 'datadir':os.path.join(RUCdir,'20060526/RUC/anl/VERIF/')}
# cases['20130815'] = {'utc':(2013,8,15,12,0,0), 'datadir':os.path.join(RUCdir,'20130815/RUC/anl/VERIF/') }
cases['20110419'] = {'utc':(2011,4,19,21,0,0), 'datadir':os.path.join(RUCdir,'20110419/RUC/anl/VERIF/') }

Nlim, Elim, Slim, Wlim = [52.0,-78.0,25.0,-128.0]

Zplot = 1
Tplot = 0
download_data = 1

for case in cases:
    if download_data:
        download_RUC(cases[case]['utc'],cases[case]['datadir'])
    if Zplot:
        fig, ax = plt.subplots()
        cb = False
        clvs = False
        mnc = False
        sm = 10
        other = False
        pt = 'contour'

        levels = {}
        # levels[300] = {'color':'blue','clvs':N.arange(8400,9600,120)}
        levels[500] = {'color':'black','clvs':N.arange(5460,6000,60),'z':10}
        # levels[850] = {'color':'green','clvs':N.arange(900,2100,30)}
        levels[925] = {'color':'#9966FF','clvs':N.arange(660,2100,30),'z':1}

        for lv in sorted(levels,reverse=True):
            im = p.plot2D('Z',utc=cases[case]['utc'],level=lv,ncdir=cases[case]['datadir'],outdir=outdir,
                        fig=fig,ax=ax,cb=cb,clvs=levels[lv]['clvs'],nct=cases[case]['utc'],match_nc=mnc,
                        other=other,smooth=sm,plottype=pt,color=levels[lv]['color'],inline=True,
                        Nlim=Nlim,Elim=Elim,Slim=Slim,Wlim=Wlim)

        datestr = '_'.join(['{0:02d}'.format(n) for n in cases[case]['utc'][2:4]])
        fpath = os.path.join(outdir,case,datestr+'_forecastfunnel.png')
        fig.tight_layout()
        fig.savefig(fpath)
        plt.close(fig)

    if Tplot:
        levels = {}
        levels[700] = {'clvs':N.arange(-10,11,1)*0.0001}
        levels[850] = {'clvs':N.arange(-10,11,1)*0.0001}

        if case=='2013':
            levels[850]['clvs'] = N.arange(-5,5.5,0.5)*0.0001
        for lv in levels:
            fig, ax = plt.subplots()
            cb = False
            clvs = False
            mnc = False
            sm = 10
            other = False
            pt = 'contour'


            im = p.plot2D('temp_advection',utc=cases[case]['utc'],level=lv,ncdir=cases[case]['datadir'],outdir=outdir,
                        fig=fig,ax=ax,cb=False,
                        # clvs=False,
                        clvs=levels[lv]['clvs'],
                        nct=cases[case]['utc'],match_nc=mnc,
                        )

            fpath = os.path.join(outdir,case+'_tempadvection_{0}hPa.png'.format(lv))
            fig.savefig(fpath)
            plt.close(fig)
