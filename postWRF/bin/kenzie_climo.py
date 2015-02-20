"""
Load/download RUC data.
Loop through dates in case study climo.
Plot certain fields with WEM.postWRF
"""

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

outdir = '/home/jrlawson/public_html/bowecho/'
RUCdir = '/chinook2/jrlawson/bowecho/RUCclimo/'

p = WRFEnviron(download_data=True)

cases = {}
cases['20091109'] = {'utc':(2009,11,10,0,0,0), 'datadir':os.path.join(RUCdir,'20060526/RUC/anl/VERIF/') }
cases['2013'] = {'utc':(2013,8,16,0,0,0), 'datadir':os.path.join(RUCdir,'20130815/RUC/anl/VERIF/') }

cases = (20091109,
plot_Z = 0
plot_T_adv = 0
plot_omega = 0

# TODO: edit WEM so a switch will automatically download the RUC data.

for case in cases:
    if plot_Z:
        fig, ax = plt.subplots()
        cb = False
        clvs = False
        mnc = False
        sm = 10
        other = False
        pt = 'contour'

        levels = {}
        # levels[300] = {'color':'blue','clvs':N.arange(8400,9600,120)}
        levels[500] = {'color':'black','clvs':N.arange(4800,6000,60)}
        levels[850] = {'color':'red','clvs':N.arange(900,2100,30)}

        for lv in levels:
            im = p.plot2D('Z',utc=cases[case]['utc'],level=lv,ncdir=cases[case]['datadir'],outdir=outdir,
                        fig=fig,ax=ax,cb=cb,clvs=levels[lv]['clvs'],nct=cases[case]['utc'],match_nc=mnc,
                        other=other,smooth=sm,plottype=pt,color=levels[lv]['color'])
        fpath = os.path.join(outdir,case+'_overview.png')
        fig.savefig(fpath)
        plt.close(fig)

    if plot_T_adv:
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

    if plot_omega:
        levels = {}
