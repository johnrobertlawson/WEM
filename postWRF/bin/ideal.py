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


ncroot = '/ptmp/jrlawson/WRF_3.6.1_ideal/run/'
outroot = '/home/jrlawson/public_html/paper5'

p = WRFEnviron()

nct = (2000,1,1,0,0,0)
t = (2000,1,1,1,30,0)
# p.plot2D('REFL_comp',level=False,utc=t,ncdir=ncroot,nct=nct,outdir=outroot,cb=True,dom=1,ideal=True)
# p.plot2D('lyapunov',level=500,utc=t,ncdir=ncroot,nct=nct,outdir=outroot,cb=True,dom=1,ideal=True)
# p.plot2D('U',level=500,utc=t,ncdir=ncroot,nct=nct,outdir=outroot,cb=True,dom=1,ideal=True)
# p.plot2D('V',level=500,utc=t,ncdir=ncroot,nct=nct,outdir=outroot,cb=True,dom=1,ideal=True)
p.plot_streamlines(t,500,ncroot,outroot,nct=nct,dom=1,ideal=True)
