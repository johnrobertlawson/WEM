import sys
import numpy as N
#import matplotlib as M
#M.use('gtkagg')
sys.path.append('/home/jrlawson/gitprojects/WEM')

from WEM.postWRF.postWRF import WRFEnviron

p = WRFEnviron()
outdir = '/home/jrlawson/public_html/bowecho'
datadir = '/chinook2/jrlawson/bowecho/20130815/VERIF'

reportday = (2013,8,15,23,0,0)
# utc = [(2013,8,16,3,m,0) for m in range(0,60,5)]
utc = (2013,8,16,3,0,0)
ncdir = '/chinook2/jrlawson/bowecho/20130815/GEFSR2/c00/ICBC'

# p.plot2D(t,utc=utc,level=2000)
p.plot_accum_rain(utc,6,ncdir,outdir,clvs=N.arange(5,85,1))

for t in utc:
    # R = Radar(t,datadir)
    # R.plot_radar(outdir,Nlim=42.0,Elim=-92.0,Slim=32.0,Wlim=-102.0)
    p.plot_radar(t,datadir,outdir,ncdir=ncdir,nct=nct)

# SPC = SPCReports(reportday,datadir,torn=False,hail=False)
