import sys
import datetime
import matplotlib as M
M.use("agg")
import matplotlib.pyplot as plt
import os

sys.path.append('/home/jrlawson/gitprojects/')
from WEM.postWRF.postWRF.obs import StormReports
from WEM.postWRF.postWRF import WRFEnviron

# SETTINGS
fpath = '/home/jrlawson/ILIN11/storm_data_search_results.csv'
datadir = '/chinook2/jrlawson/bowecho/20130815/VERIF'
# Times in local time...
itime = datetime.datetime(2011,4,19,12,0,0)
ftime = datetime.datetime(2011,4,20,6,0,0)
# Times in UTC...
# radartimes = [datetime.datetime(2011,4,19,22,0,0),
            # datetime.datetime(2011,4,20,2,30,0),
            # datetime.datetime(2011,4,20,5,0,0)]
radarthresh = 25
# radarthresh = 35
radartimes = datetime.datetime(2011,4,20,3,0,0)
lims = {'Nlim':45.0,'Elim':-80.0,'Slim':32.0,'Wlim':-95.0}
fname = 'ILIN11_reports_radar.png'
outdir = '/home/jrlawson/public_html/bowecho'

# FIGURE SETUP
fig, ax = plt.subplots(1,figsize=(5,6))
p = WRFEnviron()

# Composite radar
p.plot_radar(radartimes,datadir,outdir,compthresh=radarthresh,
                composite=False,fig=fig,ax=ax,**lims)

# Storm reports
SR = StormReports(fpath)
SR.plot('Wind',itime,ftime,fname,outdir,ax=ax,fig=fig,ss=8,color='magenta',**lims)


# Annotate four bow echoes



# Save fig
fig.tight_layout()
fig.savefig(os.path.join(outdir,fname))
