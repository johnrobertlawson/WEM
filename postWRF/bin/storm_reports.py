import sys
import datetime

sys.path.append('/home/jrlawson/gitprojects/')
from WEM.postWRF.postWRF.obs import StormReports

fpath = '/home/jrlawson/storm_data_search_results.csv'
itime = datetime.datetime(2013,8,15,12,0,0)
ftime = datetime.datetime(2013,8,16,12,0,0)
lims = {'Nlim':42.0,'Elim':-94.0,'Slim':31.0,'Wlim':-103.4}

fname = 'KSOK13_wind'
outdir = '/home/jrlawson/public_html/bowecho'
SR = StormReports(fpath)
SR.plot('Wind',itime,ftime,fname,outdir,**lims)
