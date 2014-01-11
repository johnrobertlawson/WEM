"""This script shows examples of using the package to create arrays
of data stored to disc. This is then plotted using the package.
"""
import sys
import os
import pdb
import calendar

sys.path.append('../') 

from DKE_settings import Settings
from postWRF import WRFEnviron

# Initialise settings and environment
config = Settings()
p = WRFEnviron(config)

# User settings
init_time = p.string_from_time('dir',(2009,9,10,0,0,0),strlen='hour')
rootdir = '/uufs/chpc.utah.edu/common/home/horel-group2/lawson2/'
outdir = '/uufs/chpc.utah.edu/common/home/u0737349/public_html/paper2/'

#for rundate in ('25','27','29'):
for rundate in ['29']:
    print("Computing for {0} November".format(rundate))
    foldername = '201111' + rundate + '00'
    runfolder = os.path.join(rootdir,foldername)
    path_to_wrfouts = p.wrfout_files_in(runfolder,dom=1)

    itime = (2011,11,int(rundate),0,0,0)
    ftime = (2011,12,2,12,0,0)
    #times = p.generate_times(itime,ftime,12*3600)
    times = (calendar.timegm((2011,12,1,12,0,0)))
    path_to_plots = os.path.join(outdir,runfolder)
    #pdb.set_trace()
    # Produce .npy data files with DKE data
    print("Compute_diff_energy...")
    p.compute_diff_energy('sum_z','kinetic',path_to_wrfouts,times,upper=500,
                          d_save=runfolder, d_return=0,d_fname='DKE_'+foldername)
    # Plot these .npy files
    # p.plot_diff_energy('sum_z',times,runfolder,path_to_plots)
