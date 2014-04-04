"""This script shows examples of using the package to create arrays
of data stored to disc. This is then plotted using the package.
"""
import sys
import os
import pdb
import calendar
import time

sys.path.append('/home/jrlawson/gitprojects/') 

from DKE_settings import Settings
from WEM.postWRF.postWRF import WRFEnviron
import WEM.utils.utils as utils

case = '20130815'
IC = 'GEFSR2'

# Time script
scriptstart = time.time()

# Initialise settings and environment
config = Settings()
p = WRFEnviron(config)

# User settings
init_time = p.string_from_time('dir',(2013,8,15,0,0,0),strlen='hour')

runfolder = os.path.join(config.wrfout_root,case,IC)
path_to_wrfouts = p.wrfout_files_in(runfolder,dom=1)

itime = (2013,8,15,0,0,0)
ftime = (2013,8,16,12,0,0)
times = utils.generate_times(itime,ftime,6*3600)
path_to_plots = os.path.join(config.output_root,case,IC)
#pdb.set_trace()
# Produce .npy data files with DKE data
print("Compute_diff_energy...")
p.compute_diff_energy('sum_z','kinetic',path_to_wrfouts,times,#upper=700,
                          d_save=runfolder, d_return=0,d_fname='DTE')
# Contour fixed at these values
#V = range(0,2200,200)
#p.plot_diff_energy('sum_z','total',times,runfolder,'DTE',path_to_plots,V)

print "Script took", time.time()-scriptstart, "seconds."
