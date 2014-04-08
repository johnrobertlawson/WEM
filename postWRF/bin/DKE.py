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

#case = '20060526'
#case = '20090910'
#case = '20110419'
case = '20130815'

IC = 'NAM'
experiment = 'MXMP'
ens = 'anl'
# Time script
scriptstart = time.time()

# Initialise settings and environment
config = Settings()
p = WRFEnviron(config)


runfolder = os.path.join(config.wrfout_root,case,IC)
path_to_wrfouts = p.wrfout_files_in(runfolder,dom=1)

#itime = (2006,5,26,0,0,0)
#ftime = (2006,5,27,12,0,0)

#itime = (2009,9,10,0,0,0)
#ftime = (2009,9,11,15,0,0)

#itime = (2011,4,19,0,0,0)
#ftime = (2011,4,20,15,0,0)

itime = (2013,8,15,0,0,0)
ftime = (2013,8,16,12,0,0)

times = utils.generate_times(itime,ftime,3*3600)
config.output_root = os.path.join(config.output_root,case,IC)
#pdb.set_trace()
#p.compute_diff_energy('sum_z','kinetic',path_to_wrfouts,times,
#                          d_save=runfolder, d_return=0,d_fname='DTE_MXMP')

# Contour fixed at these values
#V = range(500,5000,250)
p.plot_diff_energy('sum_xyz','total',times,runfolder,'DTE_MXMP',config.output_root,V)

print "Script took", time.time()-scriptstart, "seconds."
