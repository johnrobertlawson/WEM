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

IC = 'GEFSR2'
experiment = 'STCH'
ens = 'p09'
MP = 'Morrison_Hail'
# Time script
scriptstart = time.time()
stoch_names = ["s{0:02d}".format(n) for n in range(1,11)]
MPs = ('ICBC','WSM6_Grau','WSM6_Hail','Kessler','Ferrier','WSM5',
		'WDM5','Lin','WDM6_Grau','WDM6_Hail',
		'Morrison_Grau','Morrison_Hail')

# Initialise settings and environment
config = Settings()
p = WRFEnviron(config)


#itime = (2006,5,26,0,0,0)
#ftime = (2006,5,27,12,0,0)

#itime = (2009,9,10,0,0,0)
#ftime = (2009,9,11,15,0,0)

#itime = (2011,4,19,0,0,0)
#ftime = (2011,4,20,15,0,0)

itime = (2013,8,15,0,0,0)
ftime = (2013,8,16,12,0,0)

times = utils.generate_times(itime,ftime,3*3600)
#members = ['s0{0}'.format(n) for n in range(1,8)]

path_to_wrfouts = []
for s in stoch_names:
    fpath = os.path.join(config.wrfout_root,case,IC,ens,MP,s)
    path_to_wrfouts.append(p.wrfout_files_in(fpath,dom=1)[0])

picklefolder = os.path.join(config.wrfout_root,case,IC,ens,MP)
p.C.output_root = os.path.join(config.output_root,case,IC,ens,MP)
pfname = 'DTE_' + experiment

# pdb.set_trace()
p.compute_diff_energy('sum_z','total',path_to_wrfouts,times,
                          d_save=picklefolder, d_return=0,d_fname=pfname)

# Contour fixed at these values
V = range(250,5250,250)
VV = [100,] + V
p.plot_diff_energy('sum_z','total',times,picklefolder,pfname,VV)

#p.plot_growth_sensitivity('DTE',picklefolder,pfname,MPs)

print "Script took", time.time()-scriptstart, "seconds."

