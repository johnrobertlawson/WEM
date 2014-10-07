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
from WEM.postWRF import WRFEnviron
import WEM.utils as utils

compute = 0
plot_2D = 1
plot_1D = 1

case = '20060526'
#case = '20090910'
#case = '20110419'
# case = '20130815'

IC = 'NAM'
experiment = 'MXMP'
ens = 'anl'
MP = 'ICBC'
# Time script
scriptstart = time.time()
stoch_names = ["s{0:02d}".format(n) for n in range(1,11)]
ens_names = ['c00',] + ["p{0:02d}".format(n) for n in range(1,11)]

MPs = ['ICBC','WSM6_Grau','WSM6_Hail','Kessler','Ferrier','WSM5',
		'WDM5','Lin','WDM6_Grau','WDM6_Hail',
		'Morrison_Grau','Morrison_Hail']

# Initialise settings and environment
config = Settings()
p = WRFEnviron(config)

if case[:4] == '2006':
    itime = (2006,5,26,0,0,0)
    ftime = (2006,5,27,12,0,0)
elif case[:4] == '2009':
    itime = (2009,9,10,0,0,0)
    ftime = (2009,9,11,18,0,0)
elif case[:4] == '2011':
    itime = (2011,4,19,0,0,0)
    ftime = (2011,4,20,15,0,0)
else:
    itime = (2013,8,15,0,0,0)
    ftime = (2013,8,16,15,0,0)

times = utils.generate_times(itime,ftime,2*3600)

if experiment=='STCH':
    picklefolder = os.path.join(config.wrfout_root,case,IC,ens,MP)
    p.C.output_root = os.path.join(config.output_root,case,IC,ens,MP)
 
    path_to_wrfouts = []
    for s in stoch_names:
        fpath = os.path.join(config.wrfout_root,case,IC,ens,MP,s)
        path_to_wrfouts.append(p.wrfout_files_in(fpath,dom=1)[0])
    sensitivity=0

elif experiment=='ICBC':
    picklefolder = os.path.join(config.wrfout_root,case,IC)
    p.C.output_root = os.path.join(config.output_root,case,IC)
    sensitivity=ens_names
    #fpath = os.path.join(config.wrfout_root,case,IC,)
    #path_to_wrfouts = p.wrfout_files_in(fpath,dom=1)

elif experiment=='MXMP':
    picklefolder = os.path.join(config.wrfout_root,case,IC,ens)
    outpath = os.path.join(config.output_root,case,IC,ens)
    sensitivity = MPs
    path_to_wrfouts = []
    for mp in MPs:
        fpath = os.path.join(config.wrfout_root,case,IC,ens,MP)
        path_to_wrfouts.append(utils.wrfout_files_in(fpath,dom=1)[0])
else:
    print "Typo!"
    raise Exception
    
pfname = 'DTE_' + experiment

if compute:
    p.compute_diff_energy('sum_z','total',path_to_wrfouts,times,
                          d_save=picklefolder, d_return=0,d_fname=pfname)

if plot_2D:
    # Contour fixed at these values
    V = range(250,5250,250)
    VV = [100,] + V
    ofname = pfname + '_2D'
    for t in times:
        p.plot_diff_energy('sum_z','total',t,picklefolder,pfname,outpath,ofname,VV)

if plot_1D:
    ylimits = [0,2e8]
    ofname = pfname
    p.plot_error_growth(ofname,picklefolder,pfname,sensitivity=sensitivity,ylimits=ylimits)

#print "Script took", time.time()-scriptstart, "seconds."
