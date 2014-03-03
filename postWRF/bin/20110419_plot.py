import os
import pdb
import sys
sys.path.append('/home/jrlawson/gitprojects/')

from WEM.postWRF.postWRF import WRFEnviron
from settings import Settings

config = Settings()
p = WRFEnviron(config)

case = '20130815'
IC = 'GEFSR2' 
#ensnames = ['c00'] + ['p'+"%02d" %n for n in range(1,11)]
ensnames = ['p'+"%02d" %n for n in range(8,9)]
experiment = {'ICBC':'CTRL'}

#itime = (2011,4,19,18,0,0)
itime = (2013,8,15,18,0,0)
#ftime = (2011,4,20,10,30,0)
ftime = (2013,8,16,12,0,0)
times = p.generate_times(itime,ftime,60*60)

# variables = {'cref':{}, 'wind10':{}}
#variables = {'cref':{},'wind10':{},'CAPE':{}}
# variables['cref'] = {'lv':2000,'pt':times}
# variables['wind10'] = {'lv':2000,'pt':times}
#variables['CAPE'] = {'pt':times}
variables = {'thetae':{'lv':2000,'pt':times}, 'CAPE':{'pt':times}}

for en in ensnames:
    # Reload settings
    p.C = Settings()
    # Change paths to new location
    p.C.output_root = os.path.join(config.output_root,case,IC,en,experiment.keys()[0])
    p.C.wrfout_root = os.path.join(config.wrfout_root,case,IC,en,experiment.keys()[0])
    p.plot_2D(variables)

"""
# Postage stamp plots
list_of_wrfouts = [] 
for en in ensnames:
    p.C = Settings()
    list_of_wrfouts.append(os.path.join(config.output_root,case,IC,en,experiment.keys()[0]))
    outdirectory = os.path.join(config.output_root,case,IC)
    itime = (2011,4,19,21,0,0)
    ftime = (2011,4,20,9,30,0)
    times = p.generate_times(itime,ftime,3*60*60)
    # CODE THIS UP v v v v v v v v v
    #p.postage_stamps(variables,times,list_of_wrfouts,outdirectory)
"""
