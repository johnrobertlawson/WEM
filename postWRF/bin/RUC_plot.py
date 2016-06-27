import os
import pdb
import sys
sys.path.append('/home/jrlawson/gitprojects/')

from WEM.postWRF.postWRF import WRFEnviron
from settings import Settings
import WEM.utils.utils as utils

config = Settings()
p = WRFEnviron(config)

case = '20060526'
#case = '20090910'
#case = '20110419'
#case = '20130815'

IC = 'GEFSR2'
#IC = 'NAM'
#ensnames =  ['c00'] + ['p'+"%02d" %n for n in range(1,11)]
ensnames = ['p'+"%02d" %n for n in range(8,11)]
#ensnames = ['p04']
#ensnames = ['anl']
experiment = {'ICBC':'CTRL'}

itime = (2006,5,26,22,0,0)
#itime = (2009,9,10,23,0,0)
#itime = (2011,4,19,18,0,0)
#itime = (2013,8,15,18,0,0)
ftime = (2006,5,27,11,0,0)
#ftime = (2009,9,11,14,0,0)
#ftime = (2011,4,20,10,30,0)
#ftime = (2013,8,16,11,30,0)

times = utils.generate_times(itime,ftime,60*60)
shear_times = utils.generate_times(itime,ftime,3*60*60)
thresh = 10

variables = {'cref':{}, 'wind10':{}, 'buoyancy':{},'shear':{}, 'strongestwind':{}}
variables['cref'] = {'lv':2000,'pt':times}
variables['wind10'] = {'lv':2000,'pt':times}
variables['buoyancy'] = {'lv':2000,'pt':times}
variables['shear'] = {'top':3,'bottom':0,'pt':shear_times}
variables['strongestwind'] = {'lv':2000, 'itime':itime, 'ftime':ftime, 'range':(thresh,27.5,1.25)}
# variables = {'shear':{'pt':shear_times, 'top':3, 'bottom':0}}
# variables = {'thetae':{'lv':2000,'pt':times}, 'CAPE':{'pt':times}}
# variables = {'cref':{'lv':2000,'pt':times}, 'shear':{'pt':shear_times, 'top':3, 'bottom':0}}
# variables = {'wind10':{'lv':2000,'pt':times}}

#shear06 = {'shear':{'top':6,'bottom':0,'pt':shear_times}}


for en in ensnames:
    # pdb.set_trace()
    # Reload settings
    p.C = Settings()
    # Change paths to new location
    p.C.output_root = os.path.join(config.output_root,case,IC,en,list(experiment.keys())[0])
    p.C.wrfout_root = os.path.join(config.wrfout_root,case,IC,en,list(experiment.keys())[0])
    p.plot_2D(variables)


