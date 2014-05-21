import os
import pdb
import sys
sys.path.append('/home/jrlawson/gitprojects/')

from WEM.postWRF.postWRF import WRFEnviron
from settings import Settings
import WEM.utils.utils as utils
from WEM.postWRF.postWRF.rucplot import RUCPlot

config = Settings()
p = WRFEnviron(config)

case = '20060526'
#case = '20090910'
#case = '20110419'
#case = '20130815'

#IC = 'GEFSR2'
IC = 'NAM'
#IC = 'RUC'
#ensnames =  ['c00'] + ['p'+"%02d" %n for n in range(1,11)]
#ensnames = ['p'+"%02d" %n for n in range(8,11)]
ensnames = ['anl',]
MP = ''
#ensnames = ['anl']
#experiments = ['WDM6_Grau','WDM6_Hail','Morrison_Grau','Morrison_Hail']
#experiments = ['WSM6_Grau','WSM6_Hail','Kessler','Ferrier','WSM5','WDM5','Lin','WDM6_Grau','WDM6_Hail','Morrison_Grau','Morrison_Hail']
experiments = ['Morrison_Hail',]
#experiment = 'VERIF'
#experiments = ['s'+"%02d" %n for n in range(1,11)]

itime = (2006,5,27,0,0,0)
#itime = (2009,9,10,23,0,0)
#itime = (2011,4,19,18,0,0)
#itime = (2013,8,15,18,0,0)

ftime = (2006,5,27,13,0,0)
#ftime = (2009,9,11,14,0,0)
#ftime = (2011,4,20,10,30,0)
#ftime = (2013,8,16,12,0,0)

times = utils.generate_times(itime,ftime,60*60)
#shear_times = utils.generate_times(itime,ftime,3*60*60)
#sl_times = utils.generate_times(sl_itime,sl_ftime,1*60*60)
thresh = 10
skewT_time = (2013,8,16,3,0,0)
skewT_latlon = (35.2435,-97.4708)

#variables = {'cref':{}}
variables = {'strongestwind':{}}
#variables['cref'] = {'lv':2000,'pt':times}
variables['strongestwind'] = {'lv':2000, 'itime':itime, 'ftime':ftime, 'range':(thresh,27.5,1.25)}
#variables['PMSL'] = {'lv':2000,'pt':times,'plottype':'contour','smooth':5}
#variables['wind10'] = {'lv':2000,'pt':times}
#variables['buoyancy'] = {'lv':2000,'pt':times}
#variables['shear'] = {'top':3,'bottom':0,'pt':shear_times}
# variables = {'shear':{'pt':shear_times, 'top':3, 'bottom':0}}
# variables = {'thetae':{'lv':2000,'pt':times}, 'CAPE':{'pt':times}}
# variables = {'cref':{'lv':2000,'pt':times}, 'shear':{'pt':shear_times, 'top':3, 'bottom':0}}
#variables = {'PMSL':{'lv':2000,'pt':times,'plottype':'contour','smooth':5}}

#shear06 = {'shear':{'top':6,'bottom':0,'pt':shear_times}}

"""
for en in ensnames:
    for ex in experiments:
        # Reload settings
        p.C = Settings()

        # Change paths to new location
        p.C.output_root = os.path.join(config.output_root,case,IC,en,MP,ex)
        p.C.wrfout_root = os.path.join(config.wrfout_root,case,IC,en,MP,ex)
        p.C.pickledir = os.path.join(config.wrfout_root,case,IC,en,MP,ex)

        p.plot_skewT(skewT_time,skewT_latlon,save_output=1)

#p.plot_skewT(skewT_time,skewT_latlon,composite=1)
"""

for en in ensnames:
    for ex in experiments:
        # Reload settings
        p.C = Settings()

        # Change paths to new location
        p.C.output_root = os.path.join(config.output_root,case,IC,en,MP,ex)
        p.C.wrfout_root = os.path.join(config.wrfout_root,case,IC,en,MP,ex)
        print p.C.output_root
        print p.C.wrfout_root
        p.plot_2D(variables)

"""

for en in ensnames:
    # Reload settings
    p.C = Settings()
    # Change paths to new location
    p.C.output_root = os.path.join(config.output_root,case,IC,en,experiment)
    p.C.wrfout_root = os.path.join(config.wrfout_root,case,IC,en,experiment)
    #p.plot_2D(variables)
    print p.C.wrfout_root
    p.plot_streamlines(2000,sl_times)


# RUC file is one-per-time so .nc file is specified beforehand
en = ensnames[0]
RC = Settings()
RC.output_root = os.path.join(config.output_root,case,IC,en,experiment)
RC.path_to_RUC = os.path.join(config.RUC_root,case,IC,en,experiment)
WRF_dir = os.path.join(config.wrfout_root,case,'NAM',en,'ICBC')

variables = ['streamlines',]
level = 2000

for t in sl_times:
    RUC = RUCPlot(RC,t,wrfdir=WRF_dir)
    #limits = RUC.colocate_WRF_map(WRF_dir)
    RUC.plot(variables,level)
"""

