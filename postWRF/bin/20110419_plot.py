import os
import pdb
import sys
sys.path.append('/home/jrlawson/gitprojects/')

from WEM.postWRF.postWRF import WRFEnviron
from settings import Settings

config = Settings()
p = WRFEnviron(config)

case = '20110419'
IC = 'GEFSR2' 
ensnames = ['c00'] + ['p'+"%02d" %n for n in range(1,11)]
experiment = {'ICBC':'CTRL'}

itime = (2011,4,19,16,0,0)
ftime = (2011,4,20,11,30,0)
times = p.generate_times(itime,ftime,60*60)

variables = {'cref':{},'wind10':{},'CAPE':{}}
variables['cref'] = {'lv':2000,'pt':times}

for en in ensnames:
    addpath = os.path.join(p.output_root,case,IC,en,experiment.keys()[0])
    p.output_root += addpath
    p.wrfout_root += addpath
    p.plot_2D(variables,times)
