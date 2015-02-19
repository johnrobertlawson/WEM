import os
import pdb
import sys
import glob
sys.path.append('/home/jrlawson/gitprojects/')
import numpy as N

from WEM.postWRF.postWRF import WRFEnviron
from settings import Settings
import WEM.utils.utils as utils
from WEM.postWRF.postWRF.ecmwf import ECMWF

config = Settings()
p = WRFEnviron(config)

#case = '20060526'
#case = '20090910'
#case = '20110419'
case = '20130815'

#itime = (2006,5,26,22,0,0)
#itime = (2009,9,10,23,0,0)
#itime = (2011,4,19,18,0,0)
itime = (2013,8,15,0,0,0)
#ftime = (2006,5,27,11,0,0)
#ftime = (2009,9,11,14,0,0)
#ftime = (2011,4,20,10,30,0)
ftime = (2013,8,16,12,0,0)

times = utils.generate_times(itime,ftime,6*60*60)

ec_path = os.path.join(config.ecmwf_root,case,'ECMWF')
ec_abspath = os.path.join(ec_path,'*.nc')
# pdb.set_trace()
ec_fname = glob.glob(ec_abspath)[0]

config.output_root = os.path.join(config.output_root,case,'ECMWF')

scale = N.arange(54000,60000,300)

# pdb.set_trace()

E = ECMWF(ec_fname,config)

#E.plot('Z',850,times,scale=N.arange(13000,28000,150),wind_overlay=N.arange(10,30,2.5))
# E.plot('Z',700,times,scale=N.arange(28000,33000,200),wind_overlay=N.arange(10,40,2.5))
# E.plot('Z',500,times,scale=N.arange(54000,60000,300),wind_overlay=N.arange(15,50,2.5))
E.plot('Z',500,times,scale=N.arange(54000,60000,300),W_overlay=N.arange(-1.0,1.0,0.1))
# E.plot('Z',300,times,scale=N.arange(80000,100000,300),wind_overlay=N.arange(20,60,2.5))
