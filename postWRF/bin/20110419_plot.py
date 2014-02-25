import os
import pdb
import sys
sys.path.append('/home/jrlawson/gitprojects/')

from WEM.postWRF.postWRF import WRFEnviron
from settings import Settings

config = Settings()
p = WRFEnviron(config)

#variables = {'wind10':2000}
#variables = {'cref':2000}
variables = {'cref':2000,'wind10':2000}
#variables = {'wind':'sfc','cref':'na','thetae':850]
#variables = {'thetae':850}

itime = (2011,4,19,16,0,0)
ftime = (2011,4,20,11,30,0)
times = p.generate_times(itime,ftime,60*60)

p.plot_2D(variables,times)
#p.plot_upper_variable('thetae',850,times)
