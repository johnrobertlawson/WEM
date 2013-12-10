"""Edit this file to taste.

The class PyWRFSettings (in settings.py, same folder) is used to create config, containing all settings
"""

import sys
sys.path.append('../')
from PyWRFPlus import PyWRFEnv
from settings import PyWRFSettings
import pdb
import os

# Initialise configuration
config = PyWRFSettings()

# Initial time of WRF data file
config.inittime = (2009,9,10,0,0,0)
# Desired time for plot
config.plottime = (2009,9,11,9,0,0)
# Set data folder - work out dynamically for looping etc
config.datafolder = os.path.join('2009091000','GEFS','CTRL','c00')
# Initialise plotting environment
p = PyWRFEnv(config)
# Run some plotting scripts
p.plot_sim_ref()

# Create a tuple of times for iterating over
timelist = [(2006,5,27,n,0,0) for n in range(1,7)]
for t in timelist:
    pass
    