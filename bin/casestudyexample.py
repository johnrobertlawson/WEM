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
# Initialise plotting environment
p = PyWRFEnv(config)

#config.datafolder = os.path.join('2009091000','GEFS','CTRL','c00')

path_to_wrfouts = p.all_WRF_files_in('/path/to/ensemble/folder/') # Method that serves up all wrfout* files

# Run some plotting scripts
p.plot_sim_ref()

# Things to loop over:
plot_times = [(2009,9,11,n,0,0) for n in range(1,7)]
init_times = [(2009,9,10,0,0,0)]
variables = ['T2','shear_0_3']
ens_members = [path_to_wrfouts]
levels = [2000] # hPa levels would require pinterp rewrite into Python?
domains = [1]
domain_areas = [] # ul, lr corners lat/lon to form box

"""
Perhaps load all wrfout files found in datafolder.
Try to find initialisation time in each file's name
If unambiguous, that's your file
If not, raise error: need ens_member setting to differentiate

For postage plots, specify folder and expect all wrfout files inside to be used
Or individually name files you would like.
Title of each plot becomes folder name
If you want custom titles (e.g. "Kessler"), create dictionary passed in call? 
"""


# I would like to plot these things
# Accepting: strings, list of strings, list of one string
# Looping over all permutations of each.

p.plot_variable2D(variables,plot_times,init_times,domains,levels)
p.plot_postage(va='simref',pt=(2009,9,11,12,0,0),it=(2009,9,10,0,0,0),en=ens_members)
p.plot_xs(contour='theta',contourf='wind',pt,it,latA,lonA,latB,lonB)
p.plot_DTE()
p.plot_DKE()


