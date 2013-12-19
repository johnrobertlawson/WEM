"""Edit this file to taste.

The class PyWRFSettings (in settings.py, same folder) is used to create config, containing all settings
"""

import sys
sys.path.append('../')
from postWRF import WRFEnviron
from casestudy_settings import Settings
import pdb
import os

# Initialise configuration
config = Settings()
# Initialise plotting environment
p = WRFEnviron(config)

#config.datafolder = os.path.join('2009091000','GEFS','CTRL','c00')

#path_to_wrfouts = p.wrfout_files_in('/tera9/jrlawson/bowecho/2009091000/GEFS/CTRL/c00')
casenames = 'bowecho'
init_time = p.string_from_time('dir',(2009,9,10,0,0,0),strlen='hour')
ICs = 'GEFS'
sens = 'STCH'
IC_n = 'p09'
sens_n = 's01'
dom = 1
wrfout_dir = p.dir_from_naming(config.wrfout_root,casenames,init_time,ICs,sens,IC_n,sens_n)
path_to_plots = p.dir_from_naming(config.output_root,casenames,init_time,ICs,sens,IC_n,sens_n)
path_to_wrfouts = p.wrfout_files_in(wrfout_dir,dom=1)
# Things to loop over:
variables = ['wind']
plot_times = [(2009,9,11,n,0,0) for n in range(5,7)]
ens_members = path_to_wrfouts
levels = 2000 # hPa levels would require pinterp rewrite into Python?
domain_areas = 0 # sequence of dictionaries with N,E,S,W representing lat/lon limits.
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

p.plot_variable2D(variables,plot_times,ens_members,levels,path_to_plots)  # naming = some scheme
#p.plot_postage(va='simref',pt=(2009,9,11,12,0,0),it=(2009,9,10,0,0,0),en=ens_members)
#p.plot_xs(contour='theta',contourf='wind',pt,it,latA,lonA,latB,lonB)
#p.plot_DTE()
#p.plot_DKE()


