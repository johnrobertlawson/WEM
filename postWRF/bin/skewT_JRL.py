import os
import pdb
import sys

sys.path.append('/home/jrlawson/gitprojects/')

from WEM.postWRF.postWRF import WRFEnviron
from settings import Settings
import WEM.utils.utils as utils

config = Settings()
p = WRFEnviron(config)

# NUMBER 2: CHANGE FOLDER DETAILS HERE
case = '20130815'

#ensembletype = 'STCH'
ensembletype = 'MXMP'
IC = 'GEFSR2'
ICens = ['p09',]
#MP = 'Morrison_Hail'
MP = 'ICBC'


# NUMBER 3: SET THE TIME(s) YOU WANT PLOT
# Loop over times...
itime = (2013,8,15,18,0,0)
ftime = (2013,8,16,12,0,0)
times = utils.generate_times(itime,ftime,60*60)

# ... or set just one time
skewT_time = (2013,8,16,3,0,0)

# NUMBER 4: SET THE LAT/LON
skewT_latlon = (35.2435,-97.4708)

# NUMBER 5: 
# This is a loop over different folders
# For one plot, just use the (last line) plot_skewT command

"""
for en in ensnames:
    for ex in experiments:
        # Reload settings
        p.C = Settings()

        # Change paths to new location
        p.C.output_root = os.path.join(config.output_root,case,IC,en,MP,ex)
        p.C.wrfout_root = os.path.join(config.wrfout_root,case,IC,en,MP,ex)
        p.C.pickledir = os.path.join(config.wrfout_root,case,IC,en,MP,ex)

        # save_output saves pickle files for later use
        p.plot_skewT(skewT_time,skewT_latlon,save_output=1)
"""
path_to_wrfouts = []

if ensembletype == 'STCH':
    experiments = ['s'+"%02d" %n for n in range(1,11)]
    p.C.output_root = os.path.join(config.output_root,case,IC,ICens[0],MP)
    for ens in ICens:
        for ex in experiments:
            fpath = os.path.join(config.wrfout_root,case,IC,ens,MP,ex)
            path_to_wrfouts.append(p.wrfout_files_in(fpath,dom=1)[0])

elif ensembletype == 'MXMP':
    experiments = ('ICBC','WSM6_Grau','WSM6_Hail','Kessler','Ferrier','WSM5',
            'WDM5','Lin','WDM6_Grau','WDM6_Hail',
            'Morrison_Grau','Morrison_Hail')
    p.C.output_root = os.path.join(config.output_root,case,IC,ICens[0])
    for ens in ICens:
        for ex in experiments:
            fpath = os.path.join(config.wrfout_root,case,IC,ens,ex)
            path_to_wrfouts.append(p.wrfout_files_in(fpath,dom=1)[0])

ylim = 0
va = 'theta' ; xlim = (298,320,4) ; ylim=(1000,500,50)
#va = 'RH' ; xlim = (0,100,5)
#va = 'wind' ; xlim = (0,35,5)
p.composite_profile(va,skewT_time,skewT_latlon,path_to_wrfouts,mean=1,std=1,xlim=xlim,ylim=ylim)

