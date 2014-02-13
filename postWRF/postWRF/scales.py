""" This file contains colorbar scales for plotting.

INPUTS

va      :   variable to be plotted
lv      :   level to be plotted

OUTPUTS

cm      :   colour map
clvs    :   contour levels

Level = 2000 indicates surface.

Default settings are listed at the bottom.
These can be overwritten in user's config file... somehow
"""
import pdb
from matplotlib.colors import LinearSegmentedColormap
import numpy as N
import matplotlib.pyplot as plt

import colourtables as ct

def get_cm(va,lv):
    
    # Variable and vertical level determine contour scale
    # pdb.set_trace()
    try:
        if len(A[va][lv]) == 3:
            # This is a min-max-interval list
            clvs = N.arange(*A[va][lv])
        else:
            # This is an actual list of values
            clvs = A[va][lv]
    except KeyError:
        # If no level exists, try finding a near one
        try:
            near_lv = find_nearest_level(lv)
            clvs = A[va][near_lv]
        except:
            # Some variables don't live on a vertical level
            clvs = 0
    # except:
        # raise Exception
        
    try:
        cm = A[va]['cmap'](clvs)
        #pdb.set_trace()
    except TypeError:
        #print("Using default colourtable.")
        #def_ct = plt.cm.get_cmap("jet")
        cm = 0    
        #cm = LinearSegmentedColormap('DEF_CT',def_ct)
    # else:
        # raise Exception
    # print va
    # print type(cm)
    # pdb.set_trace()
    return cm, clvs
    
def find_nearest_level(lv):
    lv_type = utils.level_type(lv)
    
    if lv_type == 'isentropic':
        pass
        # 'K' needs stripping
        # This will be tricky, varies a lot...
    elif lv_type == 'isobaric':
        pass
        # Plot logarithmically closest
    elif lv_type == 'surface':
        raise Exception
        # Shouldn't get here, surface should be covered.
    elif lv_type == 'PV-surface':
        pass
    elif lv_type == 'geometric':
        pass
    else:
        raise Exception
        
    return near_lv
    
######## DEFAULT SETTINGS FOR LEVELS ########

A = {}

# Wind magnitude
A['wind10'] = {'cmap':0} 
A['wind10'][2000] = (5,32.5,2.5)

# Theta-e (Equivalent potential temperature)
# A['thetae'] = {'cmap':ct.thetae}

# Simulated reflectivity
A['sim_ref'] = {'cmap':ct.reflect_ncdc}
A['sim_ref'][2000] = (5,90,5)

# Simulated reflectivity
A['cref'] = {'cmap':ct.reflect_ncdc}
A['cref'][2000] = (5,90,5)

# Precipitation
A['precip'] = {'cmap':ct.precip1}
A['precip'][2000] = [0.01,0.03,0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,
                0.70,0.80,0.90,1.00,1.25,1.50,1.75,2.00,2.50]
                
# Precipitable water
A['pwat'] = {'cmap':ct.precip1} 
A['pwat'][2000] = (0.2,2.6,0.1)

# Snowfall
A['snow'] = {'cmap':ct.snow2}
A['snow'][2000] = [0.25,0.5,0.75,1,1.5,2,2.5,3,4,5,6,8,10,12,14,16,18]



