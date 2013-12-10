"""Collection of x-y cross-section classes.

"""

# Imports
import numpy as N
import matplotlib as M
from mpl_toolkits.basemap import Basemap
import pdb
# Custom imports
import meteogeneral

class Figure:
    def __init__(self,config,wrfout):
        # wrff is details about the wrf files
        # config has user settings for plot etc
        self.C = config
        self.W = wrfout
        

