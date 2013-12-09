"""Collection of x-y cross-section classes.

Describe each class here.

"""

# Imports
import numpy as N
import matplotlib as M
from mpl_toolkits.basemap import Basemap

# Custom imports
import meteogeneral

class BirdsEye:
    def __init__(self,fig,config,wrff):
        
    
    def plot2D(self,config):

    def basemap_setup(self):
        self.map = Basemap()
        xlong = nc.variables['XLONG'][0]
        xlat = nc.variables['XLAT'][0]
        x,y = map(xlong,xlat)
