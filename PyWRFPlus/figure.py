"""Collection of x-y cross-section classes.

"""

# Imports
import numpy as N
import matplotlib as M
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pdb

# Custom imports
import utils

class Figure:
    def __init__(self,config,wrfout):
        # wrff is details about the wrf files
        # config has user settings for plot etc
        self.C = config
        self.W = wrfout
        #self.fig = plt.figure()        
     
    def title_time(self):
        self.T = utils.padded_times(self.timeseq) 
        pdb.set_trace()


    def figsize(self,defwidth,defheight):
        width = getattr(self.C,'width',defwidth)
        height = getattr(self.C,'height',defheight)
        self.fig.set_size_inches(width,height)

    def save_fig(self):
        loc = self.C.output_dir # For brevity
        utils.trycreate(loc)
        fname = 'blah.png'
        fpath = os.path.join(loc,fname)
        #self.fig.savefig(fpath)
        plt.gcf().savefig(fpath,bbox_inches='tight')

