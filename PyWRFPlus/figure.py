"""Collection of x-y cross-section classes.

"""

# Imports
import numpy as N
import matplotlib as M
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pdb
import os

# Custom imports
import utils

class Figure:
    def __init__(self,config,wrfout):
        pass
    
    def create_fname(self,naming):
        """Default naming should be:
        Variable + time + level
        """
        fname = '_'.join([str(a) for a in naming]) 
        return fname  
 
    def title_time(self):
        self.T = utils.padded_times(self.timeseq) 
        pdb.set_trace()

    def figsize(self,defwidth,defheight,fig):
        width = getattr(self.C,'width',defwidth)
        height = getattr(self.C,'height',defheight)
        fig.set_size_inches(width,height)
        return fig

    def save(self,fig,p2p,fname):
        utils.trycreate(p2p)
        fpath = os.path.join(p2p,fname)
        #self.fig.savefig(fpath)
        plt.gcf().savefig(fpath,bbox_inches='tight')

