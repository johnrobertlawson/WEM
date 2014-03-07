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
import WEM.utils.utils as utils

class Figure:
    def __init__(self,config,wrfout):
        self.C = config
        self.W = wrfout
    
    def create_fname(self,*naming):
        """Default naming should be:
        Variable + time + level
        """
        fname = '_'.join([str(a) for a in naming]) 
        #pdb.set_trace()
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

    def get_limited_domain(self,da):
        if da:  # Limited domain area 
            N_idx = self.W.get_lat_idx(da['N'])
            E_idx = self.W.get_lon_idx(da['E'])
            S_idx = self.W.get_lat_idx(da['S'])
            W_idx = self.W.get_lon_idx(da['W'])

            lat_sl = slice(S_idx,N_idx)
            lon_sl = slice(W_idx,E_idx)
        else:
            lat_sl = slice(None)
            lon_sl = slice(None)
            
        return lat_sl, lon_sl
