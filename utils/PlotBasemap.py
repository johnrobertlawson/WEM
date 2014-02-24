import numpy as N
import matplotlib as M
M.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

class PlotBasemap(Basemap):
    def __init__(self):
        self.Nlim = 50
        self.Elim = -100
        self.Slim = 30
        self.Wlim = -130
        self.initial()
    
    def initial():
        (projection....)
        
    def changeView(self,domain):
        if domain == 'UK':
            pass

    def outlines(self):
        # Draw coastlines
    def saveBasemap(self,outdir,fname):
        self.fig = plt.gcf()
        plt.savefig(outdir+fname)
