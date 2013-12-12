"""Take config settings and run plotting scripts.

Classes:
C = Config = configuration settings set by user, passed from user script
D = Defaults = used when user does not specify a non-essential item
W = Wrfout = wrfout file
F = Figure = a superclass of figures
    mp = Birdseye = a lat--lon slice through data with basemap
    xs = CrossSection = distance--height slice through data with terrain 

"""

import os
import matplotlib as M
M.use('Agg')
import matplotlib.pyplot as plt
import collections

from wrfout import WRFOut
from axes import Axes
from figure import Figure
from birdseye import BirdsEye
import scales
from defaults import Defaults
import utils

class PyWRFEnv:
    def __init__(self,config):
        self.C = config

        # Set defaults if they don't appear in user's settings
        self.D = Defaults()
        
        self.font_prop = getattr(self.C,'font_prop',self.D.font_prop)
        self.usetex = getattr(self.C,'usetex',self.D.usetex)
        self.dpi = getattr(self.C,'dpi',self.D.dpi)
        self.plot_titles = getattr(self.C,'plot_titles',self.D.plot_titles) 

        # Set some general settings
        M.rc('text',usetex=self.usetex)
        M.rc('font',**self.font_prop)
        M.rcParams['savefig.dpi'] = self.dpi

    def all_WRF_files_in(self,folder,prefix='wrfout'):
        wrfouts = []
        for root,dirs,files in os.walk(folder):
            for fname in fnmatch.filder(files,prefix+'*'):
                wrfouts.append(os.path.join(root, fname))
        return wrfouts
        
    def plot_variable2D(va,it,pt,en,do,lv,da=0)
        """Plot a longitude--latitude cross-section (bird's-eye-view).
        Use Basemap to create geographical data
        
        va = variable(s)
        it = initial time(s) of WRF run
        pt = plot time(s)
        en = ensemble member(s)
        do = domain(s)
        lv = level(s) 
        da = smaller domain area(s)
        """
        
        va = self.get_sequence(va)
        it = self.get_sequence(it)
        pt = self.get_sequence(pt)
        en = self.get_sequence(en)
        do = self.get_sequence(do)
        lv = self.get_sequence(lv)
        da = self.get_sequence(da)
        
        perms = self.make_iterator(va,it,pt,en,do,lv,da)
        
        # Find some way of looping over wrfout files first, avoiding need
        # to create new W instances
        
        for n,x in enumerate(perms):
            va,it,pt,en,do,lv,da = x
            
            W = WRFOut(en)    # wrfout file class using path
            F = BirdsEye(self.C)    # 2D figure class
            F.plot2D(va,it,pt,en,do,lv,da)  # Plot/save figure
            print("Plotting #" + str(n) + " of " + str(len(perms)))
        
    def make_iterator(va,it,pt,en,do,lv,da):   
        for v in va:
            for i in it:
                for p in pt:
                    for e in en:
                        for d in do:
                            for l in lv:
                                for a in da:
                                    yield v,i,p,e,d,l,a
        
    def get_sequence(x):
        """ Returns a sequence (tuple or list) for iteration."""
        if isinstance(x, collections.Sequence) and not isinstance(x, basestring):
            return x
        else:
            return (x)
            
    def plot_CAPE(self,datatype='MLCAPE'):
        pass
    
    def plot_shear(self,upper=3,lower=0):
        #self.C.plottype = getattr(self.C,'plottype','contourf')
        #fig = axes.setup(C)
        shear = WRFOut.compute_shear(upper,lower)
    
    def plot_cross_section(self,var,latA,lonA,latB,lonB):
        xs = CrossSection()
        xs.plot(var,latA,lonA,latB,lonB)
        
    def plot_DKE(self,dims=1):
        # dims = 1 leads to time series
        # dims = 2 vertically integrates to xxx hPa or xx sigma level?
        pass
    
    def plot_DTE(self):
        pass
    
    def plot_sim_ref(self,reftype='composite'):
        self.mp = BirdsEye(self.C,self.W,self.C.plottime)
        self.mp.figsize(8,8)   # Sets default width/height if user does not specify
        self.mp.scale, self.mp.cmap = scales.comp_ref() # Import scale and cmap
        self.bmap, self.x, self.y = self.mp.basemap_setup()
 
        if reftype == 'composite':
            self.data = self.W.compute_comp_ref(self.mp.W.time_idx)
            self.mp.title = 'Composite Simulated Reflectivity'
        elif isinstance(reftype,int):
            self.data = self.W.compute_simref_atlevel()
            self.mp.title = 'Simulated Reflectivity at model level #' + str(reftype)

        self.mp.title_time()  
        self.bmap.contourf(self.x,self.y,self.data,self.mp.scale)
        if self.title:
            plt.title(self.mp.title)
        self.mp.save_fig()
        
    def plot_var(self,varlist):
        pass
        # This could be a combination of surface and upper-air data
    
    def sfc_data(self,varlist):
        # Varlist will be dictionary
        # Key is variable; value is plot type (contour, contour fill)
        # Some way of choosing plotting order?
        for v,p in varlist:
            # Plot data on top of each other in some order?
            pass            
    def upper_lev_data(self,level):
        # Levels: isentropic, isobaric, geometric,
        pass 
    

