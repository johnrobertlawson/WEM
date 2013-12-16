import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from defaults import Defaults
from figure import Figure
import utils

class BirdsEye(Figure):
    def __init__(self,config,wrfout,p2p):
        self.C = config
        self.W = wrfout
        self.D = Defaults()
        self.p2p = p2p
    
    def plot2D(self,va,pt,en,lv,da,na):
        self.fig = plt.figure()
        self.fig = self.figsize(8,8,self.fig)     # Create a default figure size if not set by user
        self.bmap,x,y = self.basemap_setup()
        
        # Work out time, level, lats, lons index
        
        time_idx = self.W.get_time_idx(pt)
        
        if lv == 2000:
            lv_idx = 0
            lv = 'sfc' # For naming purposes
        else:
            print("Non-surface levels not supported yet.")
            raise Exception
        
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
        
        # Plot settings
        PS = {'t': time_idx, 'lv': lv_idx, 'la': lat_sl, 'lo': lon_sl} 
        data = self.W.get(va,PS)

        # Set user scale
        # If not set, use default
        # If no default, do auto-scale.

        try: 
            scale_lvs = self.C.scales[va]
            SL = N.arange
        except:
            try:
                scale_lvs = self.D.scales[va]
            except:
                scale_lvs = 0

        #self.bmap.contourf(x,y,data,scale_lvs)
        la_n = data.shape[-2]
        lo_n = data.shape[-1]
        if not scale_lvs:
            self.bmap.contourf(x,y,data.reshape((la_n,lo_n))) # Dimension thing again...
        else:
            self.bmap.contourf(x,y,data.reshape((la_n,lo_n)),scale_lvs)
        if self.C.plot_titles:
            title = utils.string_from_time('title',pt)
            plt.title(title)
        if self.C.colorbar:
            plt.colorbar(orientation='horizontal')

        datestr = utils.string_from_time('output',pt)
        if not na:
            # Use default naming scheme
            na = (va,lv,datestr)
        else:
            # Come up with scheme...
            print("Coming soon: ability to create custom filenames")
            raise Exception
        self.fname = self.create_fname(na) # No da variable here
        self.save(self.fig,self.p2p,self.fname)
        self.fig.clf()

    def basemap_setup(self):
        # Fetch settings
        basemap_res = getattr(self.C,'basemap_res',self.D.basemap_res)
        
        width_m = self.W.dx*(self.W.x_dim-1)
        height_m = self.W.dy*(self.W.y_dim-1)

        m = Basemap(
            projection='lcc',width=width_m,height=height_m,
            lon_0=self.W.cen_lon,lat_0=self.W.cen_lat,lat_1=self.W.truelat1,
            lat_2=self.W.truelat2,resolution=basemap_res,area_thresh=500
            )
        m.drawcoastlines() 
        m.drawstates()
        m.drawcountries()

        # Draw meridians etc with wrff.lat/lon spacing
        # Default should be a tenth of width of plot, rounded to sig fig

        #self.W.lons = nc.variables['XLONG'][0]
        #self.W.lats = nc.variables['XLAT'][0]
        #pdb.set_trace()
        x,y = m(self.W.lons,self.W.lats)
        return m, x, y

    
