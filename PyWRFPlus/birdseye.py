from figure import Figure
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

class BirdsEye(Figure):
    def __init__(self,config,wrfout,):
        self.C = config
        self.W = wrfout
        self.plot_time_idx = self.W.get_plot_time_idx(self.C.plottime)
        self.plot_time =
        self.plot_time_str =
    
    def plot2D(self,va,it,pt,en,do,lv,da):
        w,h = self.figsize(8,8)     # Create a default figure size if not set by user
        fig = plt.figure(figsize=(w,h))
        bmap,x,y = basemap_setup()
        data = self.W.get(va,self.plot_time_idx)
        #scale_lvs =
        bmap.contourf(x,y,data,scale_lvs)
        fpath = self.get_fpath()
        fname = self.get_fname()
        fig.savefig(fname,bbox_inches='tight')
        fig.clf()

    def basemap_setup(self):
        # Fetch settings
        basemap_res = getattr(self.C,'basemap_res',self.D.basemap_res)
        
        width_m = self.W.dx*(self.W.x_dim-1)
        height_m = self.W.dy*(self.W.y_dim-1)

        m = Basemap(
            projection='lcc',width=width_m,height=height_m,
            lon_0=self.W.cen_lon,lat_0=self.W.cen_lat,lat_1=self.W.truelat1,
            lat_2=self.W.truelat2,resolution=self.basemap_res,area_thresh=500
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

    
