from figure import Figure
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

class BirdsEye(Figure):
    def __init__(self,config,wrfout,):
        self.C = config
        self.W = wrfout
    
    def plot2D(self,va,pt,en,do,lv,da):
        w,h = self.figsize(8,8)     # Create a default figure size if not set by user
        fig = plt.figure(figsize=(w,h))
        bmap,x,y = basemap_setup()
        
        # Work out time, level, lats, lons index
        
        time_idx = self.W.get_time_idx(pt)
        
        if lv = 2000:
            lv_idx = 0
        else:
            print("Non-surface levels not supported yet.")
            raise Exception
        
        if da:  # Limited domain area 
            N_idx = self.W.get_lat_idx(da['N'])
            E_idx = self.W.get_lon_idx(da['E'])
            S_idx = self.W.get_lat_idx(da['S'])
            W_idx = self.W.get_lon_idx(da['W'])

            lat_sl = slice(S_idx:N_idx)
            lon_sl = slice(W_idx:E_idx)
        else:
            lat_sl = slice(None)
            lon_sl = slice(None)
            
        data = self.W.get(va,time_idx,lv_idx,lat_sl,lon_sl)
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

    
