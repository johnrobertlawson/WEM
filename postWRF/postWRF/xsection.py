"""Create cross-sections through WRF data.

This can be time-height or distance-height.
The height can be pressure, model, geometric, or geopotential

The output can be saved to a pickle file.
This can be useful for creating composites

Input lat/lon can be either specified or from x,y co-ords
calculated from manually clicking on a map with overlaid
data (reflectivity, etc).
"""

# Imports
import numpy as N
from figure import Figure
import pdb
import matplotlib.pyplot as plt

import WEM.utils as utils
import metconstants as mc
from scales import Scales
from birdseye import BirdsEye
# from defaults import Defaults

class CrossSection(Figure):

    def __init__(self,config,wrfout,latA=0,lonA=0,latB=0,lonB=0):
        super(CrossSection,self).__init__(config,wrfout)

        if latA and lonA and latB and lonB:
            print("Using user-defined lat/lon transects.")
            self.latA = latA
            self.lonA = lonA
            self.latB = latB
            self.lonB = lonB
            self.xA, self.yA = self.get_xy_from_latlon(latA,lonA)
            self.xB, self.yB = self.get_xy_from_latlon(latB,lonB)
        else:
            print("Please click start and end points.")
            self.popup_transect()
    
        self.get_xs_slice()

    def translate_xs(self,sh):
        """
        Translate the cross-section up or down a certain number of
        points.
        For simplicity with grid spacing, the logic allows for
        45 degree translation only.
        
        sh  :   number of points to shift
        """
        
        if (self.angle > 0.0) and (self.angle < 22.5):
            shxa = -sh; shxb = -sh; shya = 0;  shyb = 0
        elif (self.angle > 22.5) and (self.angle < 67.5):
            shxa = -sh; shxb = -sh; shya = sh;  shyb = sh
        elif (self.angle > 67.5) and (self.angle < 112.5):
            shxa = 0; shxb = 0; shya = sh;  shyb = sh
        elif (self.angle > 112.5) and (self.angle < 157.5):
            shxa = sh; shxb = sh; shya = sh;  shyb = sh
        elif (self.angle > 157.5) and (self.angle < 202.5):
            shxa = sh; shxb = sh; shya = 0;  shyb = 0
        elif (self.angle > 202.5) and (self.angle < 247.5):
            shxa = sh; shxb = sh; shya = -sh;  shyb = -sh
        elif (self.angle > 247.5) and (self.angle < 292.5):
            shxa = 0; shxb = 0; shya = -sh;  shyb = -sh
        elif (self.angle > 292.5) and (self.angle < 337.5):
            shxa = -sh; shxb = -sh; shya = -sh;  shyb = -sh
        elif (self.angle > 337.5) and (self.angle < 360.0):
            shxa = -sh; shxb = -sh; shya = 0;  shyb = 0
        else:
            print("Angle {0} is weird.".format(self.angle))
            raise Exception
            
        
        self.xA += shxa
        self.xB += shxb
        self.yA += shya
        self.yB += shyb
        self.xx = N.linspace(self.xA,self.xB,self.hyp_pts)
        self.yy = N.linspace(self.yA,self.yB,self.hyp_pts)

    def get_xs_slice(self):
        self.xA, self.yA = self.get_xy_from_latlon(self.latA,self.lonA)
        self.xB, self.yB = self.get_xy_from_latlon(self.latB,self.lonB)
        self.hyp_pts = int(N.hypot(self.xB-self.xA,self.yB-self.yA))
        self.xx = N.linspace(self.xA,self.xB,self.hyp_pts)
        self.yy = N.linspace(self.yA,self.yB,self.hyp_pts)
        # self.angle = N.radians(90.0) + N.arctan((self.yy[0]-self.yy[-1])/(self.xx[-1]-self.xx[0]))
        self.angle = N.math.atan2((self.yy[0]-self.yy[-1]),(self.xx[0]-self.xx[-1])) + N.pi
        # pdb.set_trace()
        return
    
    def popup_transect(self):
        """
        Pops up window for user to select start and
        end point for the cross-section.

        Optional: this map can be overlaid with data
        to better guide the decision.

        Optional: the transect can be saved as an image
        with/without the overlaid data, useful for publication.
        """
        self.fig.canvas.mpl_connect('pick_event',self.mouseclick)
        self.xA = self.xN
        self.yA = self.yN

        self.fig.canvas.mpl_connect('pick_event',self.mouseclick)
        self.xB = self.xN
        self.yB = self.yN

        self.latA, self.lonA = self.get_latlon_from_xy(xA,yA)
        self.latB, self.lonB = self.get_latlon_from_xy(xB,yB)
        
        return

    def get_latlon_from_xy(self,x,y):
        """
        Look up lat/lon in wrfout file from x and
        y coordinates.
        """
        lat = self.W.lat1D[y]
        lon = self.W.lon1D[x]
        return lat, lon

    def mouseclick(self,event):
        self.xN = event.mouseevent.xdata
        self.yN = event.mouseevent.ydata

    def get_xy_from_latlon(self,lat,lon):
        """
        Return x and y coordinates for given lat/lon.

        exactlat, exactlon      :   exact coordinates of closest x/y
        """
        y,x,exactlat,exactlon = utils.getXY(self.W.lats1D,self.W.lons1D,lat,lon)
        return x,y

    def get_height(self,t,x,y,z,pts):
        """
        Return terrain heights along cross-section
        
        TODO: What's the diff between the outputs?

        Inputs:
        t       :   time index as int
        x       :   x indices, as int
        y       :   y imdices, as int
        z       :   number of levels, as int
        pts     :   number of points along the x-sec

        Outputs:
        terrain_z   :   ter0,rain height
        heighthalf  :   who knows?

        Assuming t=0
        """
        # TODO: change API of W.get()
        geopot = self.W.get('geopot',{'t':t,'la':y,'lo':x})
        dryairmass = self.W.get('dryairmass',{'t':t,'la':y,'lo':x})
        znu = self.W.get('ZNU',{'t':t})
        znw = self.W.get('ZNW',{'t':t})

        heighthalf = N.zeros((z,pts))
        for i in range(pts):
            pfull = dryairmass[0,i] * znw[0,:]+self.W.P_top
            phalf = dryairmass[0,i] * znu[0,:]+self.W.P_top
            for k in range(z):
                heighthalf[k,i] = self.interp(geopot[0,:,i],pfull[:],phalf[k])/mc.g

        terrain_z = geopot[0,0,:]/mc.g
        # pdb.set_trace()
        # TODO: import metconstants as mc
        # return heighthalf, terrain_z
        return terrain_z,heighthalf

    def interp(self,geopot, pres, p):
        """ Returns the interpolated geopotential at p using the values in pres.
        The geopotential for an element in pres must be given by the corresponding
        element in geopot. The length of the geopot and pres arrays must be the same.
        """
        if (len(geopot) != len(pres)):
            raise Exception, "Arrays geopot and pres must have same length"
    
        k = len(pres)-1
        while (k > 1 and pres[k-1] <= p):
            k = k-1
    
        if (pres[k] > p):
            w = 0.0
        elif (pres[k-1] < p):
            w = 1.0
        else:
            w = (p-pres[k])/(pres[k-1]-pres[k])
    
        return (1.0-w)*geopot[k] + w*geopot[k-1]




    
    def plot_xs(self,vrbl,ttime,outpath,clvs=0,ztop=0):
        """
        Inputs:
        vrbl        :   variable to plot, from this list:
                        parawind,perpwind,wind,U,V,W,T,RH,
        ttime       :   time in ... format
        outpath     :   absolute path to directory to save output

        """
        tidx = self.W.get_time_idx(ttime)

        xint = self.xx.astype(int)
        yint = self.yy.astype(int)

        # Get terrain heights
        terrain_z, heighthalf = self.get_height(self.tidx,xint,yint,self.W.z_dim,self.hyp_pts)
        
        # Set up plot
        # Length of x-section in km
        xs_len = (1/1000.0) * N.sqrt((-1.0*self.hyp_pts*self.W.dy*N.cos(self.angle))**2 +
                                    (self.hyp_pts*self.W.dy*N.sin(self.angle))**2)
        
        # Generate ticks along cross-section
        xticks = N.arange(0,xs_len,xs_len/self.hyp_pts)
        xlabels = [r"%3.0f" %t for t in xticks]
        grid = N.swapaxes(N.repeat(N.array(xticks).reshape(self.hyp_pts,1),self.W.z_dim,axis=1),0,1)

        #########
        #### ADD SELF BELOW HERE
        #########

        # Plotting
        if self.W.dx != self.W.dy:
            print("Square domains only here")
        else:
            # TODO: allow easier change of defaults?
            self.fig.gca().axis([0,(hyp_pts*self.W.dx/1000.0)-1,self.D.plot_zmin,self.D.plot_zmax+self.D.plot_dz])

        # Logic time
        # First, check to see if v is in the list of computable or default
        # variables within WRFOut (self.W)
        # If not, maybe it can be computed here.
        # If not, raise Exception.
        # pdb.set_trace()
        # TODO: vailable_vars in WRFOut to check this
        ps = {'t':tidx, 'la':yint, 'lo':xint}
        
        if vrbl in self.W.available_vrbls:
            data = self.W.get(vrbl,ps)
            
            
        elif vrbl is 'parawind':
            u = self.W.get('U',ps)
            v = self.W.get('V',ps)
            data = N.cos(angle)*u - N.sin(angle)*v
            # clvs = u_wind_levels
            extends = 'both'
            CBlabel = r'Wind Speed (ms$^{-1}$)'

        elif vrbl is 'perpwind':
            u = self.W.get('U',ps)
            v = self.W.get('V',ps)
            # Note the negative here. I think it's alright? TODO
            data = -N.cos(angle)*v + N.sin(angle)*u
            clvs = u_wind_levels
            extends = 'both'
            CBlabel = r'Wind Speed (ms$^{-1}$)'

        else:
            print("Unsupported variable",vrbl)
            raise Exception

        # lv = '2000' # Very hacky, setting surface level
        # S = Scales(vrbl,lv)

        # multiplier = S.get_multiplier(vrbl,lv)

        # This is awful....
        # if S.cm:
            # cmap = S.cm
        # elif isinstance(S.clvs,N.ndarray):
            # if plottype == 'contourf':
                # cmap = plt.cm.jet
            # else:
                # pass
        # else:
            # cmap = plt.cm.jet

        # if clvs: pass # This is where to get clvs...
        #clvs = N.arange(-25,30,5)
        kwargs = {}
        kwargs['alpha'] = 0.6
        #kwargs['extend'] = 'both'
        if isinstance(clvs,N.ndarray):
            kwargs['levels'] = clvs
            
        cf = self.ax.contourf(grid,heighthalf,data[0,...],**kwargs)#,
        # cf = self.ax.contourf(grid[0,:],grid[:,0],data[0,...],alpha=0.6,extend='both')#levels=clvs,
                        # extend='both')#, cmap=cmap)
                        #norm=,

        # cf = self.ax.contourf(data[0,...])
        # pdb.set_trace()
        self.ax.plot(xticks,terrain_z,color='k',)
        self.ax.fill_between(xticks,terrain_z,0,facecolor='lightgrey')
        # What is this? TODO
        labeldelta = 15

        self.ax.set_yticks(N.arange(self.D.plot_zmin,
                                    self.D.plot_zmax+self.D.plot_dz,self.D.plot_dz))
        self.ax.set_xlabel("Distance along cross-section (km)")
        self.ax.set_ylabel("Height above sea level (m)")
        
        datestr = utils.string_from_time('output',ttime)

        if ztop:
            self.ax.set_ylim([0,ztop*1000])
        naming = [vrbl,'xs',datestr]
        fname = self.create_fname(*naming)
        self.save(self.fig,outpath,fname)
        #self.close()
        
        CBlabel = str(vrbl)
        
        # Save a colorbar
        # Only if one doesn't exist
        self.just_one_colorbar(outpath,fname+'CB',cf,label=CBlabel)
        
        self.draw_transect(outpath,fname+'Tsct')

    def just_one_colorbar(self,fpath,fname,cf,label):
        """docstring for just_one_colorbar"""
        try:
            with open(fpath): pass
        except IOError:
            self.create_colorbar(fpath,fname,cf,label=label)

    
    def draw_transect(self,outpath,fname):
        B = BirdsEye(self.C,self.W)
        m,x,y = B.basemap_setup()
        m.drawgreatcircle(self.lonA,self.latA,self.lonB,self.latB)
        self.save(B.fig,outpath,fname)
        
    def create_linenormal_xs(self,x,y,length_pts=3):
        """
        Return a cross-section that runs normal to the
        existing cross-section contained in self.
        
        x,y         :   coordinates of intersection
        length_pts  :   length of normal line in grid points
        """
    
        self.hyp_pts, self.xx, self.yy = self.get_xs_slice()

        xint = self.xx.astype(int)
        yint = self.yy.astype(int)
        self.angle = N.arctan((self.yy[-1]-self.yy[0])/(self.xx[-1]-self.xx[0])) # In radians
        
        normal_angle = self.angle + N.radians(90)
        # pdb.set_trace()
        
        
        #return norm_xx, norm_yy
        return 1,2