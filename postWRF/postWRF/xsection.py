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
from .figure import Figure
import pdb
import matplotlib.pyplot as plt

import WEM.utils as utils
from WEM.utils import metconstants as mc
from .scales import Scales
from .birdseye import BirdsEye
# from defaults import Defaults

class CrossSection(Figure):

    def __init__(self,wrfout,latA=0,lonA=0,latB=0,lonB=0):
        super(CrossSection,self).__init__(wrfout)

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
        self.angle = N.degrees(self.angle) 
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
            print(("Angle {0} is weird.".format(self.angle)))
            raise Exception
            
        print(("Old coordinates:",self.xA,self.xB,self.yA,self.yB))
        self.xA += shxa
        self.xB += shxb
        self.yA += shya
        self.yB += shyb
        self.xx = N.linspace(self.xA,self.xB,self.hyp_pts)
        self.yy = N.linspace(self.yA,self.yB,self.hyp_pts)
        print(("New coordinates:",self.xA,self.xB,self.yA,self.yB))

        self.angle = N.radians(self.angle) 

    def get_xs_slice(self):
        self.xA, self.yA = self.get_xy_from_latlon(self.latA,self.lonA)
        self.xB, self.yB = self.get_xy_from_latlon(self.latB,self.lonB)
        self.hyp_pts = int(N.hypot(self.xB-self.xA,self.yB-self.yA))
        self.xx = N.linspace(self.xA,self.xB,self.hyp_pts)
        self.yy = N.linspace(self.yA,self.yB,self.hyp_pts)
        # self.angle = N.radians(90.0) + N.arctan((self.yy[0]-self.yy[-1])/(self.xx[-1]-self.xx[0]))
        # self.angle = N.math.atan2((self.yy[-1]-self.yy[0]),(self.xx[-1]-self.xx[0])) + N.pi
        self.angle = N.math.atan2((self.yy[0]-self.yy[-1]),(self.xx[0]-self.xx[-1])) + N.pi
        # print('angle = ',self.angle)
        # self.angle = N.math.atan2((self.yy[0]-self.yy[-1]),(self.xx[0]-self.xx[-1])) + N.pi
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

    def get_wrfout_slice(self,vrbl,utc=False,level=False,x=False,y=False):
        data = self.W.get(vrbl,utc=utc,level=level)
        sliced = data[:,:,y,x]
        return sliced

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
        # geopot = self.W.get('geopot',utc=t,lats=y,lons=x)
        geopot = N.swapaxes(self.get_wrfout_slice('geopot',utc=t,y=y,x=x)[0,:,:],1,0) # 2D
        
        # dryairmass = self.W.get('dryairmass',utc=t,lats=y,lons=x)
        dryairmass = self.get_wrfout_slice('dryairmass',utc=t,y=y,x=x)[0,0,:] #1D
        znu = self.W.get('ZNU',utc=t)[0,:,0,0] # 1D
        znw = self.W.get('ZNW',utc=t)[0,:,0,0] # 1D

        heighthalf = N.zeros((pts,z))
        for i in range(pts):
            pfull = dryairmass[i] * znw[:]+self.W.P_top
            phalf = dryairmass[i] * znu[:]+self.W.P_top
            for k in range(z):
                # import pdb; pdb.set_trace()
                heighthalf[i,k] = self.interp(geopot[i,:],pfull[:],phalf[k])/mc.g

        terrain_z = geopot[:]/mc.g
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
            raise Exception("Arrays geopot and pres must have same length")
    
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


    def plot_average(self,vrbl,avepts,ttime,outpath,clvs=0,ztop=0,f_suffix=False,
            cmap='jet',contour_vrbl='skip',contour_clvs=False,
            cflabel=False,cftix=False):

        self.tidx = self.W.get_time_idx(ttime)
        AVEDATA = {'cf_vrbl':{},'ct_vrbl':{}}
        for shn in range(-avepts,avepts+1):
            if shn == -avepts:
                self.translate_xs(shn)
            else:
                self.translate_xs(1)
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
            grid = N.repeat(N.array(xticks).reshape(self.hyp_pts,1),self.W.z_dim,axis=1)

            # Plotting
            if self.W.dx != self.W.dy:
                print("Square domains only here")
            else:
                # TODO: allow easier change of defaults?
                self.fig.gca().axis([0,(self.hyp_pts*self.W.dx/1000.0)-1,self.D.plot_zmin,self.D.plot_zmax+self.D.plot_dz])
           
            for nn, v in enumerate([vrbl,contour_vrbl]):
                print(v)
                if v == 'skip':
                    continue
                elif v in self.W.available_vrbls:
                    # data = self.W.get(vrbl,**ps)
                    data = self.get_wrfout_slice(v,utc=self.tidx,y=yint,x=xint)
                    
                elif v is 'parawind':
                    # u = self.W.get('U',**ps)
                    # v = self.W.get('V',**ps)
                    u = self.get_wrfout_slice('U',utc=self.tidx,y=yint,x=xint)
                    v = self.get_wrfout_slice('V',utc=self.tidx,y=yint,x=xint)
                    data = N.cos(self.angle)*u + N.sin(self.angle)*v
                    # data = N.cos(self.angle)*u - N.sin(self.angle)*v
                    # import pdb; pdb.set_trace()
                    # clvs = u_wind_levels
                    extends = 'both'
                    CBlabel = r'Wind Speed (ms$^{-1}$)'

                elif v is 'perpwind':
                    u = self.get_wrfout_slice('U',utc=self.tidx,y=yint,x=xint)
                    v = self.get_wrfout_slice('V',utc=self.tidx,y=yint,x=xint)
                    # u = self.W.get('U',ps)
                    # v = self.W.get('V',ps)
                    # Note the negative here. I think it's alright? TODO
                    data = -N.cos(self.angle)*v + N.sin(self.angle)*u
                    # clvs = u_wind_levels
                    extends = 'both'
                    CBlabel = r'Wind Speed (ms$^{-1}$)'

                else:
                    print(("Unsupported variable",v))
                    raise Exception

                if nn == 0:
                    data = N.swapaxes(data[0,:,:],1,0)    
                    AVEDATA['cf_vrbl'][shn] = data
                else:
                    data = N.swapaxes(data[0,:,:],1,0)    
                    AVEDATA['ct_vrbl'][shn] = data

        for nn, v in enumerate([vrbl,contour_vrbl]):
            if nn == 0:
                kwargs = {}
                kwargs['alpha'] = 0.6
                kwargs['extend'] = 'both'
                if isinstance(clvs,N.ndarray):
                    kwargs['levels'] = clvs
               
                alldata = N.zeros(((2*avepts)+1,grid.shape[0],grid.shape[1]))
                for n,nn in enumerate(AVEDATA['cf_vrbl'].keys()):
                    alldata[n,:,:] = AVEDATA['cf_vrbl'][nn]
                avedata = N.mean(alldata,axis=0)
                cf = self.ax.contourf(grid,heighthalf,avedata,cmap=cmap,**kwargs)#,
            elif contour_vrbl is not 'skip':
                alldata = N.zeros(((2*avepts)+1,grid.shape[0],grid.shape[1]))
                for n,nn in enumerate(AVEDATA['ct_vrbl'].keys()):
                    alldata[n,:,:] = AVEDATA['ct_vrbl'][nn]
                avedata = N.mean(alldata,axis=0)
                ct = self.ax.contour(grid,heighthalf,data,colors=['k',],levels=contour_clvs,linewidths=0.3)
                self.ax.clabel(ct,inline=1,fontsize=6,fmt='%d')

        self.ax.fill_between(xticks,terrain_z[:,0],0,facecolor='lightgrey')
        labeldelta = 15

        self.ax.set_yticks(N.arange(self.D.plot_zmin,
                                    self.D.plot_zmax+self.D.plot_dz,self.D.plot_dz))
        self.ax.set_xlabel("Distance along cross-section (km)")
        self.ax.set_ylabel("Height above sea level (m)")
        
        datestr = utils.string_from_time('output',ttime)

        if ztop:
            self.ax.set_ylim([0,ztop*1000])
        naming = [vrbl,'xs_ave',datestr,f_suffix]
        fname = self.create_fname(*naming)
        self.save(outpath,fname)
        #self.close()
        
        CBlabel = str(vrbl)
        
        # Save a colorbar
        # Only if one doesn't exist
        self.just_one_colorbar(outpath,fname+'CB',cf,label=cflabel,tix=cftix)
        
        self.draw_transect(outpath,fname+'Tsct')
        plt.close(self.fig)

    
    def plot_xs(self,vrbl,ttime,outpath,clvs=0,ztop=0,f_suffix=False,
                    cmap='jet',contour_vrbl='skip',contour_clvs=False):
        """
        Inputs:
        vrbl        :   variable to plot, from this list:
                        parawind,perpwind,wind,U,V,W,T,RH,
        ttime       :   time in ... format
        outpath     :   absolute path to directory to save output

        """
        self.tidx = self.W.get_time_idx(ttime)

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
        grid = N.repeat(N.array(xticks).reshape(self.hyp_pts,1),self.W.z_dim,axis=1)

        #########
        #### ADD SELF BELOW HERE
        #########

        # Plotting
        if self.W.dx != self.W.dy:
            print("Square domains only here")
        else:
            # TODO: allow easier change of defaults?
            self.fig.gca().axis([0,(self.hyp_pts*self.W.dx/1000.0)-1,self.D.plot_zmin,self.D.plot_zmax+self.D.plot_dz])

        # Logic time
        # First, check to see if v is in the list of computable or default
        # variables within WRFOut (self.W)
        # If not, maybe it can be computed here.
        # If not, raise Exception.
        # pdb.set_trace()
        # TODO: vailable_vars in WRFOut to check this
        # ps = {'utc':self.tidx, 'lats':yint, 'lons':xint}
       
        for nn, v in enumerate([vrbl,contour_vrbl]):
            print(v)
            if v == 'skip':
                continue
            elif v in self.W.available_vrbls:
                # data = self.W.get(vrbl,**ps)
                data = self.get_wrfout_slice(v,utc=self.tidx,y=yint,x=xint)
                
            elif v is 'parawind':
                # u = self.W.get('U',**ps)
                # v = self.W.get('V',**ps)
                u = self.get_wrfout_slice('U',utc=self.tidx,y=yint,x=xint)
                v = self.get_wrfout_slice('V',utc=self.tidx,y=yint,x=xint)
                data = N.cos(self.angle)*u + N.sin(self.angle)*v
                # data = N.cos(self.angle)*u - N.sin(self.angle)*v
                # import pdb; pdb.set_trace()
                # clvs = u_wind_levels
                extends = 'both'
                CBlabel = r'Wind Speed (ms$^{-1}$)'

            elif v is 'perpwind':
                u = self.get_wrfout_slice('U',utc=self.tidx,y=yint,x=xint)
                v = self.get_wrfout_slice('V',utc=self.tidx,y=yint,x=xint)
                # u = self.W.get('U',ps)
                # v = self.W.get('V',ps)
                # Note the negative here. I think it's alright? TODO
                data = -N.cos(self.angle)*v + N.sin(self.angle)*u
                # clvs = u_wind_levels
                extends = 'both'
                CBlabel = r'Wind Speed (ms$^{-1}$)'

            else:
                print(("Unsupported variable",v))
                raise Exception

            if nn == 0:
                kwargs = {}
                kwargs['alpha'] = 0.6
                kwargs['extend'] = 'both'
                if isinstance(clvs,N.ndarray):
                    kwargs['levels'] = clvs
                
                data = N.swapaxes(data[0,:,:],1,0)    
                cf = self.ax.contourf(grid,heighthalf,data,cmap=cmap,**kwargs)#,
            else:
                # import pdb; pdb.set_trace()       
                data = N.swapaxes(data[0,:,:],1,0)    
                ct = self.ax.contour(grid,heighthalf,data,colors=['k',],levels=contour_clvs,linewidths=0.3)
                self.ax.clabel(ct,inline=1,fontsize=6,fmt='%d')

        self.ax.fill_between(xticks,terrain_z[:,0],0,facecolor='lightgrey')
        labeldelta = 15

        self.ax.set_yticks(N.arange(self.D.plot_zmin,
                                    self.D.plot_zmax+self.D.plot_dz,self.D.plot_dz))
        self.ax.set_xlabel("Distance along cross-section (km)")
        self.ax.set_ylabel("Height above sea level (m)")
        
        datestr = utils.string_from_time('output',ttime)

        if ztop:
            self.ax.set_ylim([0,ztop*1000])
        naming = [vrbl,'xs',datestr,f_suffix]
        fname = self.create_fname(*naming)
        self.save(outpath,fname)
        #self.close()
        
        CBlabel = str(vrbl)
        
        # Save a colorbar
        # Only if one doesn't exist
        self.just_one_colorbar(outpath,fname+'CB',cf,label=CBlabel)
        
        self.draw_transect(outpath,fname+'Tsct')
        plt.close(self.fig)

    # def just_one_colorbar(self,fpath,fname,cf,label):
        # """docstring for just_one_colorbar"""
        # try:
            # with open(fpath): pass
        # except IOError:
            # self.create_colorbar(fpath,fname,cf,label=label)

    
    def draw_transect(self,outpath,fname,radar=True):
        B = BirdsEye(self.W)
        m,x,y = B.basemap_setup()
        m.drawgreatcircle(self.lonA,self.latA,self.lonB,self.latB)
        # tv = 'Q_pert'
        tv = 'cref';lv=False
        # lv = 800
        # clvs = N.arange(-0.005,0.0052,0.0002)
        # cmap='BrBG'
        S = Scales('cref',False)
        clvs = S.clvs
        cmap = S.cm
        m.contourf(x,y,self.W.get(tv,utc=self.tidx,level=lv)[0,0,:,:],levels=clvs,cmap=cmap)
        B.save(outpath,fname)
        plt.close(B.fig)
        
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
