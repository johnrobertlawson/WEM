import pdb
import matplotlib as M
M.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as N

from defaults import Defaults
from figure import Figure
import WEM.utils as utils
import scales

class BirdsEye(Figure):
    def __init__(self,config,wrfout):
        super().__init__(config,wrfout)

    def plot_data(self,data,mplcommand,fpath,pt,V=0):
        """
        Generic method that plots any matrix of data on a map

        Inputs:
        data        :   lat/lon matrix of data
        m           :   basemap instance
        mplcommand  :   contour or contourf
        fpath       :   absolute filepath including name
        V           :   scale for contours

        """
        # INITIALISE
        self.fig = plt.figure()
        self.fig = self.figsize(8,8,self.fig)     # Create a default figure size if not set by user
        self.bmap,x,y = self.basemap_setup()

        if mplcommand == 'contour':
            if not V:
                self.bmap.contour(x,y,data)
            else:
                self.bmap.contour(x,y,data,V)
        elif mplcommand == 'contourf':
            if not V:
                self.bmap.contourf(x,y,data,alpha=0.5)
            else:
                self.bmap.contourf(x,y,data,V,alpha=0.5)



        # LABELS, TITLES etc
        """
        Change these to hasattr!
        """
        #if self.C.plot_titles:
        title = utils.string_from_time('title',pt,tupleformat=0)
        plt.title(title)
        #if self.C.plot_colorbar:
        plt.colorbar(orientation='horizontal')

        # SAVE FIGURE
        datestr = utils.string_from_time('output',pt,tupleformat=0)
        self.fname = self.create_fname(fpath) # No da variable here
        self.save(self.fig,self.p2p,self.fname)
        self.fig.clf()

    def plot2D(self,va,**kwargs):
        """
        Inputs:

        va      :   variable

        keyword arguments:

        pt  :   plot time
        lv  :   level

        tla     :   top limit of latitude
        bla     :   bottom limit of latitude
        llo     :   left limit of longitude
        rlo     :   right limit of longitude

        plottype    :   contourf by default

        """
        # INITIALISE
        #en = self.W.path
        self.fig.set_size_inches() 
        sm = kwargs.get('smooth',1)
        self.bmap,x,y = self.basemap_setup(smooth=sm)

        # Unpack dictionary
        lv = kwargs['lv']
        pt = vardict['pt']
        plottype = vardict.get('plottype','contourf')
        # pdb.set_trace()

        # Get indices for time, level, lats, lons
        # TIME
        if pt == 'all':
            time_idx = slice(None,None)
        elif pt == 'range':
            start_frame = self.W.get_time_idx(vardict['itime'], tuple_format=1)
            end_frame = self.W.get_time_idx(vardict['ftime'], tuple_format=1)
            time_idx = slice(start_frame,end_frame)
        else:
            time_idx = self.W.get_time_idx(pt)

        # LAT/LON
        #if 'smooth' in vardict:
            #s = vardict['smooth']
  
        lat_sl, lon_sl = self.get_limited_domain(da,smooth=sm)

        # LEVEL
        vc = vardict['vc']
        if vc == 'surface':
            lv_idx = 0
        elif lv == 'all':
            lv_idx = 'all'
        else:
            print("Need to sort other levels")
            raise Exception

        lv_na = utils.get_level_naming(va,**vardict)

        """
           def plot_strongest_wind(self,dic):
                itime = dic['itime']
                ftime = dic['ftime']
                V = dic.get('range',(10,32.5,2.5))
                F = BirdsEye(self.C,W

        """

        # Now clear dictionary of old settings
        # They will be replaced with indices
        # vardict.pop('pt')
        # vardict.pop('lv')
        # try:
            # vardict.pop('da')
        # except KeyError:
            # pass

        # FETCH DATA
        PS = {'t': time_idx, 'lv': lv_idx, 'la': lat_sl, 'lo': lon_sl}
        data = self.W.get(va,PS,**vardict)

        la_n = data.shape[-2]
        lo_n = data.shape[-1]

        # COLORBAR, CONTOURING
        cm, clvs = scales.get_cm(va,**vardict)
        multiplier = scales.get_multiplier(va,**vardict)
        # Override contour levels if specified
        # clvs = vardict.get('range',clvs_default)

        # pdb.set_trace()
        if cm:
            plotargs = (x,y,data.reshape((la_n,lo_n)),clvs)
            cmap = cm
            #self.bmap.contourf(x,y,data.reshape((la_n,lo_n)),clvs,{'cmap':cm})
        elif isinstance(clvs,N.ndarray):
            #self.bmap.contourf(x,y,data.reshape((la_n,lo_n)),clvs,cmap=plt.cm.jet)
            if plottype == 'contourf':
                plotargs = (x,y,data.reshape((la_n,lo_n)),clvs)
                cmap = plt.cm.jet
            else:
                plotargs = (x,y,data.reshape((la_n,lo_n)),clvs)
        else:
            plotargs = (x,y,data.reshape((la_n,lo_n)))
            cmap = plt.cm.jet
            #self.bmap.contourf(x,y,data.reshape((la_n,lo_n)))


        if plottype == 'contourf':
            self.bmap.contourf(*plotargs,cmap=cmap)
        elif plottype == 'contour':
            ctplt = self.bmap.contour(*plotargs,colors='k')
            scaling_func = M.ticker.FuncFormatter(lambda x, pos:'{0:d}'.format(int(x*multiplier)))
            plt.clabel(ctplt, inline=1, fmt=scaling_func, fontsize=9, colors='k')

        # LABELS, TITLES etc
        if self.C.plot_titles:
            title = utils.string_from_time('title',pt,**vardict)
            plt.title(title)
        if plottype == 'contourf' and self.C.colorbar:
            plt.colorbar(orientation='horizontal')
        
        # SAVE FIGURE
        datestr = utils.string_from_time('output',pt)
        if not na:
            # Use default naming scheme
            na = (va,lv_na,datestr)
        else:
            # Come up with scheme...
            print("Coming soon: ability to create custom filenames")
            raise Exception
        self.fname = self.create_fname(*na) # No da variable here
        self.save(self.fig,self.p2p,self.fname)
        plt.close()

    def plot_streamlines(self,lv,pt,da=0):
        self.fig = plt.figure()
        m,x,y = self.basemap_setup()

        time_idx = self.W.get_time_idx(pt)

        if lv==2000:
            lv_idx = None
        else:
            print("Only support surface right now")
            raise Exception 

        lat_sl, lon_sl = self.get_limited_domain(da)

        slices = {'t': time_idx, 'lv': lv_idx, 'la': lat_sl, 'lo': lon_sl}

        if lv == 2000:
            u = self.W.get('U10',slices)[0,:,:]
            v = self.W.get('V10',slices)[0,:,:]
        else:
            u = self.W.get('U',slices)[0,0,:,:]
            v = self.W.get('V',slices)[0,0,:,:]
        # pdb.set_trace()
        
        #div = N.sum(N.dstack((N.gradient(u)[0],N.gradient(v)[1])),axis=2)*10**4
        #vort = (N.gradient(v)[0] - N.gradient(u)[1])*10**4
        #pdb.set_trace()
        lv_na = utils.get_level_naming('wind',lv=2000)

        m.streamplot(x[self.W.x_dim/2,:],y[:,self.W.y_dim/2],u,v,
                        density=2.5,linewidth=0.75,color='k')
        #div_Cs = N.arange(-30,31,1)
        #divp = m.contourf(x,y,vort,alpha=0.6)
        #divp = m.contour(x,y,vort)

        #plt.colorbar(divp,orientation='horizontal')
        if self.C.plot_titles:
            title = utils.string_from_time('title',pt)
            plt.title(title)
        datestr = utils.string_from_time('output',pt)
        na = ('streamlines',lv_na,datestr)
        self.fname = self.create_fname(*na) 
        self.save(self.fig,self.p2p,self.fname)
        plt.clf()
        plt.close()

    def basemap_setup(self,smooth=1):
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

        s = slice(None,None,smooth)
        x,y = m(self.W.lons[s,s],self.W.lats[s,s])
        return m, x, y


