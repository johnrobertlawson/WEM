"""
Scripts related to plotting of observed data, such as surface observations,
radar reflectivity, severe weather reports.
"""


import scipy.ndimage
import pdb
import numpy as N
import os
import calendar
import glob
import datetime

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import pygrib

import WEM.utils as utils
from .birdseye import BirdsEye
from .gribfile import GribFile

# pygrib is optional import in StageIV class.

class Obs:
    """
    An instance represents a data set.
    """
    def __init__(self,fpath):
        self.fpath = fpath

class Radar(Obs):
    def __init__(self,utc,datapath):
        """
        Composite radar archive data from mesonet.agron.iastate.edu.

        :param datapath:        Absolute path to folder or .png file
        :type datapath:         str
        :param wldpath:         Absolute path to .wld file
        :type wldpath:          str
        :param fmt:             format of data - N0Q or N0R
        :type fmt:              str
        """
        self.utc = utc
        fname_root = self.get_radar_fname() 

        # Check for file
        # Download if not available
        fpath = os.path.join(datapath,fname_root)
        for ex in ('.png','.wld'):
            scan = glob.glob(fpath+ex)
            print("Looking in",datapath)
            print("Contents:",scan)

            if len(scan) == 0:
                url = self.get_radar_url()
                urlf = os.path.join(url,fname_root+ex)
                cmd1 = 'wget {0} -P {1}'.format(urlf,datapath)
                os.system(cmd1)

        png = fpath+'.png'
        wld = fpath+'.wld'

        self.data = scipy.ndimage.imread(png,mode='P') 
        # if len(self.data.shape) == 3:
            # self.data = self.data[:,:,0]

        self.xlen, self.ylen, = self.data.shape

        # Metadata
        f = open(wld,'r').readlines()

        # pixel size in the x-direction in map units/pixel
        self.xpixel = float(f[0])

        # rotation about y-axis
        self.roty = float(f[1])

        # rotation about x-axis
        self.rotx = float(f[2])

        # pixel size in the y-direction in map units,
        self.ypixel = float(f[3])

        # x-coordinate of the center of the upper left pixel
        self.ulx = float(f[4])

        # y-coordinate of the center of the upper left pixel
        self.uly = float(f[5])
       
        # lower right corner
        self.lrx = self.ulx + self.ylen*self.xpixel
        self.lry = self.uly + self.xlen*self.ypixel

        if self.fmt == 'n0r':
            self.clvs = N.arange(0,80,5)
        elif self.fmt == 'n0q':
            self.clvs = N.arange(0,90.0,0.5)
        # import pdb; pdb.set_trace()

        self.lats = N.linspace(self.lry,self.uly,self.xlen)[::-1]
        self.lons = N.linspace(self.ulx,self.lrx,self.ylen)

    def get_radar_fname(self):
        tt = utils.ensure_timetuple(self.utc)
       
        # Date for format change
        change = (2010,10,25,0,0,0)

        if tt[0] < 1995:
            print("Too early in database.")
            raise Exception
        elif calendar.timegm(tt) < calendar.timegm(change): 
            self.fmt = 'n0r'
        else:
            self.fmt = 'n0q'

        fname = '{0}_{1:04d}{2:02d}{3:02d}{4:02d}{5:02d}'.format(
                    self.fmt,*tt)
        return fname

    def get_radar_url(self):
        dt = utils.ensure_timetuple(self.utc)
        return ('mesonet.agron.iastate.edu/archive/data/'
                '{0:04d}/{1:02d}/{2:02d}/GIS/uscomp/'.format(*dt))

    def generate_basemap(self,fig,ax,Nlim=False,Elim=False,Slim=False,
                            Wlim=False):
        """
        Generate basemap.

        :param ax:      The axis on which to create the basemap
        :type ax:       matplotlib.basemap?
        """
        self.fig = fig
        self.ax = ax
        if not isinstance(Nlim,float):
            Nlim = self.uly
            Elim = self.lrx
            Slim = self.lry
            Wlim = self.ulx

        self.m = Basemap(projection='merc',
                    llcrnrlat=Slim,
                    llcrnrlon=Wlim,
                    urcrnrlat=Nlim,
                    urcrnrlon=Elim,
                    lat_ts=(Nlim-Slim)/2.0,
                    resolution='l',
                    ax=self.ax)
        self.m.drawcoastlines()
        self.m.drawstates()
        self.m.drawcountries()

    def get_subdomain(self,Nlim,Elim,Slim,Wlim,overwrite=False):
        """
        Return data array between bounds

        If overwrite is True, replace class data with new subdomain
        """
        data,lats,lons = utils.return_subdomain(self.data,self.lats,
                                self.lons,Nlim,Elim,Slim,Wlim)

        if overwrite:
            self.lats = lats
            self.lons = lons
            self.data = data
            return
        else:
            return data,lats,lons

    def get_dBZ(self,data):
        if data == 'self':
            data = self.data
        
        # pdb.set_trace()
        if self.fmt == 'n0q':
            dBZ = (data*0.5)-32
            # dBZ = (data*0.25)-32
        elif self.fmt == 'n0r':
            dBZ = (data*5.0)-30 
        return dBZ

    def plot_radar(self,outdir=False,fig=False,ax=False,fname=False,Nlim=False,
                    Elim=False, Slim=False,Wlim=False,cb=True,
                    drawcounties=False):
        """
        Plot radar data.
        """
        # if not fig:
            # fig, ax = plt.subplots()
        # self.generate_basemap(fig,ax,Nlim,Elim,Slim,Wlim)
        #lons, lats = self.m.makegrid(self.xlen,self.ylen)
        if isinstance(Nlim,float):
            data, lats, lons = self.get_subdomain(Nlim,Elim,Slim,Wlim)
            # x,y = self.m(lons,lats)
        else:
            data = self.data
            lats = self.lats #flip lats upside down?
            lons = self.lons
            # x,y = self.m(*N.meshgrid(lons,lats))
            
        # x,y = self.m(*N.meshgrid(lons,lats))
        # x,y = self.m(*N.meshgrid(lons,lats[::-1]))

        # Custom colorbar
        from . import colourtables as ct
        radarcmap = ct.reflect_ncdc(self.clvs)
        # radarcmap = ct.ncdc_modified_ISU(self.clvs)

        # Convert pixel levels to dBZ
        dBZ = self.get_dBZ(data)

           
        # dBZ[dBZ<0] = 0
        
        # def plot2D(self,data,fname,outdir,plottype='contourf',
                    # save=1,smooth=1,lats=False,lons=False,
                    # clvs=False,cmap=False,title=False,colorbar=True,
                    # locations=False):
        if not fname:
            tstr = utils.string_from_time('output',self.utc)
            fname = 'verif_radar_{0}.png'.format(tstr)
        F = BirdsEye(fig=fig,ax=ax)
        if cb:
            cb = 'horizontal'
        if (fig is not False) and (ax is not False):
            save = False
        else:
            save = True
        F.plot2D(dBZ,fname,outdir=outdir,lats=lats,lons=lons,
                    cmap=radarcmap,clvs=N.arange(5,90,5),
                    cb=cb,cblabel='Composite reflectivity (dBZ)',
                    drawcounties=drawcounties,save=save)
        # im = self.ax.contourf(x,y,dBZ,alpha=0.5,cmap=radarcmap,
                                # levels=N.arange(5.0,90.5,0.5))
        # outpath = os.path.join(outdir,fname)
        # self.fig.colorbar(im,ax=self.ax)
        # self.fig.savefig(outpath)

class SPCReports(Obs):
    def __init__(self,utc,datadir,wind=True,hail=True,torn=True):
        self.utc = utc
        tt = utils.ensure_timetuple(utc)
        yr = str(tt[0])[-2:]
        mth = '{0:02d}'.format(tt[1])
        day = '{0:02d}'.format(tt[2])

        threats = []
        if wind:
            threats.append('wind')
        if hail:
            threats.append('hail')
        if torn:
            threats.append('torn')

        self.reports = {}
        for threat in threats:
            fname = '{0}{1}{2}_rpts_{3}.csv'.format(
                        yr,mth,day,threat)
            # Check to see if file exists
            fpath = os.path.join(datadir,fname)
            scan = glob.glob(fpath)
            if len(scan) == 0:
                url = 'http://www.spc.noaa.gov/climo/reports/'
                cmd = 'wget {0}{1} -P {2}'.format(
                                url,fname,datadir)
                os.system(cmd)

            if threat=='wind':
                names = ('time','speed','location','county',
                        'state','lat','lon')
                formats = ('S4','S4','S4','S4',
                            'S4','f4','f4')
            elif threat=='hail':
                names = ('time','size','location','county',
                        'state','lat','lon')
                formats = ('S4','S4','S4','S4',
                            'S4','f4','f4')
            elif threat=='torn':
                names = ('time','fscale','location','county',
                        'state','lat','lon')
                formats = ('S4','S4','S4','S4',
                            'S4','f4','f4')
            self.reports[threat] = N.loadtxt(fpath,dtype={'names':names,'formats':formats},
                                        skiprows=1,delimiter=',',usecols=list(range(8)))
            #times = reports['time']
            self.threats = threats

    def report_datenum(self,timestamp):
        """
        convert timestamp to datenum format.
        """
        tt = utils.ensure_timetuple(self.utc)
        itime_dn = calendar.timegm(tt[0],tt[1],tt[2],12,0,0)

        hr = int(timestamp[:2])
        mn = int(timestamp[2:])
        
        if hr<11:
            # After midnight UTC
            hr = hr+24
        
        tdelta = ((hr-12)*60*60) + (mn*60)  
        return itime_dn + tdelta

    def plot_reports(self,fig=False,ax=False):
        plot_all = True
        for threat in threats:
            if plot_all:
                for t in self.reports[threat]['time']:
                    utc = self.report_datenum(t)

class StormReports(Obs):
    def __init__(self,fpath,):
        self.r = N.genfromtxt(fpath,dtype=None,
                names=True,delimiter=',',)#missing='',filling_values='none')
        # import pdb; pdb.set_trace()
        self.convert_times() 

    def convert_times(self,):
        LOLtimes = self.r['BEGIN_TIME']
        padtimes = []
        for t in LOLtimes:
            intt = int(t)
            padtimes.append('{0:04d}'.format(intt))

        hours = [s[:-2] for s in padtimes]
        mins = [s[-2:] for s in padtimes]
        self.datetimes = N.array([datetime.datetime.strptime(s+h+m,'%m/%d/%Y%H%M')
                        for s,h,m in zip(self.r['BEGIN_DATE'],hours,mins)])
        # import pdb; pdb.set_trace()
        # import numpy.lib.recfunctions
        # self.r = numpy.lib.recfunctions.append_fields(self.r,'datetimes',N.array(dates))

    def plot(self,reports,itime,ftime,fname,outdir,Nlim=False,
            Elim=False,Slim=False,Wlim=False,
            annotate=True,fig=False,ax=False,ss=50,color='blue'):
        reportidx = N.array([n for n,t in zip(list(range(len(self.r['EVENT_TYPE']))),self.r['EVENT_TYPE']) if reports in t])
        lateidx = N.where(self.datetimes > itime)
        earlyidx = N.where(self.datetimes < ftime)
        timeidx = N.intersect1d(earlyidx,lateidx,)#assume_unique=True)
        plotidx = N.intersect1d(reportidx,timeidx)

        from mpl_toolkits.basemap import Basemap

        if fig==False:
            fig,ax = plt.subplots(1,figsize=(6,6))
        m = Basemap(projection='merc',
                    llcrnrlat=Slim,
                    llcrnrlon=Wlim,
                    urcrnrlat=Nlim,
                    urcrnrlon=Elim,
                    lat_ts=(Nlim-Slim)/2.0,
                    resolution='i',
                    ax=ax)

        m.drawcoastlines()
        m.drawstates()
        m.drawcountries()

        m.scatter(self.r['BEGIN_LON'][plotidx],self.r['BEGIN_LAT'][plotidx],latlon=True,
                    marker='D',facecolors=color,edgecolors='black',s=ss)
        fig.tight_layout()
        plt.savefig(os.path.join(outdir,fname))

class MRMS(Obs):
    def __init__(self,fpath=False,rootdir=False,product=False,
                    utc=False):
        """A helper function to automatically generate file name
        to load needs to be written.

        Arguments:
            fpath (str, optional): file path to data. If False, other
                information must be presented to search for data (below).
            rootdir (str, optional): if fpath is False, this is required
                to search for data based on product and time
            product (str, optional): if fpath is False, this is required
                to find correct product. Pick from PRECIPRATE...
            utc (datetime.datetime, optional): if fpath is False, this
                is required to find data file for valid time.
        """
        if not fpath:
            fpath = self.generate_fpath(rootdir,product,utc)

        super().__init__(fpath)

    def generate_fpath(self,rootdir,prod,utc):
        """Return an absolute path based on product and time of data
        that is required, in a root directory.
        """
        # Fill this out
        valid_products = ('PRECIPRATE',)
        pdb.set_trace()
        if prod not in valid_products:
            raise Exception("Product name not valid.")

        fname = '{0}.{1}{2}{3}.{4}{5}{6}'.format(prod,utc.year,utc.month,
                                utc.day,utc.hour,utc.minute,utc.second)
        fpath = os.path.join(rootdir,fname)
        return fpath

class StageIV(GribFile):
    def __init__(self,dir_or_file,load_1h=True,load_6h=False,load_24h=False,
                    loadobj=False):
        """"Check for 9999s (missing) or negative. Other codes

        dir_or_file -   if directory, it scans for all files
                        if file, it loads that one file.
        """

        try:
            import pygrib
        except ImportError:
            print("You need to install pygrib for the StageIV class.")
            raise Exception
        else:
            self.pygrib = pygrib
        
        self.loadobj = loadobj

        # Determine whether to load one file or search in directory
        try:
            G = self.pygrib.open(dir_or_file)
        except OSError:
            ST4s = os.path.join(rootdir,'ST4*')
            print("Loading files in {0}".format(ST4s))
            fps = glob.glob(ST4s)
        else:
            fps = [dir_or_file,]

        self.DATA = {}
        if load_1h:
            self.DATA['01h'] = {}
        if load_6h:
            self.DATA['06h'] = {}
        if load_24h:
            self.DATA['24h'] = {}
        for fp in fps:
            # fp is the full path to the (grib?) file
            # f is the name of the file only
            f = os.path.basename(fp)
            t = self.date_from_fname(f)

            # if f.endswith('01h'):
                # d1h[t] = self.load_data(f)
            # elif f.endswith('06h'):
                # d6h[t] = self.load_data(f)
            # elif f.endswith('24h'):
                # d24h[t] = self.load_data(f)
            # else:
                # pass
            for accum_t in ('01h','06h','24h'):
                if f.endswith(accum_t) and (accum_t in self.DATA.keys()):
                    answer = self.load_data(fp,loadobj)
                    if answer is not False:
                        self.DATA[accum_t][t] = answer
                else:
                    pass

        # Assign all projection stats
        # pdb.set_trace()
        # print("All files in ",ST4s)
        # for fp in fps:
            # print(fp)
        # print("-"*10)

        self.projection()

    def get(self,utc,accum_hr='01h'):
        """
        Get a given time, in similar manner to WRFOut.

        Wrapper for return_array with reshape to 4D
        """
        data2D = self.return_array(utc,accum_hr=accum_hr)
        data4D = data2D[N.newaxis,N.newaxis,:,:]
        return data4D

    def date_from_fname(self,f):
        _1, d, _2 = f.split('.')
        fmt = '%Y%m%d%H'
        utc = datetime.datetime.strptime(d,fmt)
        return utc

    def load_data(self,f,loadobj=False):
        if loadobj:
            # Return Pygrib object if valid grib file
            try:
                G = self.pygrib.open(f)
            except OSError:
                return False
            else:
                return G
        else:
            # Return file path if valid grib file
            try: 
                G_test = self.pygrib.open(f)
            except OSError:
                return False
            else:
                return f

    def load_gribpart(self,G):
        if G is None:
            G = self.arbitrary_pick()
        if isinstance(G,str):
            G = self.load_data(G,loadobj=True)
        G.seek(0)
        gg = G.select(name='Total Precipitation')[0]
        return gg

    def load_accum(self,G):
        gg = self.load_gribpart(G)
        arr = gg.values
        return arr

    def return_latlon(self,G):
        if G is None:
            G = self.arbitrary_pick()
        gg = self.load_gribpart(G)
        latlon = gg.latlons()
        lats, lons = latlon
        return lats,lons

    def return_array(self,utc,accum_hr='01h'):
        G = self.DATA[accum_hr][utc]
        return self.load_accum(G)

    def return_point(self,utc,lat,lon,accum_hr='01h'):
        lats, lons = self.return_latlon(None)
        latidx,lonidx = utils.get_latlon_idx(lats,lons,lat,lon)
        arr = self.return_array(utc,accum_hr=accum_hr)
        # pdb.set_trace()
        return arr[latidx,lonidx]


    def projection(self):
        self.m = Basemap(projection='npstere',lon_0=-105.0,#lat_1=60.0,
                # llcrnrlon=lllon,llcrnrlat=lllat,urcrnrlon=urlon,urcrnrlat=urlat,
                            boundinglat=24.701632)
        G = self.arbitrary_pick()
        self.lats, self.lons = self.return_latlon(G)
        self.xx, self.yy = self.m(self.lons,self.lats)
        # pdb.set_trace()
        # self.mx, self.my = N.meshgrid(self.xx,self.yy)

        # lllon = -119.023 
        self.lllon = self.lons[0,0]
        # lllat = 23.117 
        self.lllat = self.lats[0,0]
        # urlon = -59.9044 
        self.urlon = self.lons[-1,-1]
        # urlat = 45.6147234 
        self.urlat = self.lats[-1,-1]

        self.shape = self.lats.shape
        assert self.lats.shape == self.lons.shape

    def arbitrary_pick(self):
        # pdb.set_trace()
        accum = list(self.DATA.keys())[0]
        utc = list(self.DATA[accum].keys())[0]
        return self.DATA[accum][utc]

