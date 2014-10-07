# Run this script with and without uintah by changing pywrfplotParams variable 'uintah'.

### IMPORTS
import numpy as N
import math
import matplotlib as M
M.use('Agg')
import matplotlib.pyplot as plt
import pdb
import cPickle as pickle
import sys
import os

# User imports

#from Params import mc.Tz,mc.Tb,mc.kappa,barb_increments,self.P_bot,outdir,ens
#from Utils import gamma_s,td,e,openWRF,getDimensions,convert_time

from figure import Figure
from wrfout import WRFOut
import WEM.utils as utils
import metconstants as mc

class Profile(Figure):
    def __init__(self,config,wrfout=0):
        # self.C = config
        # self.path_to_WRF = self.C.wrfout_root
        # self.path_to_output = self.C.output_root
        # if wrfout:
            # self.W = wrfout
        super(Profile,self).__init__(config,wrfout,ax=0,fig=0)

    def composite_profile(self,va,plot_time,plot_latlon,wrfouts,outpath,
                            dom=1,mean=1,std=1,xlim=0,ylim=0,fig=0,ax=0,
                            locname=0,ml=-2):
        """
        Loop over wrfout files.
        Get profile of variable
        Plot all members

        Optional standard deviation
        Optional mean

        If ax, save image to that axis

        ml     :   member level. negative number that corresponds to the 
                            folder in absolute string for naming purposes.
        """

        # Set up figure
        if isinstance(fig,M.figure.Figure):
            self.fig = fig
            self.ax = ax
        else:
            self.fig, self.ax = plt.subplots()
            
        # Plot settings
        if xlim:
            xmin, xmax, xint = xlim
        if ylim:
            if ylim[1] > ylim[0]:
                # top to bottom
                P_top, P_bot, dp = [y*100.0 for y in ylim]
            else:
                P_bot, P_top, dp = [y*100.0 for y in ylim]
        else:
            P_bot = 100000.0
            P_top = 20000.0
            dp = 10000.0

        # plevs = N.arange(P_bot,P_top,dp)

        # Get wrfout prototype for information
        W = WRFOut(wrfouts[0])

        lat, lon = plot_latlon
        datestr = utils.string_from_time('output',plot_time)
        t_idx = W.get_time_idx(plot_time,)
        y, x, exact_lat, exact_lon = utils.getXY(W.lats1D,W.lons1D,lat,lon)
        slices = {'t': t_idx, 'la': y, 'lo': x}
        #var_slices = {'t': t_idx, 'lv':0, 'la':y, 'lo':x}

        # Initialise array
        # Number of levels and ensembles:
        nPlevs = W.z_dim
        data = W.get(va,slices)
        nvarlevs = data.shape[1]
        nens = len(wrfouts)

        # 2D: (profile,member)
        profile_arr = N.zeros((nvarlevs,nens))
        composite_P = N.zeros((nPlevs,nens))

        # Set up legend
        labels = []
        colourlist = utils.generate_colours(M,nens)
        # M.rcParams['axes.color_cycle'] = colourlist

        # Collect profiles
        for n,wrfout in enumerate(wrfouts):
            W = WRFOut(wrfout)

            # Get pressure levels
            composite_P[:,n] = W.get('pressure',slices)[0,:,0,0]
            #elev = self.W.get('HGT',H_slices)

            #pdb.set_trace()
            # Grab variable
            profile_arr[:,n] = W.get(va,slices)[0,:,0,0]

            # Plot variable on graph
            self.ax.plot(profile_arr[:,n],composite_P[:,n],color=colourlist[n])
           
            member = wrfout.split('/')[ml]
            labels.append(member)
            # if locname=='KOAX': pdb.set_trace()

        # Compute mean, std etc
        if mean:
            profile_mean = N.mean(profile_arr,axis=1)
            profile_mean_P = N.mean(composite_P,axis=1)
            self.ax.plot(profile_mean,profile_mean_P,color='black')
        if std:
            # Assume mean P across ensemble is correct level
            # to plot standard deviation
            profile_std = N.std(profile_arr,axis=1)
            std_upper = profile_mean + profile_std
            std_lower = profile_mean - profile_std
            self.ax.plot(std_upper,profile_mean_P,'k--')
            self.ax.plot(std_lower,profile_mean_P,'k--')

        if not locname:
            fname = '_'.join(('profile_comp',va,datestr,'{0:03d}'.format(x),'{0:03d}'.format(y))) + '.png'
        else:
            fname = '_'.join(('profile_comp',va,datestr,locname)) + '.png'

        # Set semi-log graph
        self.ax.set_yscale('log')
        # Plot limits, ticks
        yticks = N.arange(P_bot,P_top+dp*100.0,-100*100.0)
        # yticks = N.arange(P_bot,P_top+dp,-dp*100)
        # self.ax.set_yticks(yticks)
        ylabels = ["%4u" %(p/100.0) for p in yticks]
        self.ax.set_yticks(yticks)
        self.ax.set_yticklabels(ylabels)
        # import pdb; pdb.set_trace()
        # self.ax.yaxis.tick_right()

        #plt.axis([-20,50,105000.0,20000.0])
        #plt.xlabel(r'Temperature ($^{\circ}$C) at 1000 hPa')
        #plt.xticks(xticks,['' if tick%10!=0 else str(tick) for tick in xticks])
        self.ax.set_ylabel('Pressure (hPa)')
        #yticks = N.arange(self.P_bot,P_t-1,-10**4)
        #plt.yticks(yticks,yticks/100)

        if xlim:
            self.ax.set_xlim([xmin,xmax])
            xticks = N.arange(xmin,xmax+xint,xint)
            self.ax.set_xticks(xticks)

        # Flip y axis
        # Limits are already set
        self.ax.set_ylim([P_bot,P_top])

        # plt.tight_layout(self.fig)
        #plt.autoscale(enable=1,axis='x')
        #ax = plt.gca()
        #ax.relim()
        #ax.autoscale_view()
        #plt.draw()
        self.ax.legend(labels,loc=2,fontsize=6)
        self.save(outpath,fname)
        plt.close(self.fig)

class SkewT(Figure):
    def __init__(self,config,wrfout=0):
        # self.C = config
        # self.path_to_WRF = self.C.wrfout_root
        # self.path_to_output = self.C.output_root
        # if wrfout:
            # self.W = wrfout
        super(SkewT,self).__init__(config,wrfout,ax=0,fig=0)

        #self.plevs = self.W.z_dim-1


    def plot_skewT(self,plot_time,plot_latlon,dom,outpath,save_output=0,save_plot=1):
        

        # Defines the ranges of the plot, do not confuse with self.P_bot and self.P_top
        self.barb_increments = {'half': 2.5,'full':5.0,'flag':25.0}
        self.skewness = 37.5
        self.P_bot = 100000.
        self.P_top = 10000.
        self.dp = 100.
        self.plevs = N.arange(self.P_bot,self.P_top-1,-self.dp)

        # pdb.set_trace()
        prof_lat, prof_lon = plot_latlon
        datestr = utils.string_from_time('output',plot_time)
        t_idx = self.W.get_time_idx(plot_time)
        y,x, exact_lat, exact_lon = utils.getXY(self.W.lats1D,self.W.lons1D,prof_lat,prof_lon)


        # Create figure
        if save_plot:
            # height, width = (10,10)

            # fig = plt.figure(figsize=(width,height))
            self.isotherms()
            self.isobars()
            self.dry_adiabats()
            self.moist_adiabats()


            P_slices = {'t': t_idx, 'la': y, 'lo': x}
            H_slices = {'t':t_idx, 'lv':0, 'la':y, 'lo':x}
            # pdb.set_trace()
            P = self.W.get('pressure',P_slices)[0,:,0,0]

            elev = self.W.get('HGT',H_slices)

            thin_locs = utils.thinned_barbs(P)

            self.windbarbs(self.W.nc,t_idx,y,x,P,thin_locs,n=45,color='blue')
            self.temperature(self.W.nc,t_idx,y,x,P,linestyle='solid',color='blue')
            self.dewpoint(self.W.nc,t_idx,y,x,P,linestyle='dashed',color='blue')

            xticks = N.arange(-20,51,5)
            yticks = N.arange(100000.0,self.P_top-1,-10**4)
            ytix = ["%4u" %(p/100.0) for p in yticks]
            self.ax.set_xticks(xticks,['' if tick%10!=0 else str(tick) for tick in xticks])
            self.ax.set_yticks(yticks,ytix)

            self.ax.axis([-20,50,105000.0,20000.0])
            self.ax.set_xlabel(r'Temperature ($^{\circ}$C) at 1000 hPa')
            self.ax.set_xticks(xticks,['' if tick%10!=0 else str(tick) for tick in xticks])
            self.ax.set_ylabel('Pressure (hPa)')
            self.ax.set_yticks(yticks,ytix)
            #yticks = N.arange(self.P_bot,P_t-1,-10**4)
            #plt.yticks(yticks,yticks/100)

            fname = '_'.join(('skewT',datestr,'{0:03d}'.format(x),'{0:03d}'.format(y))) + '.png'
            # utils.trycreate(self.path_to_output)

            # fpath = os.path.join(self.path_to_output,fname)
            # plt.savefig(fpath)
            self.save(outpath,fname)
            plt.close()

        # For saving Skew T data
        if save_output:

            # Pickle files are saved here:
            pickle_fname = '_'.join(('WRFsounding',datestr,'{0:03d}'.format(x),
                            '{0:03d}'.format(y)+'.p'))
            pickle_path = os.path.join(self.C.pickledir,pickle_fname)
            u,v = self.return_data('wind',self.W.nc,t_idx,y,x,thin_locs)
            T = self.return_data('temp',self.W.nc,t_idx,y,x,thin_locs,P=P)
            Td = self.return_data('dwpt',self.W.nc,t_idx,y,x,thin_locs,P=P)
            
            data_dict = {'u':u,'v':v,'T':T,'Td':Td,'P':P}
            with open(pickle_path,'wb') as p:
                pickle.dump(data_dict,p)
            print("Saving data to {0}".format(pickle_path))

            return
        else:
            return

    def skewT_composite(self,):
        """
        Open pickle files from multiple soundings
        Find mean Td/T
        """
        pass

    def skewnessTerm(self,P):
        return self.skewness * N.log(self.P_bot/P)


    def windbarbs(self,nc,time,y,x,P,thin_locs,n=45.0,color='black'):
        uwind = 0.5*(nc.variables['U'][time,:,y,x]+nc.variables['U'][time,:,y,x+1])
        vwind = 0.5*(nc.variables['V'][time,:,y,x]+nc.variables['V'][time,:,y+1,x])
        zmax = len(uwind[thin_locs])
        delta = 1
        baraxis = [n for _j in range(0,zmax,delta)]
        # pdb.set_trace()
        plt.barbs(baraxis,P[thin_locs],uwind[thin_locs],vwind[thin_locs],
                 barb_increments=self.barb_increments, linewidth = .75,color=color)

    def windbarbs_real(self,uwind,vwind,P,delta=3,color='red',n=37.5):
        # Is wind in kt or m/s?   .... uwind*
        those = N.where(uwind==-9999) # Find nonsense values
        uwind = N.delete(uwind,those)
        vwind = N.delete(vwind,those)
        P = N.delete(P,those)
        zmax = len(uwind)
        # n is x-ax position on skewT for barbs.
        baraxis = [n for _j in range(0,zmax,delta)]
        plt.barbs(baraxis,P[0:zmax:delta],uwind[0:zmax:delta],vwind[0:zmax:delta],
        barb_increments=self.barb_increments, linewidth = .75, barbcolor = color, flagcolor = color)

    def temperature(self,nc,time,y,x,P,linestyle='solid',color='black'):
        theta = nc.variables['T'][time,:,y,x] + mc.Tb
        T = theta*(P/self.P_bot)**mc.kappa - mc.Tz # Temperatur i halvflatene (C)
        plt.semilogy(T + self.skewnessTerm(P), P, basey=math.e, color = color, \
                     linestyle=linestyle, linewidth = 1.5)

    def temperature_real(self,T,P,color='red',linestyle='dashed'):
        plt.semilogy(T + self.skewnessTerm(P), P, basey=math.e, color = color, \
                     linestyle=linestyle, linewidth = 1.5)

    def dewpoint(self,nc,time,y,x,P,linestyle='dashed',color='black'):
        w = nc.variables['QVAPOR'][time,:,y,x]
        plt.semilogy(self.td(self.e(w,P)) + self.skewnessTerm(P), P, basey=math.e, color = color, \
                     linestyle=linestyle, linewidth = 1.5)

    def dewpoint_real(self,td,P,color='red',linestyle='dashed'):
        plt.semilogy(td + self.skewnessTerm(P), P, basey=math.e, color = color, \
                     linestyle=linestyle, linewidth = 1.5)

    def return_data(self,whichdata,nc,time,y,x,thin_locs,P=None):
        if whichdata == 'wind':
            uwind = 0.5*(nc.variables['U'][time,:,y,x]+nc.variables['U'][time,:,y,x+1])
            vwind = 0.5*(nc.variables['V'][time,:,y,x]+nc.variables['V'][time,:,y+1,x])
            return uwind[thin_locs],vwind[thin_locs]
        elif whichdata == 'temp':
            theta = nc.variables['T'][time,:,y,x] + mc.Tb
            T = theta*(P/self.P_bot)**mc.kappa - mc.Tz
            return T
        elif whichdata == 'dwpt':
            w = nc.variables['QVAPOR'][time,:,y,x]
            Td = self.td(self.e(w,P))
            return Td
        else:
            print 'Use valid variable.'

    def gettime(self,):
        t = convert_time(dom,timetuple)
        return t

    def isotherms(self,):
        for temp in N.arange(-140,50,10):
            plt.semilogy(temp + self.skewnessTerm(self.plevs), self.plevs,  basey=math.e, \
                         color = ('blue' if temp <= 0 else 'red'), \
                         linestyle=('solid' if temp == 0 else 'dashed'), linewidth = .5)

    def isobars(self,):
        for n in N.arange(self.P_bot,self.P_top-1,-10**4):
            plt.plot([-40,50], [n,n], color = 'black', linewidth = .5)
            
    def dry_adiabats(self,):
        for tk in mc.Tz+N.arange(-30,210,10):
            dry_adiabat = tk * (self.plevs/self.P_bot)**mc.kappa - mc.Tz + self.skewnessTerm(self.plevs)
            plt.semilogy(dry_adiabat, self.plevs, basey=math.e, color = 'brown', \
                         linestyle='dashed', linewidth = .5)

    def moist_adiabats(self,):
        ps = [p for p in self.plevs if p<=self.P_bot]
        for temp in N.concatenate((N.arange(-40.,10.1,5.),N.arange(12.5,45.1,2.5))):
            moist_adiabat = []
            for p in ps:
                temp -= self.dp*self.gamma_s(temp,p)
                moist_adiabat.append(temp + self.skewnessTerm(p))
            plt.semilogy(moist_adiabat, ps, basey=math.e, color = 'green', \
                         linestyle = 'dotted', linewidth = .5)

 

    """
     Plots skewT-lN-diagram from WRF-output file.
    author Geir Arne Waagbo
    see http://code.google.com/p/pywrfplot/
     
     Formulas taken from Rogers&Yau: A short course in cloud physics (Third edition)
     Some inspiration from:
     http://www.atmos.washington.edu/~lmadaus/pyscript/plot_wrf_skewt.txt

    from Params import mc.Tz,mc.Tb,mc.kappa,barb_increments,self.P_bot, naming, outdir
    from Utils import gamma_s,td,e,openWRF,getDimensions,convert_time

    skewness = 37.5
    # Defines the ranges of the plot, do not confuse with self.P_bot and self.P_top
    P_b = 105000.
    P_t = 10000.
    self.dp = 100.
    self.plevs = N.arange(P_b,P_t-1,-self.dp)
    """

    def gamma_s(self,T,p):
        """Calculates moist adiabatic lapse rate for T (Celsius) and p (Pa)
        Note: We calculate dT/dp, not dT/dz
        See formula 3.16 in Rogers&Yau for dT/dz, but this must be combined with
        the dry adiabatic lapse rate (gamma = g/cp) and the
        inverse of the hydrostatic equation (dz/dp = -RT/pg)"""
        a = 2./7.
        b = ((mc.R/mc.Rv)*(mc.L**2))/(mc.R*mc.cp)
        c = a*(mc.L/mc.R)

        esat = self.es(T)
        wsat = (mc.R/mc.Rv)*esat/(p-esat) # Rogers&Yau 2.18
        numer = a*(T+mc.Tz) + c*wsat
        denom = p * (1 + b*wsat/((T+mc.Tz)**2))
        return numer/denom # Rogers&Yau 3.16

    def td(self,e):
        """Returns dew point temperature (C) at vapor pressure e (Pa)
        Insert Td in 2.17 in Rogers&Yau and solve for Td"""
        return 243.5 * N.log(e/611.2)/(17.67-N.log(e/611.2))

    def e(self,w,p):
        """Returns vapor pressure (Pa) at mixing ratio w (kg/kg) and pressure p (Pa)
        Formula 2.18 in Rogers&Yau"""
        return w*p/(w+(mc.R/mc.Rv))

    def es(self,T):
        """Returns saturation vapor pressure (Pascal) at temperature T (Celsius)
        Formula 2.17 in Rogers&Yau"""
        return 611.2*N.exp(17.67*T/(T+243.5))
