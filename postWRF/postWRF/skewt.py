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

# User imports

from Params import T_zero,T_base,kappa,barb_increments,P_bot,outdir,ens
from Utils import gamma_s,td,e,openWRF,getDimensions,convert_time

from WEM.utils import unix_tools
from WEM.utils import generalmet
from WEM.utils import gridded_data

class SkewT(Figure):
    def __init__(self,config):
        self.C = config

    def plot_skewT(self,wrfout,plot_time,prof_lat,prof_lon,dom,save_output):
        # OPTIONS
        # plotting
        self.W = wrfout
        height, width = (10,10)

        datestr = utils.string_from_time('output',plot_time)

        # Pickle files are saved here:
        pickledir = self.C.pickdir
        pickle_fname = '_'.join('WRFsounding',datestr+'.p')

        # Top and bottom limit of Skew T plots (Pa)
        P_b = 105000.0
        P_t = 20000.0

        # Create figure
        fig = plt.figure(figsize=(width,height))
        isotherms()
        isobars()
        dry_adiabats()
        moist_adiabats()


        t_idx = self.W.get_time_idx(plot_time,tuple_format=1)

        x,y, exact_lat, exact_lon = gridded_data.getXY(self.W.lats,self.W.lons,prof_lat,prof_lon)
        P_slices = {'t': t_idx, 'lv': slice[None,None], 'la': y, 'lo': x}
        H_slices = {'lv':0, 'la':y, 'lo':x}
        P = self.W.get('pressure',P_slices)
        elev = self.W.get('HGT',H_slices)

        thin_locs = gridded_data.thinned_barbs(P)

        ST._windbarbs(W.nc,t_idx,y,x,P,thin_locs,n=45,color='blue')
        ST._temperature(nc,t_idx,y,x,P,linestyle='solid',color='blue')
        ST._dewpoint(nc,t_idx,y,x,P,linestyle='dashed',color='blue')

        plt.xticks(xticks,['' if tick%10!=0 else str(tick) for tick in xticks])
        plt.yticks(yticks,ytix)

        plt.axis([-20,50,P_b,P_t])
        plt.xlabel(r'Temperature ($^{\circ}$C) at 1000\,hPa')
        xticks = N.arange(-20,51,5)
        plt.xticks(xticks,['' if tick%10!=0 else str(tick) for tick in xticks])
        plt.ylabel('Pressure (hPa)')
        yticks = N.arange(P_bot,P_t-1,-10**4)
        ytix = ["%4u" %(p/100.0) for p in yticks]
        plt.yticks(yticks,ytix)
        #yticks = N.arange(P_bot,P_t-1,-10**4)
        #plt.yticks(yticks,yticks/100)

        plt.savefig(outdirfull+'verifSkewT_'+ens+'_'+nicetime+'.png',bbox_inches='tight',pad_inches=0.1)
        plt.clf()

        # For saving Skew T data
        if save_output:
            u,v = ST.return_data('wind',nc,time,y,x,thin_locs)
            T = ST.return_data('temp',nc,time,y,x,thin_locs,P=P)
            Td = ST.return_data('dwpt',nc,time,y,x,thin_locs,P=P)
            
            data_dict = {'u':u,'v':v,'T':T,'Td':Td,'P':P}
            with open(picklef,'wb') as p:
                pickle.dump(data_dict,p)

    def skewTPlot(nest,timetuple):   
        """
        OLD VERSION!!!!
         This is the method to use from the outside
         
         nest: The nesting level of the nc-file from WRF 
         time: The time for which to plot
        """  
        nc = openWRF(nest)
        time = convert_time(nest,timetuple)
        _Nx,_Ny,_Nz,_longs,_lats,_dx,_dy,x,y = getDimensions(nc)

        plt.figure()
        _isotherms()
        _isobars()
        _dry_adiabats()
        _moist_adiabats()

        P = nc.variables['P'][time,:,y,x] + nc.variables['PB'][time,:,y,x] 
        
        _windbarbs(nc,time,y,x,P)
        _temperature(nc,time,y,x,P)
        _dewpoint(nc,time,y,x,P)
        
        plt.axis([-40,50,P_b,P_t])
        plt.xlabel('Temperature ($^{\circ}\! C$) at 1000hPa')
        xticks = np.arange(-40,51,5)
        plt.xticks(xticks,['' if tick%10!=0 else str(tick) for tick in xticks])
        plt.ylabel('Pressure (hPa)')
        yticks = np.arange(P_bot,P_t-1,-10**4)
        plt.yticks(yticks,yticks/100)

        sfcT = nc.variables['T2'][time,y,x]-T_zero
        sfcP = nc.variables['PSFC'][time,y,x]
        sfcW = nc.variables['Q2'][time,y,x]
        sfcTd = td(e(sfcW,sfcP))
        plt.suptitle('Drybulb: %4.1f$^{\circ}\! C$  Dewpoint: %3.1f$^{\circ}\! C$  Pressure: %5.1f hPa' % (sfcT,sfcTd,0.01*sfcP), \
                     fontsize=10, x = 0.5, y = 0.03)        

        #plt.show()
        plt.savefig(outdir+naming+str(nest)+'_skewT.png')
        plt.title('Skew-T plot for (location)')
        plt.close()

    def _skewnessTerm(P):
        return skewness * np.log(P_bot/P)



    def _windbarbs(nc,time,y,x,P,thin_locs,n=45.0,color='black'):
        uwind = 0.5*(nc.variables['U'][time,:,y,x]+nc.variables['U'][time,:,y,x+1])
        vwind = 0.5*(nc.variables['V'][time,:,y,x]+nc.variables['V'][time,:,y+1,x])
        zmax = len(uwind[thin_locs])
        delta = 1
        baraxis = [n for _j in range(0,zmax,delta)]
        plt.barbs(baraxis,P[thin_locs],uwind[thin_locs],vwind[thin_locs],
                 barb_increments=barb_increments, linewidth = .75,color=color)

    def _windbarbs_real(uwind,vwind,P,delta=3,color='red',n=37.5):
        # Is wind in kt or m/s?   .... uwind*
        those = N.where(uwind==-9999) # Find nonsense values
        uwind = N.delete(uwind,those)
        vwind = N.delete(vwind,those)
        P = N.delete(P,those)
        zmax = len(uwind)
        # n is x-ax position on skewT for barbs.
        baraxis = [n for _j in range(0,zmax,delta)]
        plt.barbs(baraxis,P[0:zmax:delta],uwind[0:zmax:delta],vwind[0:zmax:delta], 
        barb_increments=barb_increments, linewidth = .75, barbcolor = color, flagcolor = color)

    def _temperature(nc,time,y,x,P,linestyle='solid',color='black'):
        theta = nc.variables['T'][time,:,y,x] + T_base 
        T = theta*(P/P_bot)**kappa - T_zero # Temperatur i halvflatene (C)
        plt.semilogy(T + _skewnessTerm(P), P, basey=math.e, color = color, \
                     linestyle=linestyle, linewidth = 1.5)

    def _temperature_real(T,P,color='red',linestyle='dashed'):
        plt.semilogy(T + _skewnessTerm(P), P, basey=math.e, color = color, \
                     linestyle=linestyle, linewidth = 1.5)

    def _dewpoint(nc,time,y,x,P,linestyle='dashed',color='black'):
        w = nc.variables['QVAPOR'][time,:,y,x]
        plt.semilogy(td(e(w,P)) + _skewnessTerm(P), P, basey=math.e, color = color, \
                     linestyle=linestyle, linewidth = 1.5)

    def _dewpoint_real(td,P,color='red',linestyle='dashed'):
        plt.semilogy(td + _skewnessTerm(P), P, basey=math.e, color = color, \
                     linestyle=linestyle, linewidth = 1.5)

    def return_data(whichdata,nc,time,y,x,thin_locs,P=None):
        if whichdata == 'wind':
            uwind = 0.5*(nc.variables['U'][time,:,y,x]+nc.variables['U'][time,:,y,x+1])
            vwind = 0.5*(nc.variables['V'][time,:,y,x]+nc.variables['V'][time,:,y+1,x])
            return uwind[thin_locs],vwind[thin_locs]
        elif whichdata == 'temp':
            theta = nc.variables['T'][time,:,y,x] + T_base 
            T = theta*(P/P_bot)**kappa - T_zero
            return T
        elif whichdata == 'dwpt':
            w = nc.variables['QVAPOR'][time,:,y,x]
            Td = td(e(w,P))
            return Td
        else:
            print 'Use valid variable.'

    def gettime():
        t = convert_time(dom,timetuple)
        return t

    def _isotherms():
        for temp in np.arange(-140,50,10):
            plt.semilogy(temp + _skewnessTerm(plevs), plevs,  basey=math.e, \
                         color = ('blue' if temp <= 0 else 'red'), \
                         linestyle=('solid' if temp == 0 else 'dashed'), linewidth = .5)

    def _isobars():
        for n in np.arange(P_bot,P_t-1,-10**4):
            plt.plot([-40,50], [n,n], color = 'black', linewidth = .5)
            
    def _dry_adiabats():
        for tk in T_zero+np.arange(-30,210,10):
            dry_adiabat = tk * (plevs/P_bot)**kappa - T_zero + _skewnessTerm(plevs)
            plt.semilogy(dry_adiabat, plevs, basey=math.e, color = 'brown', \
                         linestyle='dashed', linewidth = .5)

    def _moist_adiabats():
        ps = [p for p in plevs if p<=P_bot]
        for temp in np.concatenate((np.arange(-40.,10.1,5.),np.arange(12.5,45.1,2.5))):
            moist_adiabat = []
            for p in ps:
                temp -= dp*gamma_s(temp,p)
                moist_adiabat.append(temp + _skewnessTerm(p))
            plt.semilogy(moist_adiabat, ps, basey=math.e, color = 'green', \
                         linestyle = 'dotted', linewidth = .5)

 

    """
     Plots skewT-lnP-diagram from WRF-output file.
    @author Geir Arne Waagbø
    @see http://code.google.com/p/pywrfplot/
     
     Formulas taken from Rogers&Yau: A short course in cloud physics (Third edition)
     Some inspiration from:
     http://www.atmos.washington.edu/~lmadaus/pyscript/plot_wrf_skewt.txt

    from Params import T_zero,T_base,kappa,barb_increments,P_bot, naming, outdir
    from Utils import gamma_s,td,e,openWRF,getDimensions,convert_time

    skewness = 37.5
    # Defines the ranges of the plot, do not confuse with P_bot and P_top
    P_b = 105000.
    P_t = 10000. 
    dp = 100.
    plevs = np.arange(P_b,P_t-1,-dp)
    """

