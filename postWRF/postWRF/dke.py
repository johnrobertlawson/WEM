# This script creates plots for pulsation data


### IMPORTS
import copy
import pickle as P
import numpy as N
import pytz
import matplotlib as M
UTC = pytz.utc
import matplotlib.pyplot as plt
import datetime
import scipy.signal
import pdb
import calendar as cal
import random 
import scipy.io 
import scipy 
import scipy.stats as stats
import scipy.stats.mstats as SSM
from matplotlib.colors import cnames
from mpl_toolkits.mplot3d import Axes3D


### OPTIONS/CONSTANTS
runningmeanswitch = 0 # Use butterworth filter

### PLOTTING

### FUNCTIONS
def estimated_autocorrelation(x):
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = N.correlate(x, x, mode = 'full')[-n:]
    #assert N.allclose(r, N.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(N.arange(n, 0, -1)))
    return result

def runningmean(interval,windowsize):
    window = N.ones(int(windowsize))/float(windowsize)
    return N.convolve(interval, window, 'same')

def window(St,End,data,time):
    diffSt = abs(time-St)
    diffEnd = abs(time-End)
    nSt = N.where(diffSt==(N.min(diffSt)))[0][0]
    nEnd = N.where(diffEnd==(N.min(diffEnd)))[0][0]
    windowdata = data[nSt:nEnd]
    windowtime = time[nSt:nEnd]
    return windowdata,windowtime

def period_ticks(X):
    X2 = (1/X)/60
    return ["%.1f" %x for x in X2]

def butterworth_hp(data,samplerate, cutoff):
    nyq = samplerate * 0.5 #Nyquist frequency
    #bN,wn = scipy.signal.buttord(wp=cutoff/nyq, ws=cutoff*1.2/nyq,gpass = 0.1, gstop = 10.0)
    #bN,wn = scipy.signal.buttord(wp=0.5, ws=0.1 ,gpass = 0.1, gstop = 10.0)
    #bb,ba = scipy.signal.butter(bN, wn, btype='high')
    bb,ba = scipy.signal.butter(2, cutoff/nyq, btype='high')
    #buttered = scipy.signal.lfilter(bb,ba, interval)
    buttered = scipy.signal.filtfilt(bb,ba,data)
    return buttered, bb, ba

def confidence(Pxx,freqs,NFFT,window=mlab.window_hanning,Confidencelevel=0.95):
    # Filipe PA Fernades
    avfnc = N.asarray([1.])
    pad_to = NFFT
    if M.cbook.iterable(window):
        assert(len(window) == NFFT)
        windowVals = window
    else:
        windowVals = window(N.ones((NFFT,), x.dtype))
    edof = (1. + (1. / N.sum(avfnc ** 2) - 1.) * N.float(NFFT) /
            N.float(pad_to) * windowVals.mean())

    a1 = (1. - Confidencelevel) / 2.
    a2 = 1. - a1

    lower = Pxx * stats.chi2.ppf(a1, 2 * edof) / stats.chi2.ppf(0.50, 2 * edof)
    upper = Pxx * stats.chi2.ppf(a2, 2 * edof) / stats.chi2.ppf(0.50, 2 * edof)

    cl = N.c_[upper, lower]
    return cl

### DATA LOADING
with f as open(fname,'r'):
    data = load(f,'r')

#OR...

# DATA
nc1 = self.W1.nc
nc2 = self.W2.nc

### PLOTS
for s in D.keys():
    for v in vars:
        fig = plt.figure(figsize=(13,8))
        ax = plt.subplot(111)

        # Matplotlib float format for dates
        hrs = M.dates.HourLocator(tz=UTC)
    
        # Have to chop off last piece of data thanks to a weird time measurement
        t = D[s]['pytime'].astype('int64')[0:-2]
        dts = [datetime.datetime.fromtimestamp(tt,tz=UTC) for tt in t]
        fds = M.dates.date2num(dts)
        fmt = '%H'
        hfmt = M.dates.DateFormatter(fmt,tz=UTC)
        
        ax.plot(fds,D[s][v][0:-2])
        
        # Ticks
        ax.xaxis.set_major_locator(hrs)
        ax.xaxis.set_major_formatter(hfmt)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        
        # Labels
        plt.xlabel('Hours (UTC)',fontsize=20)
        plt.ylabel('Wind speed (m/s)',fontsize=20)
        pname = '_'.join(('meteogram',v,s,'.png'))
        plt.savefig(OUTDIR+pname)
        plt.clf()
"""
g, gt = window(gSt,gEnd,glovers['wsms'][:-2],glovers['pytime'][:-2])
c1,c1t = window(c1St,c1End,church['wsms'][:-2],church['pytime'][:-2])
c2,c2t = window(c2St,c2End,church['wsms'][:-2],church['pytime'][:-2])
tg, tgt = window(gSt,gEnd,glovers['tmpc'][:-2],glovers['pytime'][:-2])
tc1,tc1t = window(c1St,c1End,church['tmpc'][:-2],church['pytime'][:-2])
tc2,tc2t = window(c2St,c2End,church['tmpc'][:-2],church['pytime'][:-2])
ag, agt = window(gSt,gEnd,glovers['alti'][:-2],glovers['pytime'][:-2])
ac1,ac1t = window(c1St,c1End,church['alti'][:-2],church['pytime'][:-2])
ac2,ac2t = window(c2St,c2End,church['alti'][:-2],church['pytime'][:-2])

# Export this to matlab for finding regress 
gst = N.vstack((g,gt))
scipy.io.savemat('gdata.mat', mdict={'g':gst})

c1st = N.vstack((c1,c1t))
scipy.io.savemat('c1data.mat', mdict={'c1':c1st})

c2st = N.vstack((c2,c2t))
scipy.io.savemat('c2data.mat', mdict={'c2':c2st})

"""

pxx = [] #Power
freqs = [] # Frequencies, in Hz
#cl = [] # Confidence limit
#sflist = [] # Red noise generator
#noisef = [] # noise freqs
#iter = (g,c1,c2)
#iter = (g,tg,ag)
#times = (gt,tgt,agt)
#nm = ('glovers','churchearly','churchlate')
#nm = 'glovers'*3
#siglist = (3.9371,1.9277,2.3655)

if allplots == 1:

    runningmeanmatrix = []
    #for it,t,n,sige in zip(iter,times,nm,siglist):
    for s in D.keys(): # Iterate over station
        for w in wvars: # iterate over windowed variables
            # Running mean for glovers due to non-linear detrending
            if s == 'glovers':
                if runningmeanswitch == 1:
                    ym = runningmean(D[s][w]['data'],10)
                    y = (D[s][w]['data']-ym)[10:-10]
                    x = D[s][w]['times'][10:-10]
                    D[s][w]['rmm'] = D[s][w]['data']-ym
                else:
                    cutoff =  2.77777e-4 # 60 minutes in Hz
                    samplerate = 1/60.0 # Every 60 seconds
                    #samplerate = 1 # Every 60 seconds
                    y, bb, ba = butterworth_hp(D[s][w]['data'],samplerate,cutoff)
                    x = D[s][w]['times']
                    # Calculate response
                    (ww,hh) = scipy.signal.freqz(bb,ba)#,x)
                    ww = ww/(2*N.pi) / 60 # Get into minutes
            else:
                y = scipy.signal.detrend(D[s][w]['data'])
                x = D[s][w]['times']
            plt.figure()
            plt.plot(x,y)
            plt.savefig(OUTDIR+'_'.join(('detrended',s,w)))
            plt.clf()

            plt.figure()
            ax1 = plt.subplot(111)
            ax1.grid(True)
            
            pmin, pmax = (-5,-2)
            xmin, xmax = (10**pmin, 10**pmax)

            ax2 = ax1.twiny()
            X2vals = N.array(N.logspace(pmin,pmax,12))
            X2ticks = N.array(N.linspace(0,1,12)) 

            
            ax1.semilogx(ww,N.abs(hh))
            ax1.set_xlim(xmin,xmax)  

            ax2.set_xticks(X2ticks) 
            ax2.set_xticklabels(period_ticks(X2vals),color='red')
            ax2.set_xlabel('Period (min)') 
            ax2.grid(True,'major',color='red')
            plt.savefig(OUTDIR+'butterworthfreqresponse.png',bbox='tight')
            plt.clf()
            
            # Estimate autocorrelation and phi
            result = estimated_autocorrelation(y)
            phi = result[1]
            
            # Set Hanning window length and observation rate
            if s=='glovers':
                wlen = 64
                #fs = (1/60.)/60 # Every minute
                fs = 1/60. # Every minute
            else: 
                wlen = 32
                fs = 1/300. # Every 5 minutes
                #fs = (1/300.)/60 # Every 5 minutes
            
            # FFT with optimal Welch overlap (half the window length)
            out1, out2 = plt.psd(y,NFFT=wlen,Fs=fs,detrend=mlab.detrend,window=mlab.window_hanning,scale_by_freq=True,noverlap=wlen/2)
            #out1, out2, out3 = TS.psd_ci(it,NFFT=NFFT,Fs=FS,detrend=mlab.detrend,window=mlab.window_hanning,scale_by_freq=True,noverlap=NFFT/2, Confidencelevel=0.05)
            D[s][w]['pxx'] = out1
            D[s][w]['freqs'] = out2
            #cl.append(out3)
            
            #plt.savefig(OUTDIR+'PSD_'+n+'.png')
            # Calculate red noise
            #phi = result[1]
            #print sige
            #sige = 0 
            #Num = len(it)
            #y = N.zeros([Num])
            #y[0] = random.random() * sige
            #for i in range(1,Num):
            #    y[i] = (phi*y[i-1]) + random.random()*sige
            #y = scipy.signal.detrend(y)
            #P1, fp1, dummy1 = TS.psd_ci(y,NFFT=NFFT,window=mlab.window_hanning,Fs=FS)
            
            # Calculate red noise powers
            sf = (1-(phi**2))/((1+(phi**2)) - 2*phi*N.cos(N.linspace(0,N.pi,len(out2))))
            sf = sf*(N.sum(out1)/N.sum(sf))
            D[s][w]['rnoise'] = sf
            cl = confidence(sf,out2,wlen)
            D[s][w]['cl'] = cl
           #mdatan = N.vstack((fp1,sf))
            #mn = str(n)+'noise'
            #scipy.io.savemat(mn+'.mat',mdict={mn : mdatan})
            
            #mdatar = N.vstack((out1,out2))
            #mr = str(n) + 'real'
            #scipy.io.savemat(mr+'.mat',mdict={mr : mdatar})
            fig = plt.figure(figsize=(13,8))
            ax1 = plt.subplot(111)
            ax1.grid(True)
            
            #if s=='glovers':
            #    pmin, pmax = (-4,-1)
            #else:
            pmin, pmax = (-4,-2)
            xmin, xmax = (10**pmin, 10**pmax)
            # Add period x-axis too
            ax2 = ax1.twiny()
            X2vals = N.array(N.logspace(pmin,pmax,12))
            X2ticks = N.array(N.linspace(0,1,12))
            
            ax1.loglog(out2,out1,'k')
            ax1.loglog(out2,sf,'k--')
            ax1.set_xlim(xmin,xmax)
             
            ax2.set_xticks(X2ticks)
            ax2.set_xticklabels(period_ticks(X2vals),color='red')
            ax2.set_xlabel('Period (min)')
            
            ax2.grid(True,'major',color='red')
            
            #plt.title(''.join(('Power spectral density for',s,w)))
            pname = '_'.join(('PSD',s,w))
            plt.savefig(OUTDIR+pname)   
            plt.clf()
            
# Now we have pxx and freqs, ready to plot all three together for the poster - per variable
D['glovers']['label'] = ('Glovers Lane','r')
D['church1']['label'] = ('Church Butte: early','g')
D['church2']['label'] = ('Church Butte: late', 'b')
iter = range(0,3)

if allplots == 1:
    for w,n in zip(wvars,varnames):
        fig = plt.figure(figsize=(13,8))
        ax1 = plt.subplot(111)
        ax1.grid(True)
        
        pmin,pmax = (-4,-2)
        xmin, xmax = (10**pmin, 10**pmax)

        # Add period x-axis too
        ax2 = ax1.twiny()
        X2vals = N.array(N.logspace(pmin,pmax,12))
        X2ticks = N.array(N.linspace(0,1,12))
        
        labellist = [] # Legend creation from dictionary
        for i,s in enumerate(D.keys()): # Plot all three stations at once
            ax1.loglog(D[s][w]['freqs'],D[s][w]['pxx'],D[s]['label'][1])
            labellist.append(D[s]['label'][0])
        ax1.legend(labellist)
        for i,s in enumerate(D.keys()): # Plot all three stations at once
            ax1.loglog(D[s][w]['freqs'],D[s][w]['rnoise'],D[s]['label'][1]+'--')
        
        ax1.set_xlim(xmin,xmax)
        ax2.set_xticks(X2ticks)
        ax2.set_xticklabels(period_ticks(X2vals),color='red')
        ax2.set_xlabel('Period (min)')
        ax2.grid(True,'major',color='red')

        pname = '_'.join(('PSDcombined',n,'.png'))
        plt.savefig(OUTDIR+pname)    
        plt.clf()

# Output mat files for harmonic analysis
if runningmeanswitch == 1:
    gwind = N.vstack(([D['glovers']['wwind'][x] for x in ('data','rmm','times')]))
    galti = N.vstack(([D['glovers']['walti'][x] for x in ('data','rmm')]))
    gtemp = N.vstack(([D['glovers']['wtemp'][x] for x in ('data','rmm')]))
    scipy.io.savemat('gloversmatrix.mat', mdict={'gwind':gwind, 'galti':galti,'gtemp':gtemp})


# Combine all for Glovers Lane
pmin, pmax = (-4,-2)
cutoff =  2.77777e-4 # 60 minutes in Hz
samplerate = 1/60.0 # Every 60 seconds
xmin, xmax = (10**pmin, 10**pmax)
wlen = 64
X2vals = N.array(N.logspace(pmin,pmax,12))
X2ticks = N.array(N.linspace(0,1,12))


# Just Glovers Lane for publication.
s = 'glovers'
fig = plt.figure(figsize=(13,8))
ax1 = plt.subplot(111)
ax1.grid(True)
ax2 = ax1.twiny()
labellist = []
handlist = []

NAMES = {'walti':'Pressure','wtemp':'Temperature','wwind':'Wind Speed'}
for w in wvars: 
    psp = ax1.loglog(D[s][w]['freqs'],D[s][w]['pxx'])
    handlist.append(psp)
    labellist.append(NAMES[w])
    ax1.loglog(D[s][w]['freqs'],D[s][w]['rnoise'],'k--')
    ax1.loglog(D[s][w]['freqs'],D[s][w]['cl'][:,0],'k--')
    #ax1.loglog(D[s][w]['freqs'],D[s][w]['cl'][:,1],'k--')

ax2.set_xticks(X2ticks)
ax2.set_xticklabels(period_ticks(X2vals),color='red')
ax2.set_xlabel('Period (min)')
ax1.set_xlabel('Period (Hz)')

ax2.grid(True,'major',color='red')

#ax1.set_xlim(xmin,xmax)
#ax2.set_xlim(72,1.7)

ax1.set_ylabel('Power (dB/Hz)')
ax1.legend(handlist,labellist,loc=3)
plt.savefig(OUTDIR+'GloversLanePSD.png',bbox_inches='tight',pad_inches=0.4)   
plt.clf()

"""
# Attractor for Glovers Lane?

#names = ('g','c1','c2')
### 3D attractor
#for n in names:
#inp = input('which station?')
#for n in inp:
n = 'g'
wind = eval(n) 
temp = eval('t'+n)
alti = eval('a'+n)
# Z score
w,t,a = ([scipy.signal.detrend(i) for i in (wind,temp,alti)])
# For Glovers...
if n == 'g':
    l = 10
    wrm = runningmean(wind,l)
    w = (wind-wrm)
else:
    l = 1
zw = SSM.zscore(w)
zt = SSM.zscore(t)
za = SSM.zscore(a)
# Plot trajectories
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(zw[l:-l],zt[l:-l],za[l:-l])
ax.legend()
plt.show()
plt.savefig(OUTDIR+'3dphase.png')
"""
