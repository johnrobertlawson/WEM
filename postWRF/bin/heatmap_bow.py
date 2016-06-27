

import os
import pdb
import sys
import matplotlib as M
M.use('agg')
import matplotlib.pyplot as plt
import numpy as N
import datetime
import pickle as pickle

# import WEM.lazyWRF.lazyWRF as lazyWRF
from WEM.postWRF.postWRF import WRFEnviron
import WEM.utils as utils
import WEM.utils.getdata as getdata
from WEM.postWRF.postWRF.ruc import RUC
#from WEM.postWRF.postWRF.rucplot import RUCPlot

RUCdir = '/chinook2/jrlawson/bowecho/RUCclimo/heatmap/'
outdir = '/home/jrlawson/public_html/bowecho/heatmaps'

p = WRFEnviron()

def download_RUC(utc,fpath):
    print(('Downloading {0} RUC file.'.format(utc)))
    utils.getruc(utc,ncpath=fpath,convert2nc=True,duplicate=False)

fpath = '/home/jrlawson/pythoncode/bowecho/snively.csv'
names = ('casedate','caseno','casetime','casestate','bowdate',
        'bowlat','bowlon','bowtime','bowstate','score',
        'initlat','initlon','inittime','type','comments')
formats = ['S16',]*len(names)
cases = N.loadtxt(fpath,dtype={'names':names,'formats':formats},skiprows=1,delimiter=',')

DATA = {'INIT':{},'BOW':{}}
# DATA = {'INIT':{},}

# Compute/create pickle file?
compute = 1
# Manipulate and plot data?
plot = 1


lytest = 1

# Set size of box to get data over
# boxwidth = 7
if lytest:
    boxwidth = 15
else:
    boxwidth = 15

if boxwidth%2 == 0:
    raise Exception("Box width needs to be odd number")
else:
    boxmax = int((boxwidth-1)/2)
    boxmin = -boxmax

# Fetch values of variables from 0400 UTC Day 1 to 1400 UTC Day 2
starthr = 4
# That's 35 times.
ntimes = 35
# Key is variable, value is level(s) in tuple
if lytest:
    vrbls = {'lyapunov':(500,)}
else:
    vrbls = {'temp_advection':(700,850),'omega':(500,),'lyapunov':(300,500,),'RH':(700,850),'shear':(False,)}
    # vrbls = {'shear':(False,)}
# vrbls = {'temp_advection':(850,)}
lookup_cmap = {'temp_advection':M.cm.Reds, 'omega':M.cm.PuRd, 'lyapunov':M.cm.Blues,
                'shear':M.cm.Greys, 'RH':M.cm.YlGn}
# Rows are cases (14), columns are values (each time)

arr = N.zeros([14,ntimes])

# Create data arrays
for d in list(DATA.keys()):
    for v in list(vrbls.keys()):
        DATA[d][v] = {}
        for lv in vrbls[v]:
            DATA[d][v][lv] = N.zeros_like(arr)

ok_types = ('serial','progressive')

printme = ['{0} {1},{2}'.format(a,b,c) for a,b,c in zip(cases['casedate'],cases['initlat'],cases['initlon'])]
for o,p in enumerate(printme):
    if cases['type'][o] in ok_types:
        print(p)
# raise Exception

if compute:

    cnx = -1
    for cn, case in enumerate(cases):
        print(("Fetching data for case {0}".format(case['casedate'])))

        # Ignore ambiguous cases (leaving 14 good ones)
        if case['type'] not in ok_types:
            print(("Skipping {0} as type is {1}.".format(case['casedate'],
                                                    case['type'])))
            continue
        else:
            cnx += 1

        day1 = case['casedate']
        year = int(day1[:4])
        mth = int(day1[4:6])
        day = int(day1[6:8])
        day1_dt = datetime.datetime(year,mth,day,starthr,0,0)

        # Now progress through every hour
        times = [day1_dt + datetime.timedelta(hours=n) for n in range(ntimes)]

        # outstr = case['casedate'] + '_{0:02d}Z'.format(utc.hour)
        for tn,t in enumerate(times):
            print(("Getting data for time {0}".format(t)))
            download_RUC(t,RUCdir)

            # Load RUC data
            RUCg = getdata.RUC_fname(t)
            fname, ext = RUCg.split('.')
            RUCf = fname + '.nc'
            RUCpath = os.path.join(RUCdir,RUCf)
            try:
                R = RUC(RUCpath)
            except RuntimeError:
                print("Missing RUC data. Skipping...")
                R = None 
            else:
                pass


            for d in list(DATA.keys()):
                if R is not None:
                    if d=='INIT':
                        ptlat = float(case['initlat'])
                        ptlon = float(case['initlon'])
                    elif d=='BOW':
                        ptlat = float(case['bowlat'])
                        ptlon = float(case['bowlon'])

                    # Create square of lat/lon indices to fetch
                    # For some reason, this is the 'wrong way round' for x and y
                    yi, xi = utils.interp2point(None,ptlat,ptlon,R.lats,R.lons,xyidx=True)

                    xis = N.array([int(xi+diff) for diff in N.arange(boxmin,boxmax+1)])
                    yis = N.array([int(yi+diff) for diff in N.arange(boxmin,boxmax+1)])
                    xidx, yidx = N.meshgrid(xis,yis)

                for vrbl in vrbls:
                    lvs = DATA[d][vrbl]
                    for lv in lvs:
                        if R is None:
                            val = N.nan
                        else:
                            # nanmean because of Lyapunov vals sometimes missing
                            # import pdb; pdb.set_trace()
                            if lytest:
                                val = N.nanmax(R.get(vrbl,utc=t,level=lv)[0,0,yidx,xidx]) 
                            else:
                                val = N.nanmean(R.get(vrbl,utc=t,level=lv)[0,0,yidx,xidx])

                        DATA[d][vrbl][lv][cnx,tn] = val
                        print(("Value for {0} = {1}".format(vrbl,val)))


    # Sort array into descending order of skill
    dict_fname = os.path.join(RUCdir,'heatmap_dict_{0}box.pickle'.format(boxwidth))
    with open(dict_fname,'wb') as f:
        pickle.dump(DATA,f)

if plot:
    dict_fpath = os.path.join(RUCdir,'heatmap_dict_{0}box.pickle'.format(boxwidth))
    with open(dict_fpath,'rb') as f:
        HEAT = pickle.load(f)

    for rel_choice in list(DATA.keys()):
        for vrbl in list(vrbls.keys()):
            for lv in DATA[rel_choice][vrbl]:
                data = HEAT[rel_choice][vrbl][lv]
                # This is the list of cases (rows in array)

                # Columns are Day 1 0000 UTC to Day 2 2300 UTC
                # Create range of datetimes for these cases

                # Get initiation time for all cases

                # This array will store T-12 to T+6 (19 in total)
                relative = N.zeros([14,19])
                caselabels = []
                scores = []
                casetypes = []
                # For each case, find datetimes that fall within the range
                # CASES = {}
                sn = -1
                for case in cases:
                    if case['type'] not in ok_types:
                        continue
                    else:
                        sn += 1

                    # Add label to y-axis labels
                    casedate = case['casedate']
                    caselabels.append(casedate)
                    scores.append(float(case['score']))
                    casetypes.append(case['type'])

                    # Set centre date/time
                    if rel_choice == 'INIT':
                        time0 = case['inittime']
                    elif rel_choice == 'BOW':
                        time0 = case['bowtime']

                    # Offset day1 if time is Day 1 or Day 2
                    year = int(casedate[:4])
                    mth = int(casedate[4:6])
                    day = int(casedate[6:8])
                    hr = int(time0[:2])
                    centre_dt = datetime.datetime(year,mth,day,hr,0,0)

                    if hr>12:
                        # This is Day 1
                        date0 = centre_dt
                    else:
                        date0 = centre_dt + datetime.timedelta(hours=24)

                    # We know the first time with data is this:
                    start_dt = datetime.datetime(year,mth,day,4,0,0)

                    # Now create list of times before and after
                    # picktimes = ([date0 - datetime.timedelta(hours=n) for n in range(13)[::-1]] +
                                # [date0 + datetime.timedelta(hours=n) for n in range(1,7)])

                    # So the difference is:
                    diff_dt = date0 - start_dt
                    centre_idx = int(diff_dt.seconds/3600.0)
                    idx0 = centre_idx-12
                    idx1 = centre_idx+6
                    tidx = N.arange(idx0,idx1+1)

                    # Extract the indices from data into relative
                    relative[sn,:] = data[sn,tidx] 

                    # Deal with nans by interpolating

                    def interpolate_nans(arr1d):
                        num = -N.isnan(arr1d)
                        xp = num.nonzero()[0]
                        fp = arr1d[-N.isnan(arr1d)]
                        x = N.isnan(arr1d).nonzero()[0]
                        arr1d[N.isnan(arr1d)] = N.interp(x,xp,fp)
                        return arr1d

                    relative[sn,:] = interpolate_nans(relative[sn,:])

                # Normalise to 0 to 1 across whole climatology
                rel_norm = (relative-N.mean(relative))/(relative.max()-relative.min())

                if vrbl is 'omega':
                    rel_norm = rel_norm*-1

                # Now plot
                fig, ax = plt.subplots(dpi=500)
                
                cmap = lookup_cmap[vrbl]
                heatmap = ax.pcolor(rel_norm, cmap=cmap, alpha=0.8)
                
                # Put ticks into centre of each row/column
                ax.set_yticks(N.arange(rel_norm.shape[0]) + 0.5, minor=False)
                ax.set_xticks(N.arange(rel_norm.shape[1]) + 0.5, minor=False)
                ax.invert_yaxis()
                ax.xaxis.tick_top()

                timelabels = (['T-{0}'.format(n) for n in range(1,13)[::-1]] + 
                                ['T=0',] + ['T+{0}'.format(n) for n in range(1,7)])
                # caselabels already done.
                ylabels = ["{0} = {1:.2f}".format(c,s) for c,s in zip(caselabels,scores)]
                ax.set_xticklabels(timelabels, minor=False)
                ax.set_yticklabels(ylabels,minor=False)

                # Draw thick line to mark time of initiation
                plt.axvline(x=12,lw=2,color='k')
                plt.axvline(x=13,lw=2,color='k')

                # Make grid prettier
                ax.grid(False)

                for tk in ax.xaxis.get_major_ticks():
                    tk.tick10n = False
                    tk.tick20n = False
                for t in ax.yaxis.get_major_ticks():
                    tk.tick10n = False
                    tk.tick20n = False
                ax.set_xlim(0,19)

                ax2 = ax.twinx()
                ax2.invert_yaxis()
                ax2.set_yticks(N.arange(rel_norm.shape[0]) + 0.5, minor=False)
                ax2.set_xticks(N.arange(rel_norm.shape[1]) + 0.5, minor=False)
                ax2.set_yticklabels([c[0].upper() for c in casetypes],minor=False)
                # plt.gca().set_axis_direction(left='right')

                ax.tick_params(axis='both',which='both',bottom='off',
                            top='off',left='off',right='off')

                ax2.tick_params(axis='both',which='both',bottom='off',
                            top='off',left='off',right='off')
                ax2.set_ylim(14,0)

                outfname = 'Heatmap_{0}_{1}_{2}hPa_{3}box'.format(
                            rel_choice,vrbl,lv,boxwidth)
                if lytest:
                    outfname = outfname + '_lytest2.png'
                else:
                    outfname = outfname + '.png'
                outfpath = os.path.join(outdir,outfname)
                fig.tight_layout()
                fig.savefig(outfpath)
                print(("Saved figure to {0}".format(outfpath)))
