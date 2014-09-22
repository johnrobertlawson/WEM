import os
import pdb
import sys
import matplotlib as M
M.use('gtkagg')
import matplotlib.pyplot as plt
import numpy as N

sys.path.append('/home/jrlawson/gitprojects/')

from WEM.postWRF import WRFEnviron
from settings import Settings
import WEM.utils as utils
#from WEM.postWRF.postWRF.rucplot import RUCPlot

config = Settings()
p = WRFEnviron(config)

skewT = 0
plot2D = 0
streamlines = 0
rucplot = 0
coldpoolstrength = 0
spaghetti = 0
std = 0
profiles = 0
frontogenesis = 1
upperlevel = 0

# enstype = 'STCH'
# enstype = 'ICBC'
enstype = 'MXMP'

#case = '20060526'
case = '2006052612'
#case = '20090910'
# case = '20110419'
# case = '20130815'

# IC = 'GEFSR2'
IC = 'NAM'
# IC = 'RUC'


#ensnames = ['anl']
#experiment = 'VERIF'


if enstype == 'STCH':
    experiments = ['s'+"%02d" %n for n in range(1,11)]
    ensnames = ['p09',]
    MP = 'ICBC'
elif enstype == 'MXMP':
    # experiments = ['WSM6_Hail','Kessler','Ferrier',
    experiments = ['WSM6_Grau','WSM6_Hail','Kessler','Ferrier',
                    'WSM5','WDM5','Lin','WDM6_Grau','WDM6_Hail',
                    'Morrison_Grau','Morrison_Hail']
    # experiments = ['ICBC',]
    ensnames = ['anl',]
elif enstype == 'ICBC':
    ensnames =  ['c00'] + ['p'+"%02d" %n for n in range(1,11)]
    experiments = ['ICBC',]

if case[:4] == '2006':
    itime = (2006,5,26,12,0,0)
    ftime = (2006,5,27,13,0,0)
    times = [(2006,5,27,3,0,0),]
elif case[:4] == '2009':
    itime = (2009,9,10,23,0,0)
    ftime = (2009,9,11,14,0,0)
elif case[:4] == '2011':
    itime = (2011,4,19,18,0,0)
    ftime = (2011,4,20,10,30,0)
elif case[:4] == '2013':
    itime = (2013,8,15,3,0,0)
    ftime = (2013,8,16,12,0,0)
    # times = [(2013,8,16,3,0,0),]
else:
    raise Exception

hourly = 6
levels = 2000

def get_folders(en,ex):
    if enstype == 'STCH':
        out_sd = os.path.join(case,IC,en,MP,ex)
        wrf_sd = os.path.join(case,IC,en,MP,ex)
    else:
        out_sd = os.path.join(case,IC,en,ex)
        wrf_sd = os.path.join(case,IC,en,ex)
    return out_sd, wrf_sd


# times = utils.generate_times(itime,ftime,hourly*60*60)

#shear_times = utils.generate_times(itime,ftime,3*60*60)
#sl_times = utils.generate_times(sl_itime,sl_ftime,1*60*60)
# skewT_time = (2013,8,16,3,0,0)
# skewT_latlon = (35.2435,-97.4708)
# thresh = 10

#variables = {'cref':{}} ; variables['cref'] = {'lv':2000,'pt':times}
#variables = {'strongestwind':{}} ; variables['strongestwind'] = {'lv':2000, 'itime':itime, 'ftime':ftime, 'range':(thresh,27.5,1.25)}
#variables['PMSL'] = {'lv':2000,'pt':times,'plottype':'contour','smooth':5}
#variables['wind10'] = {'lv':2000,'pt':times}
#variables['buoyancy'] = {'lv':2000,'pt':times}
#variables['shear'] = {'top':3,'bottom':0,'pt':shear_times}
# variables = {'shear':{'pt':shear_times, 'top':3, 'bottom':0}}
# variables = {'thetae':{'lv':2000,'pt':times}, 'CAPE':{'pt':times}}
# variables = {'cref':{'lv':2000,'pt':times}, 'shear':{'pt':shear_times, 'top':3, 'bottom':0}}
#variables = {'PMSL':{'lv':2000,'pt':times,'plottype':'contour','smooth':5}}

#shear06 = {'shear':{'top':6,'bottom':0,'pt':shear_times}}
if skewT:
    for en in ensnames:
        for ex in experiments:
            # Reload settings
            p.C = Settings()
    
            # Change paths to new location
            p.C.output_root = os.path.join(config.output_root,case,IC,en,MP,ex)
            p.C.wrfout_root = os.path.join(config.wrfout_root,case,IC,en,MP,ex)
            p.C.pickledir = os.path.join(config.wrfout_root,case,IC,en,MP,ex)
    
            p.plot_skewT(skewT_time,skewT_latlon,save_output=1)
    
    #p.plot_skewT(skewT_time,skewT_latlon,composite=1)

if plot2D:
    for en in ensnames:
        for ex in experiments:
            out_sd, wrf_sd = get_folders(en,ex)
            
            # p.plot_strongest_wind(itime,ftime,2000,wrf_sd=wrf_sd,out_sd=out_sd)
            p.plot2D('cref',times,levels,wrf_sd=wrf_sd,out_sd=out_sd)

if streamlines:
    for en in ensnames:
        # Reload settings
        p.C = Settings()
        # Change paths to new location
        p.C.output_root = os.path.join(config.output_root,case,IC,en,experiment)
        p.C.wrfout_root = os.path.join(config.wrfout_root,case,IC,en,experiment)
        #p.plot_2D(variables)
        print p.C.wrfout_root
        p.plot_streamlines(2000,sl_times)

if rucplot:
    # RUC file is one-per-time so .nc file is specified beforehand
    en = ensnames[0]
    RC = Settings()
    RC.output_root = os.path.join(config.output_root,case,IC,en,experiment)
    RC.path_to_RUC = os.path.join(config.RUC_root,case,IC,en,experiment)
    WRF_dir = os.path.join(config.wrfout_root,case,'NAM',en,'ICBC')
    
    variables = ['streamlines',]
    level = 2000
    
    for t in sl_times:
        RUC = RUCPlot(RC,t,wrfdir=WRF_dir)
        #limits = RUC.colocate_WRF_map(WRF_dir)
        RUC.plot(variables,level)

if coldpoolstrength:
    for t in times:
        for en in ensnames:
            for ex in experiments:
                fig = plt.figure(figsize=(8,6))
                gs = M.gridspec.GridSpec(1,2,width_ratios=[1,3])
                ax0 = plt.subplot(gs[0])
                ax1 = plt.subplot(gs[1])
                
                out_sd, wrf_sd = get_folders(en,ex)
                # print out_sd, wrf_sd
                cf0, cf1 = p.cold_pool_strength(t,wrf_sd=wrf_sd,out_sd=out_sd,
                                    swath_width=130,fig=fig,axes=(ax0,ax1),dz=1)
                plt.close(fig)

if spaghetti:
    wrf_sds = [] 
    for en in ensnames:
        for ex in experiments:
            out_sd, wrf_sd = get_folders(en,ex)
            wrf_sds.append(wrf_sd)
    
    lv = 2000
    # Save to higher directory
    out_d = os.path.dirname(out_sd) 
    for t in times:
        p.spaghetti(t,lv,'cref',40,wrf_sds[:4],out_d)
                
if std:
    wrf_sds = [] 
    for en in ensnames:
        for ex in experiments:
            out_sd, wrf_sd = get_folders(en,ex)
            wrf_sds.append(wrf_sd)
    
    lv = 2000
    # Save to higher directory
    out_d = os.path.dirname(out_sd) 
    if enstype == 'ICBC':
        out_d = os.path.dirname(out_d)
    for t in times:
        p.std(t,lv,'RH',wrf_sds,out_d,clvs=N.arange(0,26,1))

if profiles:
    wrf_sds = [] 
    for en in ensnames:
        for ex in experiments:
            out_sd, wrf_sd = get_folders(en,ex)
            wrf_sds.append(wrf_sd)

    locs = {'KTOP':(39.073,-95.626),'KOAX':(41.320,-96.366),
           'KOUN':(35.244,-97.471)}
    lv = 2000
    vrbl = 'RH'; xlim=[0,110,10]
    # vrbl = 'wind'; xlim=[0,50,5]
    # Save to higher directory
    ml = -2
    out_d = os.path.dirname(out_sd) 
    if enstype == 'ICBC':
        out_d = os.path.dirname(out_d)
        ml = -3
    for t in times:
        for ln,ll in locs.iteritems():
            p.twopanel_profile(vrbl,t,wrf_sds,out_d,two_panel=1,
                                xlim=xlim,ylim=[500,1000,50],
                                latlon=ll,locname=ln,ml=ml)


if frontogenesis:
    for en in ensnames:
        for ex in experiments:
            out_sd, wrf_sd = get_folders(en,ex)
            for time in times: 
                p.frontogenesis(time,850,wrf_sd=wrf_sd,out_sd=out_sd,
                                clvs=N.arange(-5.0,5.25,0.25)*10**-11
                                # clvs = N.arange(0,1.3,0.01)*10**-3
                                )

if upperlevel:
    for en in ensnames:
        for ex in experiments:
            out_sd, wrf_sd = get_folders(en,ex)
            for time in times: 
                p.upperlevel_W(time,850,wrf_sd=wrf_sd,out_sd=out_sd,
                                clvs = N.arange(0,1.0,0.01)
                                )
 
