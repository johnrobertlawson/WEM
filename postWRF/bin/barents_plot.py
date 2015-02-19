import os
import pdb
import sys
import matplotlib as M
M.use('gtkagg')
import matplotlib.pyplot as plt

sys.path.append('/home/jrlawson/gitprojects/')

from WEM.postWRF import WRFEnviron
from barents_settings import Settings
import WEM.utils as utils
#from WEM.postWRF.postWRF.rucplot import RUCPlot

config = Settings()
p = WRFEnviron(config)

skewT = 0
plot2D = 0
streamlines = 1
rucplot = 0
coldpoolstrength = 0

itime = (2007,7,20,0,0,0)
ftime = (2007,7,22,0,0,0)
# times = [(2013,8,15,18,0,0),]
levels = 2000

times = utils.generate_times(itime,ftime,2*60*60)

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
    # p.plot_strongest_wind(itime,ftime,2000,wrf_sd=wrf_sd,out_sd=out_sd)
    p.plot2D('olr',times,levels)

if streamlines:
    for t in times:
        p.plot_streamlines(2000,t)

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
                cf0, cf1 = p.cold_pool_strength(t,wrf_sd=wrf_sd,out_sd=out_sd,swath_width=130,fig=fig,axes=(ax0,ax1))
                plt.close(fig)
