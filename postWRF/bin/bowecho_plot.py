import os
import pdb
import sys
import matplotlib as M
M.use('agg')
import matplotlib.pyplot as plt
import numpy as N
import glob
import collections

sys.path.append('/home/jrlawson/gitprojects/')

from WEM.postWRF.postWRF import WRFEnviron
import WEM.utils as utils
#from WEM.postWRF.postWRF.rucplot import RUCPlot

outroot = '/home/jrlawson/public_html/bowecho/'
ncroot = '/chinook2/jrlawson/bowecho/'

p = WRFEnviron()

skewT = 0
plot2D = 0
radarplot = 0
radarcomp = 0
streamlines = 0
rucplot = 0
coldpoolstrength = 0
spaghetti = 0
std = 0
profiles = 0
frontogenesis = 0
upperlevel = 0
strongestwind = 0
accum_rain = 0
compute_dte = 0
plot_1D_dte = 0 # To produce top-down maps
plot_3D_dte = 0 # To produce line graphs
all_3D_dte = 1 # To produce line graphs for all averages
delta_plot = 0
powerspectrum = 0
probability = 0

# enstype = False
# enstype = 'STCH'
# enstype = 'STCH5'
# enstype = 'ICBC'
enstype = 'MXMP'
# enstype = 'STMX'

case = '20060526'
# case = '20090910'
# case = '20110419'
# case = '20130815'
 
# IC = 'GEFSR2'
IC = 'NAM'
# IC = 'RUC'
# IC = 'GFS'
# IC = 'RUC'
# IC = False

if enstype == 'STCH':
    experiments = ['s'+"%02d" %n for n in range(1,11)]
    ensnames = ['anl',]
    # ensnames = ['p09',]
    # MP = 'ICBC'
    MP = 'WSM5'
    # MP = 'WDM6_Grau'
    # MP = 'Morrison_Hail'
elif enstype == 'STCH5':
    experiments = ['ss'+"%02d" %n for n in range(1,11)]
    ensnames = ['anl',]
    MP = 'WDM6_Grau'
    # MP = 'ICBC'
elif enstype == 'STMX':
    # experiments = ['WSM6_Hail','Kessler','Ferrier',
    experiments = ['WSM6_Grau_STCH','WSM6_Hail_STCH','Kessler_STCH',
                    'Ferrier_STCH', 'WSM5_STCH','WDM5_STCH','Lin_STCH',
                    'WDM6_Grau_STCH','WDM6_Hail_STCH',
                    'Morrison_Grau_STCH','Morrison_Hail_STCH',]
                    # 'ICBC_STCH']
    ensnames = ['anl',]
elif enstype == 'MXMP':
    experiments = ['WSM6_Grau','WSM6_Hail','Kessler','Ferrier',
                    'WSM5','WDM5','Lin','WDM6_Grau','WDM6_Hail',
                    'Morrison_Grau','Morrison_Hail','ICBC']
    experiments = ['WSM5',]
    # experiments = ['WDM6_Grau',]
    ensnames = ['anl',]
elif enstype == 'ICBC':
    ensnames =  ['c00'] + ['p'+"%02d" %n for n in range(1,11)]
    experiments = ['ICBC',]
else:
    ensnames = ['anl']
    if IC == 'RUC':
        experiments = ['VERIF',]
    else:
        experiments = ['WDM6_Grau','ICBC',]

if case[:4] == '2006':
    nct = (2006,5,26,0,0,0)
    itime = (2006,5,26,3,0,0)
    ftime = (2006,5,26,6,0,0)
    iwind = (2006,5,26,18,0,0)
    fwind = (2006,5,27,12,0,0)
    compt = [(2006,5,d,h,0,0) for d,h in zip((26,27,27),(23,3,6))]
    # times = [(2006,5,27,6,0,0),]
    matchnc = '/chinook2/jrlawson/bowecho/20060526/GFS/anl/ICBC/wrfout_d01_2006-05-26_00:00:00'

elif case[:4] == '2009':
    inittime = (2009,9,10,23,0,0)
    itime = (2009,9,10,23,0,0)
    ftime = (2009,9,11,14,0,0)
elif case[:4] == '2011':
    nct = inittime = (2011,4,19,0,0,0)
    itime = (2011,4,19,18,0,0)
    ftime = (2011,4,20,12,0,0)
    iwind = (2011,4,19,18,0,0)
    fwind = (2011,4,20,12,0,0)
elif case[:4] == '2013':
    inittime = (2013,8,15,0,0,0)
    itime = (2013,8,15,0,0,0)
    ftime = (2013,8,16,12,0,0)
    times = [(2013,8,16,3,0,0),]
    dtetime = [(2013,8,16,0,0,0),]
    compt = [(2013,8,d,h,0,0) for d,h in zip((15,16,16),(22,2,6))]
    powertime = (2013,8,16,3,0,0)
    iwind = (2013,8,15,18,0,0)
    fwind = (2013,8,16,11,0,0)
    matchnc = '/chinook2/jrlawson/bowecho/20130815_hires/wrfout_d02_2013-08-15_00:00:00'
else:
    raise Exception

# experiments = ['Morrison_Hail_STCH',]
# ensnames = ['anl',]
# itime = (2006,5,25,12,0,0)
# ftime = (2006,5,27,12,0,0)

hourly = 3
level = 2000
times = utils.generate_times(itime,ftime,hourly*60*60)
# dtetimes = utils.generate_times(itime,ftime,hourly*60*60)

def get_folders(en,ex):
    if enstype and enstype[:4] == 'STCH':
        out_sd = os.path.join(outroot,case,IC,en,MP,ex)
        wrf_sd = os.path.join(ncroot,case,IC,en,MP,ex)
    else:
        out_sd = os.path.join(outroot,case,IC,en,ex)
        wrf_sd = os.path.join(ncroot,case,IC,en,ex)
    return out_sd, wrf_sd

def get_verif_dirs():
    outdir = os.path.join(outroot,case,'VERIF')
    datadir = os.path.join(ncroot,case,'VERIF')
    return outdir,datadir

def get_pickle_dirs(en):
    if enstype[:4]=='STCH':
        picklefolder = os.path.join(ncroot,case,IC,en,MP)
        outfolder = os.path.join(outroot,case,IC,en,MP)
    elif enstype == 'ICBC':
        picklefolder = os.path.join(ncroot,case,IC)
        outfolder = os.path.join(outroot,case,IC)
    else:
        picklefolder = os.path.join(ncroot,case,IC,en)
        outfolder = os.path.join(outroot,case,IC,en)

    return picklefolder,outfolder

def create_ensemble():
    ensemble = {}
    if enstype == 'ICBC':
        for member in ensnames:
            path = os.path.join(ncroot,case,IC,member,'ICBC')
            ncfname = glob.glob(os.path.join(path,'wrfout_d*'))[0]
            fpath = os.path.join(path,ncfname)
            ensemble[member] = {'path':fpath,'control':(member=='c00')}
    elif enstype[:4] == 'STCH':
        # If you need control...
        # path = os.path.join(ncroot,case,IC,ensnames[0],MP
        for member in experiments:
            path = os.path.join(ncroot,case,IC,ensnames[0],MP,member)
            ncfname = glob.glob(os.path.join(path,'wrfout_d*'))[0]
            fpath = os.path.join(path,ncfname)
            ensemble[member] = {'path':fpath,'control':False}
    elif enstype == 'MXMP' or enstype=='STMX':
        for member in experiments:
            path = os.path.join(ncroot,case,IC,ensnames[0],member)
            ncfname = glob.glob(os.path.join(path,'wrfout_d*'))[0]
            fpath = os.path.join(path,ncfname)
            ensemble[member] = {'path':fpath,'control':False}
    return ensemble

#shear_times = utils.generate_times(itime,ftime,3*60*60)
#sl_times = utils.generate_times(sl_itime,sl_ftime,1*60*60)
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

skewT_time = (2006,5,27,0,0,0)
skewT_latlon = (36.73,-102.51) # Boise City, OK

if skewT:
    for en in ensnames:
        for ex in experiments:
            outdir, ncdir = get_folders(en,ex)
            p.plot_skewT(skewT_time,latlon=skewT_latlon,outdir=outdir,ncdir=ncdir)

locs = {'Norman':(35.2,-97.4)}
if plot2D or radarplot:
    for en in ensnames:
        for ex in experiments:
            for t in times:
                outdir, ncdir = get_folders(en,ex)
                if plot2D:
                    # p.plot2D('Z',t,500,wrf_sd=wrf_sd,out_sd=out_sd,plottype='contour',smooth=10)
                    # p.plot2D('Td2',t,ncdir=ncdir,outdir=outdir,nct=t,match_nc=matchnc,clvs=N.arange(260,291,1))
                    # p.plot2D('Q2',t,ncdir=ncdir,outdir=outdir,nct=t,match_nc=matchnc,clvs=N.arange(1,20.5,0.5)*10**-3)
                    # p.plot2D('RAINNC',t,ncdir=wrf_sd,outdir=out_sd,locations=locs,clvs=N.arange(1,100,2))
                    # p.plot2D('REFL_comp',t,ncdir=ncdir,outdir=outdir,cb=True,match_nc=matchnc)
                    p.plot2D('cref',t,ncdir=ncdir,outdir=outdir,cb=True)
                    # p.plot2D('wind10',t,ncdir=ncdir,outdir=outdir,locations=locs,cb=True,clvs=N.arange(10,32,2))

                if radarplot:
                    outdir,datadir = get_verif_dirs()
                    p.plot_radar(t,datadir,outdir=outdir,ncdir=ncdir)

if radarcomp:
    en = ensnames[0]
    ex = experiments[0]
    out_sd, wrf_sd = get_folders(en,ex)
    outdir, datadir = get_verif_dirs()
    # p.plot_radar(compt,datadir,outdir,ncdir=wrf_sd,composite=True)
    p.plot_radar(compt,datadir,outdir,composite=True,
                    # Nlim=40.1,Elim=-94.9,Slim=34.3,Wlim=-100.8)
                    Nlim=42.7,Elim=-94.9,Slim=37.0,Wlim=-101.8)

if streamlines:
    for en in ensnames:
        for ex in experiments:
            for t in times:
                out_sd, wrf_sd = get_folders(en,ex)
                p.plot_streamlines(t,2000,out_sd=out_sd,wrf_sd=wrf_sd)

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

    # locs = {'KTOP':(39.073,-95.626),'KOAX':(41.320,-96.366),'KOUN':(35.244,-97.471)}
    locs = {'KAMA':(35.2202,-101.7173)}
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
        for ln,ll in locs.items():
            p.twopanel_profile(vrbl,t,wrf_sds,out_d,two_panel=1,
                                xlim=xlim,ylim=[500,1000,50],
                                latlon=ll,locname=ln,ml=ml)


if frontogenesis:
    for en in ensnames:
        for ex in experiments:
            outdir, ncdir = get_folders(en,ex)
            for t in times:
                lv = 2000
                p.frontogenesis(t,lv,ncdir=ncdir,outdir=outdir,
                                clvs=N.arange(-2.0,2.125,0.125)*10**-7,
                                # clvs = N.arange(-500,510,10)
                                smooth=3, cmap='bwr',cb='only'
                                )

if upperlevel:
    for en in ensnames:
        for ex in experiments:
            out_sd, wrf_sd = get_folders(en,ex)
            for time in times: 
                p.upperlevel_W(time,850,wrf_sd=wrf_sd,out_sd=out_sd,
                                clvs = N.arange(0,1.0,0.01)
                                )

windlvs = N.arange(10,31,1)
if strongestwind:
    for en in ensnames:
        for ex in experiments:
            outdir, ncdir = get_folders(en,ex)
            # p.plot_strongest_wind(iwind,fwind,2000,ncdir,outdir,clvs=windlvs)
            p.plot_strongest_wind(iwind,fwind,2000,ncdir=ncdir,nct=nct,outdir=outdir,clvs=windlvs,dom=1,
                    cmap='jet',cb=True)

if accum_rain:
    for en in ensnames:
        for ex in experiments:
            for t in times:
                outdir, ncdir = get_folders(en,ex)
                p.plot_accum_rain(t,6,ncdir,outdir,
                        Nlim=42.7,Elim=-94.9,Slim=37.0,Wlim=-101.8
                        )

if compute_dte or plot_3D_dte or plot_1D_dte or powerspectrum:
    pfname = 'DTE_' + enstype
    ofname = enstype
    pickledir,outdir = get_pickle_dirs(ensnames[0])
    path_to_wrfouts = []
    for en in ensnames:
        for ex in experiments:
            od,fpath = get_folders(en,ex)
            print(fpath)
            path_to_wrfouts.append(utils.netcdf_files_in(fpath))

    if compute_dte:
        p.compute_diff_energy('1D','DTE',path_to_wrfouts,dtetimes,
                              d_save=pickledir, d_return=0,d_fname=pfname)

    if plot_1D_dte:
        # Contour fixed at these values
        # V = range(250,5250,250)
        V = list(range(500,18000,500))
        VV = [250,] + V
        ofname = pfname
        p.plot_diff_energy('1D','DTE',pickledir,outdir,dataf=pfname,outprefix=ofname,clvs=VV,utc=False,cb=True)

    if plot_3D_dte:
        SENS = {'ICBC':ensnames,'MXMP':experiments,'STCH':0,'STCH5':0,'STMX':experiments}
        ylimits = [0,2e8]
        ofname = pfname
        p.plot_error_growth(outdir,pickledir,dataf=pfname,sensitivity=SENS[enstype],ylim=ylimits,f_prefix=enstype)

# if powerspectrum:
    # listofncfiles = 
    # p.plot_diff_power_spectrum('DTE',pickledir,outdir,dataf=pfname,outprefix=ofname,utc=powertime,meanenergy=listofncfiles)

if all_3D_dte:
    if case[:4] == '2006':
        # EXS = {'GEFS-ICBC':{},'NAM-MXMP':{},'NAM-STMX':{},'NAM-STCH5':{},'GEFS-STCH':{},'NAM-STCH':{},'GEFS-MXMP':{}}
        # EXS = {'ICBC (GEFS/R2)':{},'MXMP (NAM)':{},'STMX (NAM)':{},'STCH5 (NAM; WDM6 Grau)':{},'STCH (GEFS/R2 c00)':{},'STCH (NAM; WDM6 Grau)':{},'MXMP (GEFS/R2 c00)':{}}
        # EXS = {'ICBC-Thompson':{},'NAM-MXMP':{},'NAM-STMX':{},'NAM-WDM6Grau-STCH5':{},'c00-Thompson-STCH':{},'NAM-WDM6Grau-STCH':{},'c00-MXMP':{}}
        EXS = collections.OrderedDict([('NAM-MXMP',{}),('NAM-STMX',{}),
                ('ICBC-Thompson',{}),('NAM-WDM6Grau-STCH',{}),
                ('NAM-WDM6Grau-STCH5',{}),('c00-MXMP',{}),('c00-Thompson-STCH',{})])
        IC = 'NAM';ensnames = 'anl'; MP = 'WDM6_Grau'
    else:
        # EXS = {'GEFS-ICBC':{},'NAM-MXMP':{},'GEFS-MXMP':{},'GEFS-STCH-thomp':{},'GEFS-STCH-morrh':{},'GEFS-STMX':{}}
        # EXS = {'ICBC (GEFS/R2)':{},'MXMP (NAM)':{},'MXMP (GEFS/R2 p09)':{},'STCH (GEFS/R2 p09; Thompson)':{},'STCH (GEFS/R2 p09; Morrison Hail)':{},'STMX (GEFS/R2 p09)':{}}
        # EXS = {'ICBC-Thompson':{},'NAM-MXMP':{},'p09-MXMP':{},'p09-Thompson-STCH':{},'p09-MorrisonHail-STCH':{},'p09-STMX':{}}
        EXS = collections.OrderedDict([('ICBC-Thompson',{}),('p09-STMX',{}),('p09-MXMP',{}),('NAM-MXMP',{}),('p09-MorrisonHail-STCH',{}),('p09-Thompson-STCH',{})])
        # Compare Morrison-Hail and Thompson STCH members.
        MP = 'ICBC'
        

    for exper in EXS:
        # exs = exper.split('-')
        # if len(exs)==2:
            # IC, enstype = exs
            # stchmp = False
        # else:
            # IC,enstype,stchmp = exs
        if 'NAM' in exper:
            IC = 'NAM'
        else:
            IC = 'GEFSR2'

        if 'ICBC' in exper:
            enstype = 'ICBC'
            col = '#56B4E9'
        elif 'MXMP' in exper:
            enstype = 'MXMP'
            col = '#F0E442'
        elif 'STCH5' in exper:
            enstype = 'STCH5'
            # col = '#D55E00'
            col = '#CC79A7'
        elif 'STCH' in exper:
            enstype = 'STCH'
            col = '#CC79A7'
        elif 'STMX' in exper:
            enstype = 'STMX'
            col = '#E69F00'
        else:
            raise Exception

        if 'Thompson' in exper:
            stchmp = 'thomp'
        elif 'Morrison' in exper:
            stchmp = 'morrh'
        else:
            stchmp = False

        if IC=='GEFSR2':
            if case[:4] == '2006':
                ensnames = 'c00'
            else:
                ensnames = 'p09'
            MP = 'ICBC'
        elif IC=='NAM':
            ensnames = 'anl'
            MP = 'WDM6_Grau'

        if stchmp=='morrh':
            MP = 'Morrison_Hail'
        elif stchmp == 'thomp':
            MP = 'ICBC'

        if 'STCH' in enstype:
            pass
        elif 'STMX' in enstype:
            pass
        elif 'MXMP' in enstype:
            pass
        elif 'ICBC' in enstype:
            pass

        pickledir,dummy = get_pickle_dirs(ensnames)
        print(exper, pickledir)
        # EXS[exper] = {'datadir':pickledir,'dataf':'DTE_'+enstype}
        EXS[exper]['datadir'] = pickledir
        EXS[exper]['dataf'] = 'DTE_'+enstype
        EXS[exper]['col'] = col
        if 'NAM' in exper:
            EXS[exper]['ls'] = '--'
        else:
            EXS[exper]['ls'] = '-'

        if 'STCH5' in exper:
            EXS[exper]['ls'] = ':'
        if 'MorrisonHail' in exper:
            EXS[exper]['ls'] = '-.'

    ylimits = [0,2e8]
    outdir = os.path.join(outroot,case)
    p.all_error_growth(outdir,EXS,f_prefix='DTE',ylim=ylimits)

if delta_plot:
    for t in times:
        # nc1
        IC = 'GEFSR2'
        en = 'p09'
        enstype = 'STCH'
        ex = 's01'
        MP = 'ICBC'
        outdir, ncdir1 = get_folders(en,ex)
        # nc2
        IC = 'GEFSR2'
        en = 'p09'
        ex = 'ICBC'
        enstype = 'ICBC'
        MP = 'ICBC'
        outdir, ncdir2 = get_folders(en,ex)

        clvs = N.arange(-9,10,1)
        p.plot_delta('wind10',t,ncdir1=ncdir1,ncdir2=ncdir2,outdir=outdir,cb=True,clvs=clvs,)


if probability:
    ensemble = create_ensemble()
    outdir2, ncdir = get_folders(ensnames[0],experiments[0])
    outdir = os.path.dirname(outdir2)
    # outdir = '/home/jrlawson/public_html/bowecho/{0}/{1}/'.format(case,IC)
    smooth = 'maxfilter'
    thresh = 20
    p.probability_threshold(ensemble,'wind10','over',thresh,iwind,fwind,smooth=smooth,outdir=outdir,level=False)
