import os
import pdb
import sys
import matplotlib as M
M.use('agg')
import matplotlib.pyplot as plt
import numpy as N

sys.path.append('/home/jrlawson/gitprojects/')

from WEM.postWRF.postWRF import WRFEnviron
import WEM.utils as utils
#from WEM.postWRF.postWRF.rucplot import RUCPlot


# case = '20130815'
case = '20110419'
ncroot = '/chinook2/jrlawson/bowecho/{0}_hires/'.format(case)
outroot = '/home/jrlawson/public_html/bowecho/hires/{0}'.format(case)

p = WRFEnviron()

skewT = 0
plot2D = 1
radarplot = 0
axesofdilatation = 0
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
all_3D_dte = 0 # To produce line graphs for all averages
delta_plot = 0

lims = {'Nlim':41.64,'Wlim':-89.12,'Slim':38.55,'Elim':-87.04}
windlvs = N.arange(10,31,1)
dom = 1
ensnames = ['s{0:02d}'.format(e) for e in range(21,31)] + ['c00h',False]
ensnames = [False,'c00h',]
# ensnames = ['s25','s26',]

if case[:4] == '2006':
    nct = (2006,5,26,0,0,0)
    itime = (2006,5,26,0,0,0)
    ftime = (2006,5,27,12,0,0)
    iwind = (2006,5,26,18,0,0)
    fwind = (2006,5,27,12,0,0)
    compt = [(2006,5,d,h,0,0) for d,h in zip((26,27,27),(23,3,6))]
    # times = [(2006,5,27,6,0,0),]
    # matchnc = '/chinook2/jrlawson/bowecho/20060526/GFS/anl/ICBC/wrfout_d01_2006-05-26_00:00:00'

elif case[:4] == '2011':
    nct = (2011,4,19,0,0,0)
    itime = (2011,4,20,0,0,0)
    ftime = (2011,4,20,3,0,0)
    iwind = (2011,4,19,18,0,0)
    fwind = (2011,4,20,12,0,0)
    matchnc = '/chinook2/jrlawson/bowecho/20110419_hires/s21/wrfout_d02_2011-04-19_00:00:00'

elif case[:4] == '2013':
    nct = inittime = (2013,8,15,0,0,0)
    itime = (2013,8,15,21,0,0)
    ftime = (2013,8,16,8,0,0)
    iwind = (2013,8,15,21,0,0)
    fwind = (2013,8,16,7,0,0)
    ptime = (2013,8,16,3,0,0)
    compt = [(2013,8,d,h,0,0) for d,h in zip((15,16,16),(22,2,6))]
    matchnc = '/chinook2/jrlawson/bowecho/20130815_hires/wrfout_d02_2013-08-15_00:00:00'
else:
    raise Exception

# hourly = (1.0/12)
hourly = 1
level = 2000
times = utils.generate_times(itime,ftime,hourly*60*60)
# times = [(2011,4,20,2,0,0),]
# times = [(2013,8,15,22,0,0),]
# dtetimes = utils.generate_times(itime,ftime,3*60*60)

skewT_time = (2006,5,27,0,0,0)
skewT_latlon = (36.73,-102.51) # Boise City, OK

for ens in ensnames: 
    if skewT:
        for en in ensnames:
            for ex in experiments:
                outdir, ncdir = get_folders(en,ex)
                p.plot_skewT(skewT_time,latlon=skewT_latlon,outdir=outdir,ncdir=ncdir)

    locs = {'Norman':(35.2,-97.4)}
    if plot2D or radarplot or strongestwind:
        if ens:
            outdir = os.path.join(outroot,'d0{0}'.format(dom),ens)
            ncdir = os.path.join(ncroot,ens)
        else:
            ncdir = ncroot
            outdir = os.path.join(outroot,'d0{0}'.format(dom))

        if strongestwind:
            p.plot_strongest_wind(iwind,fwind,2000,ncdir=ncdir,nct=nct,outdir=outdir,clvs=windlvs,dom=dom,
                    cmap='jet',cb=True)

        for t in times:
            if plot2D:
                # p.plot2D('Z',t,500,wrf_sd=wrf_sd,out_sd=out_sd,plottype='contour',smooth=10)
                # p.plot2D('Td2',t,ncdir=ncdir,outdir=outdir,nct=t,match_nc=matchnc,clvs=N.arange(260,291,1))
                # p.plot2D('Q2',t,ncdir=ncdir,outdir=outdir,nct=t,match_nc=matchnc,clvs=N.arange(1,20.5,0.5)*10**-3)
                # p.plot2D('RAINNC',t,ncdir=wrf_sd,outdir=out_sd,locations=locs,clvs=N.arange(1,100,2))
                # p.plot2D('fluidtrapping',t,ncdir=ncroot,nct=nct,outdir=outdir,cb=True,dom=dom,clvs=N.arange(-1,1,0.1)*10**-6)
                # p.plot2D('lyapunov',t,700,ncdir=ncdir,nct=nct,outdir=outdir,cb=True,dom=dom,clvs=N.arange(-7.5,8.0,0.5)*10**-3,cmap='bwr')
                # print(ncdir)
                # p.plot2D('WSPD10MAX',t,ncdir=ncdir,nct=nct,outdir=outdir,cb=True,dom=dom,clvs=N.arange(10,31,1))
                p.plot2D('cref',t,ncdir=ncdir,nct=nct,outdir=outdir,cb=True,dom=dom,**lims)
                # p.plot2D('REFL_comp',t,ncdir=ncdir,nct=nct,outdir=outdir,cb=True,dom=dom)
                # p.plot2D('shear',t,ncdir=ncdir,nct=nct,outdir=outdir,cb=True,dom=dom)
                # p.plot2D('wind10',t,ncdir=ncdir,outdir=outdir,locations=locs,cb=True,clvs=N.arange(5,32,2))

            if radarplot:
                verifdir = '/chinook2/jrlawson/bowecho/{0}/VERIF'.format(case)
                p.plot_radar(t,verifdir,outdir=outdir,ncdir=ncroot,nct=nct,dom=dom)

    if axesofdilatation:
        outdir = os.path.join(outroot,'hires','d0{0}'.format(dom))
        for t in times:
            p.plot_axes_of_dilatation(utc=t,ncdir=ncroot,nct=nct,outdir=outdir,cb=True,dom=dom)

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
        if ens:
            outdir = os.path.join(outroot,'d0{0}'.format(dom),ens)
            ncdir = os.path.join(ncroot,ens)
        else:
            ncdir = ncroot
            outdir = os.path.join(outroot,'d0{0}'.format(dom))
        for t in times:
            print(ncdir)
            p.plot_streamlines(t,700,ncdir,outdir,nct=nct,dom=dom)

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
        if ens:
            outdir = os.path.join(outroot,'hires','d0{0}'.format(dom),ens)
            ncdir = os.path.join(ncroot,ens)
        else:
            outdir = os.path.join(outroot,'hires','d0{0}'.format(dom))
            ncdir = ncroot
        for t in times:
            fig = plt.figure(figsize=(8,6))
            gs = M.gridspec.GridSpec(1,2,width_ratios=[1,3])
            ax0 = plt.subplot(gs[0])
            ax1 = plt.subplot(gs[1])
            
            # import pdb; pdb.set_trace()
            cf0, cf1 = p.cold_pool_strength(t,ncdir=ncdir,outdir=outdir,nct=nct,dom=dom,
                                swath_width=130,fig=fig,axes=(ax0,ax1),dz=1)
            hr = utils.ensure_timetuple(t)[3]
            fpath = os.path.join(outdir,'coldpool_{0:02d}Z.png'.format(hr))
            fig.savefig(fpath)
            print(("Saving figure to {0}".format(fpath)))
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

    # if strongestwind:
        # for en in ensnames:
            # for ex in experiments:
                # outdir, ncdir = get_folders(en,ex)
                # p.plot_strongest_wind(iwind,fwind,2000,ncdir,outdir,clvs=windlvs)

    if accum_rain:
        for en in ensnames:
            for ex in experiments:
                for t in times:
                    outdir, ncdir = get_folders(en,ex)
                    p.plot_accum_rain(t,6,ncdir,outdir,
                            Nlim=42.7,Elim=-94.9,Slim=37.0,Wlim=-101.8
                            )

    if compute_dte or plot_3D_dte or plot_1D_dte:
        pfname = 'DTE_' + enstype
        ofname = enstype
        pickledir,outdir = get_pickle_dirs(ensnames[0])
        path_to_wrfouts = []
        for en in ensnames:
            for ex in experiments:
                od,fpath = get_folders(en,ex)
                # print fpath
                path_to_wrfouts.append(utils.netcdf_files_in(fpath))

        if compute_dte:
            p.compute_diff_energy('1D','DTE',path_to_wrfouts,dtetimes,
                                  d_save=pickledir, d_return=0,d_fname=pfname)

        if plot_1D_dte:
            # Contour fixed at these values
            V = list(range(250,5250,250))
            VV = [100,] + V
            ofname = pfname
            p.plot_diff_energy('1D','DTE',pickledir,outdir,dataf=pfname,outprefix=ofname,clvs=VV,utc=False)

        if plot_3D_dte:
            SENS = {'ICBC':ensnames,'MXMP':experiments,'STCH':0,'STCH5':0,'STMX':experiments}
            ylimits = [0,2e8]
            ofname = pfname
            p.plot_error_growth(outdir,pickledir,dataf=pfname,sensitivity=SENS[enstype],ylim=ylimits,f_prefix=enstype)

    if all_3D_dte:
        if case[:4] == '2006':
            EXS = {'GEFS-ICBC':{},'NAM-MXMP':{},'NAM-STMX':{},'NAM-STCH5':{},'GEFS-STCH':{},'NAM-STCH':{}}
            IC = 'NAM';ensnames = 'anl'; MP = 'WDM6_Grau'
            # IC = 'NAM';ensnames = 'anl'; MP = 'WDM6_Grau'
        else:
            EXS = {'GEFS-ICBC':{},'NAM-MXMP':{},'GEFS-MXMP':{},'GEFS-STCH-thomp':{},'GEFS-STCH-morrh':{},'GEFS-STMX':{}}
            # Compare Morrison-Hail and Thompson STCH members.
            MP = 'ICBC'
            

        for exper in EXS:
            exs = exper.split('-')
            if len(exs)==2:
                IC, enstype = exs
                stchmp = False
                
            else:
                IC,enstype,stchmp = exs

            if IC=='GEFS':
                IC = 'GEFSR2'
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
            EXS[exper] = {'datadir':pickledir,'dataf':'DTE_'+enstype}

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
            p.plot_delta('wind10',t,ncdir1=ncdir1,ncdir2=ncdir2,outdir=outdir,cb=True,clvs=clvs)
