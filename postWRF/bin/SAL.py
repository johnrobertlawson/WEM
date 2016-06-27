import matplotlib as M
M.use('Agg')
import matplotlib.pyplot as plt
import datetime
import os
import numpy as N 
import pickle as pickle

from WEM.postWRF.postWRF.sal import SAL
from WEM.postWRF.postWRF import WRFEnviron
import WEM.utils as utils

# case = '2013081500'
# case = '2011041900'
# 3km NESTED, 3km SINGLE, or 1km HIRES
# nest = 'NESTED'
# nest = 'SINGLE'
# nest = 'HIRES'
paper = 2
# thresh = 30
# footprint = 300

# threshs = [15,20,25,30,35,40]
# fps = [100,200,400,600]
threshs = [15,]
fps = [199,]
cases = ['2013081500',]

import itertools
for case in cases:
    for thresh, footprint in itertools.product(threshs,fps):

        # vrbl = 'accum_precip';clvs=N.arange(5,65,5)
        vrbl = 'cref'; clvs=False; use_radar_obs = True
        # vrbl = 'REFL_comp'; clvs=False; use_radar_obs = True
        accum_hr = 1

        plotfig = 0
        SALplot = 1
        SALplot_only = 1

        if case[:4] == '2011':
            # itime = (2011,4,19,23,0,0)
            itime = (2011,4,19,0,0,0)
            ftime = (2011,4,20,12,30,0)
        elif case[:4] == '2013':
            itime = (2013,8,15,0,0,0)
            ftime = (2013,8,16,13,0,0)
        hourly = 1

        times = utils.generate_times(itime,ftime,hourly*60*60)

        ALLSPREAD = {}
        if paper == 1 and case[:4] == '2013':
            casedir = case+'_paper1'
        elif paper == 1 and case[:4] == '2011':
            raise Exception
        else:
            casedir = case
        outdir = '/home/jrlawson/public_html/bowecho/SAL/{0}'.format(casedir)
        utils.trycreate(outdir)


        def nest_logic(nest):
            if nest == 'NESTED':
                dom = 1
                hires = True
            elif nest == 'SINGLE':
                dom = 1
                hires = False
            elif nest == 'HIRES':
                dom = 2
                hires = True
            return dom, hires


        def case_logic(case):
            if case[:4] == '2013':
                nct = datetime.datetime(2013,8,15,0,0,0)
                casestr = '20130815'
                W_fname = 'wrfout_d0{0}_2013-08-15_00:00:00'.format(dom)
            else:
                nct = datetime.datetime(2011,4,19,0,0,0)
                casestr = '20110419'
                W_fname = 'wrfout_d0{0}_2011-04-19_00:00:00'.format(dom)
            return nct, casestr, W_fname

        def hires_logic(casestr,hires):
            if hires:
                casestr = casestr + '_hires'
                ctrl_dir = '/chinook2/jrlawson/bowecho/{0}'.format(casestr)
                ensnames = ['s{0:02d}'.format(e) for e in range(21,31)]
            else:
                if casestr[:4] == '2013':
                    if paper == 2:
                        ctrl_dir = '/chinook2/jrlawson/bowecho/20130815/GEFSR2_paper2/p09/ICBC'
                    else:
                        ctrl_dir = '/chinook2/jrlawson/bowecho/20130815/GEFSR2_paper1/p09/ICBC'
                else:
                    ctrl_dir = '/chinook2/jrlawson/bowecho/20110419/NAM/anl/WSM5'
                ensnames = ['s{0:02d}'.format(e) for e in range(1,11)]
            return casestr, ctrl_dir, ensnames

            ensnames = ensname_logic(case)

        def ensname_logic(nest):
            if nest=='SINGLE':
                ensnames = ['s{0:02d}'.format(e) for e in range(1,11)]
            elif nest == 'NESTED':
                ensnames = ['s{0:02d}'.format(e) for e in range(21,31)]
            return ensnames

        if paper == 1:
            nestlist = ['SINGLE',]
        else:
            nestlist = ['SINGLE','NESTED']
        #'HIRES'):

        if not SALplot_only:
            for nest in nestlist:
                ALLSPREAD[nest] = {}
                for dnt in times: 
                    utc = datetime.datetime(*utils.ensure_timetuple(dnt)[:6]) 
                    ALLSPREAD[nest][dnt] = {}

                    dom, hires = nest_logic(nest)
                    nct, casestr, W_fname = case_logic(case)
                    casestr, ctrl_dir, ensnames = hires_logic(casestr,hires)

                    ctrl_fpath = os.path.join(ctrl_dir,W_fname)
                    radar_datadir = os.path.join('/chinook2/jrlawson/bowecho/',case[:8],'VERIF')
                    fhr_ = utc-nct
                    # import pdb; pdb.set_trace()
                    totalsec = fhr_.seconds + 24*60*60*(fhr_.days)
                    fhr = '{0:02d}'.format(totalsec/(60*60))
                    DATA = {}
                    if plotfig:
                        p = WRFEnviron()
                        fig, axes = plt.subplots(3,4,figsize=(8,6))
                        if use_radar_obs:
                            p.plot_radar(utc,radar_datadir,ncdir=ctrl_dir,fig=fig,ax=axes.flat[0],cb=False,nct=nct,dom=dom)
                            axes.flat[0].set_title("NEXRAD")
                        else:
                            p.plot2D(vrbl,utc,fig=fig,ax=axes.flat[0],ncdir=ctrl_dir,nct=nct,cb=False,dom=dom,accum_hr=accum_hr,clvs=clvs)
                            axes.flat[0].set_title("Ctrl")
                    else:
                        axes = N.array([False,] * 12)

                    if use_radar_obs:
                        ctrl_fpath = False
                    else:
                        radar_datadir = False

                    print(("----- {0}\t{1} ------".format(nest,utc)))
                    # for ens in ensnames[0:1]:
                    for ens,ax in zip(ensnames,axes.flat[1:12]):
                        # if ens=='s21':
                            # continue
                        # import pdb; pdb.set_trace()
                        ALLSPREAD[nest][dnt][ens] = {}
                        DATA[ens] = {}
                        mod_dir = os.path.join(ctrl_dir,ens)
                        mod_fpath = os.path.join(mod_dir,W_fname)
                        sal = SAL(ctrl_fpath,mod_fpath,vrbl,utc,accum_hr=accum_hr,radar_datadir=radar_datadir,thresh=thresh,
                                    footprint=footprint)
                        # sal = SAL(ctrl_fpath,mod_fpath,'REFL_comp',utc)
                        DATA[ens]['S'] = sal.S
                        DATA[ens]['A'] = sal.A
                        DATA[ens]['L'] = sal.L
                        ALLSPREAD[nest][dnt][ens]['S'] = sal.S
                        ALLSPREAD[nest][dnt][ens]['A'] = sal.A
                        ALLSPREAD[nest][dnt][ens]['L'] = sal.L
                        

                        if plotfig:
                            p.plot2D(vrbl,utc,fig=fig,ax=ax,ncdir=mod_dir,nct=nct,cb=False,dom=dom,accum_hr=accum_hr,clvs=clvs)
                            ax.set_title(ens)
                        
                    if plotfig:
                        axes.flat[-1].axis('off')

                        fig.tight_layout()
                        fig.savefig(os.path.join(outdir,'thumbnails_{0}_{1}_{2}h.png'.format(vrbl,nest,fhr)))
                        plt.close(fig)

                        fig = plt.figure(1)
                        ax = fig.add_subplot(111)
                        # L_range = N.linspace(0,2,9)
                        # colors = plt.cm.coolwarm(L_range)
                        plt.axhline(0, color='k')
                        plt.axvline(0, color='k')
                        ax.set_xlim([-2,2])
                        ax.set_ylim([-2,2])
                        for k,v in list(DATA.items()):
                            sc = ax.scatter(v['S'],v['A'],c=v['L'],vmin=0,vmax=2,s=25,cmap=plt.cm.get_cmap('nipy_spectral_r'))
                            ax.annotate(k[1:],xy=(v['S']+0.03,v['A']+0.03),xycoords='data',fontsize=5)
                        # CBax t= fig.add_axes([0.15,0.05,0.7,0.02])
                        ax.set_xlabel("Structural component")
                        ax.set_ylabel("Amplitude component")
                        plt.colorbar(sc,label="Location component")
                        fig.tight_layout()
                        fig.savefig(os.path.join(outdir,'SAL_{0}_{1}_{2}h_{3}dBZ_{4}fp.png'.format(vrbl,nest,fhr,thresh,footprint)))
                        plt.close(fig)

            # Pickle SAL dictionary here
            picklefname = 'SAL_{0}_{1}dBZ_{2}fp.pickle'.format(vrbl,thresh,footprint)
            picklef = os.path.join(outdir,picklefname)
            pickle.dump(ALLSPREAD, open(picklef, 'wb'))

        if SALplot or SALplot_only:
            nplt = len(nestlist)
            fig, axes = plt.subplots(nplt,1,figsize=(9,4*nplt))

            picklefname = 'SAL_{0}_{1}dBZ_{2}fp.pickle'.format(vrbl,thresh,footprint)
            picklef = os.path.join(outdir,picklefname)
            SALDATA = pickle.load(open(picklef, 'rb'))
            arrERR = {}
            for nn,nest in enumerate(nestlist):
                ensnames = ensname_logic(nest)
                arrERR[nest] = N.empty([len(ensnames),len(times)])
                for nt,t in enumerate(times):
                    arrSAL = N.empty([len(ensnames),3])
                    for ne, ens in enumerate(ensnames):
                        arrSAL[ne,0] = N.abs(SALDATA[nest][t][ens]['S'])
                        arrSAL[ne,1] = N.abs(SALDATA[nest][t][ens]['S'])
                        arrSAL[ne,2] = N.abs(SALDATA[nest][t][ens]['S'])
                    if N.any(arrSAL[:,:] > 1.99):
                        arrERR[nest][:,nt] = N.zeros_like(arrERR[nest][:,nt])
                    else:
                        arrERR[nest][:,nt] = N.sum(arrSAL,axis=1)

            colorcode = 1
            if paper == 1:
                # colorcode = 0
                try: 
                    ax = axes.flat[nn]
                except:
                    ax = axes
                ax.grid(zorder=0,color='k')
                if not colorcode:
                    # ax.axhline(2,lw=1,color='k',zorder=2)
                    ax.boxplot(arrERR[nest],sym='',whis=50000,boxprops={
                            'facecolor':'deepskyblue','zorder':100,},patch_artist=True,medianprops={
                            'color':'midnightblue','zorder':105,'lw':3},whiskerprops={'color':'black','linestyle':'solid'},
                            widths=[0.7,]*arrERR[nest].shape[1])
                else:
                    # load SINGLE and NESTED vals
                    # Compare!

                    sm_dir = '/home/jrlawson/public_html/bowecho/SAL/2013081500'
                    pickle_sm_fname = 'SAL_spreads_meds_{0}_{1}dBZ_{2}fp.pickle'.format(vrbl,thresh,footprint)
                    pickle_sm_f = os.path.join(sm_dir,pickle_sm_fname)
                    SM_DATA = pickle.load(open(pickle_sm_f, 'rb'))

                    for nt,t in enumerate(times):
                        LG16val = arrERR[nest][:,nt]

                        LG16spread = N.abs(N.percentile(LG16val,25) - N.percentile(LG16val,75))
                        singspread = SM_DATA['singspreads'][nt]
                        nestspread = SM_DATA['nestspreads'][nt]

                        LG16med = N.median(LG16val)
                        singmed = SM_DATA['singmeds'][nt]
                        nestmed = SM_DATA['nestmeds'][nt]
                        # import pdb; pdb.set_trace()

                        if LG16med == singspread == nestspread == 0.0:
                            continue

                        if (singspread == 0.0) and (nestspread == 0.0):
                            boxcol = 'white'
                        elif (LG16spread > singspread) and (LG16spread > nestspread):
                            boxcol = 'yellow'
                        elif (LG16spread < singspread) and (LG16spread < nestspread):
                            boxcol = 'lightgray'
                        else:
                            boxcol = 'white'

                        if (singmed == 0.0) and (nestmed == 0.0):
                            medcol = 'black'
                        elif (LG16med > singmed) and (LG16med > nestmed):
                            medcol = 'red'
                        elif (LG16med < singmed) and (LG16med < nestmed):
                            medcol = 'darkgreen'
                        else:
                            medcol = 'black'

                        if LG16spread > 0.0:
                            ax.boxplot(arrERR[nest][:,nt],positions=N.array([nt,]),sym='',whis=50000,boxprops={
                                'facecolor':boxcol,'zorder':100,},patch_artist=True,medianprops={
                                'color':medcol,'zorder':105,'lw':3},whiskerprops={'color':'black','linestyle':'solid'},
                                widths=[0.7,])

                ax.set_ylim([0,6.1])
                ax.set_xticks(N.arange(1,37,1))
                ax.set_xticklabels(['{0}h'.format(tlab) for tlab in N.arange(37)])

            elif colorcode:
                EXPORT = {}
                EXPORT['singspreads'] = []
                EXPORT['singmeds'] = []
                EXPORT['nestspreads'] = []
                EXPORT['nestmeds'] = []
                axsing = axes.flat[0]
                axnest = axes.flat[1]
                # ax.axhline(2,lw=1,color='k',zorder=2)
                axsing.grid(zorder=0,color='k')
                axnest.grid(zorder=0,color='k')

                for nt,t in enumerate(times):
                    
                    singval = arrERR['SINGLE'][:,nt]
                    nestval = arrERR['NESTED'][:,nt]

                    singspread = N.abs(N.percentile(singval,25) - N.percentile(singval,75))
                    nestspread = N.abs(N.percentile(nestval,25) - N.percentile(nestval,75))

                    singmed = N.median(singval)
                    nestmed = N.median(nestval)
                    # import pdb; pdb.set_trace()
                    EXPORT['singspreads'].append(singspread)
                    EXPORT['singmeds'].append(singmed)
                    EXPORT['nestspreads'].append(nestspread)
                    EXPORT['nestmeds'].append(nestmed)

                    # import pdb; pdb.set_trace()
                    if singspread == nestspread == 0.0:
                        continue

                    if (singspread == 0.0) or (nestspread == 0.0):
                        singboxcol = 'white'
                        nestboxcol = 'white'
                    elif singspread > nestspread:
                        singboxcol = 'yellow'
                        nestboxcol = 'lightgray'
                    else:
                        singboxcol = 'lightgray'
                        nestboxcol = 'yellow'

                    if (singmed == 0.0) or (nestmed == 0.0):
                        singmedcol = 'black'
                        nestmedcol = 'black'
                    elif singmed > nestmed:
                        singmedcol = 'red'
                        nestmedcol = 'darkgreen'
                    else:
                        singmedcol = 'darkgreen'
                        nestmedcol = 'red'

                    if singspread > 0.0:
                        axsing.boxplot(arrERR['SINGLE'][:,nt],positions=N.array([nt,]),sym='',whis=50000,boxprops={
                            'facecolor':singboxcol,'zorder':100,},patch_artist=True,medianprops={
                            'color':singmedcol,'zorder':105,'lw':3},whiskerprops={'color':'black','linestyle':'solid'},
                            widths=[0.7,])
                    if nestspread > 0.0:
                        axnest.boxplot(arrERR['NESTED'][:,nt],positions=N.array([nt,]),sym='',whis=50000,boxprops={
                            'facecolor':nestboxcol,'zorder':100},patch_artist=True,medianprops={
                            'color':nestmedcol,'zorder':105,'lw':3},whiskerprops={'color':'black','linestyle':'solid'},
                            widths=[0.7,])
                    utc = datetime.datetime(*utils.ensure_timetuple(t)[:6]) 



                axsing.set_ylim([0,6.1])
                axsing.set_xticks(N.arange(0,37,1))
                axsing.set_xticklabels(['{0}h'.format(tlab) for tlab in N.arange(37)])
                axsing.set_title("Single domain")
                axsing.set_ylabel("Total absolute SAL error")
                axnest.set_ylim([0,6.1])
                axnest.set_xticks(N.arange(0,37,1))
                axnest.set_xticklabels(['{0}h'.format(tlab) for tlab in N.arange(37)])
                axnest.set_title("Nested domain")
                axnest.set_xlabel("Forecast time (h)")
                axnest.set_ylabel("Total absolute SAL error")

                picklefname = 'SAL_spreads_meds_{0}_{1}dBZ_{2}fp.pickle'.format(vrbl,thresh,footprint)
                picklef = os.path.join(outdir,picklefname)
                pickle.dump(EXPORT, open(picklef, 'wb'))

            # handles, labels = ax.get_legend_handles_labels()
            # ax.legend(handles,labels)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir,'BoxPlot_SAL_{0}_{1}dBZ_{2}fp.png'.format(vrbl,thresh,footprint))) 
            plt.close(fig)


