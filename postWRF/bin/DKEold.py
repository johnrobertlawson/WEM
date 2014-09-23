"""This script shows examples of using the package to create arrays
of data stored to disc. This is then plotted using the package.
"""
import sys
import os
import pdb
import calendar
import time
import matplotlib.pyplot as plt

sys.path.append('/uufs/chpc.utah.edu/common/home/u0737349/gitprojects/')

from DKE_settings import Settings
from WEM.postWRF import WRFEnviron
import WEM.utils as utils


# Time script
scriptstart = time.time()

# Initialise settings and environment
config = Settings()
p = WRFEnviron(config)

# User settings
init_time = utils.string_from_time('dir',(2009,9,10,0,0,0),strlen='hour')
rootdir = '/uufs/chpc.utah.edu/common/home/horel-group2/lawson2/'
outdir = '/uufs/chpc.utah.edu/common/home/u0737349/public_html/paper2/'

# Create lists of all fig, ax, cb objects
figs = {}
axes = {}
cbs = {}

ensnames = ['c00'] + ['p{0:02d}'.format(n) for n in range(1,11)]

#for rundate in ('25','27','29'):
plot_all = 1
if plot_all:
    # for rundate in ['25','27','29']:
    for rundate in ['27','29']:
        # print("Computing for {0} November".format(rundate))
        foldername = '201111' + rundate + '00'
        runfolder = os.path.join(rootdir,foldername)
        path_to_wrfouts = utils.wrfout_files_in(runfolder,dom=1)
    
        itime = (2011,11,int(rundate),0,0,0)
        ftime = (2011,12,2,12,0,0)
        times = p.generate_times(itime,ftime,6*3600)
        
        path_to_plots = os.path.join(outdir,foldername)
        # for time in times:
        #pdb.set_trace()
        # Produce .npy data files with DKE data
        # print("Compute_diff_energy...")
        wrfouts = ['/uufs/chpc.utah.edu/common/home/horel-group2/lawson2/201111{0}00/{1}/wrfout_d01_2011-11-{2}_00:00:00_PLEV'.format(rundate,e,rundate)
                    for e in ensnames]
        clvs = 0
        plotfname = 'deltaDKE'
        # p.compute_diff_energy('sum_z','kinetic',path_to_wrfouts,times,upper=500,
                                 # d_save=runfolder, d_return=0,d_fname='DKE_500_sixhrly_'+foldername)
        p.delta_diff_energy('sum_z','kinetic',runfolder,'DKE_500_'+foldername,path_to_plots,plotfname,
                            clvs,wrfouts,'GHT',)
                                
        # Contour fixed at these values
            # plotfname = 'DKE_500_'
            # V = range(200,2200,200)
            # p.plot_diff_energy('sum_z','kinetic',time,runfolder,'DKE_500_'+foldername,
                                # path_to_plots,plotfname,V,no_title=1,ax=ax)
    
if plot_all:
    raise Exception
#print "Script took", time.time()-scriptstart, "seconds."
#pdb.set_trace()

# Create publication figures
fig14, axes14 = plt.subplots(3,1,figsize=(5,7))
initdate = '2011112500'
plotfname = 'DKE_500_'

times = [(2011,m,d,12,0,0) for m,d in zip((11,11,12),(29,30,1))]
runfolder = os.path.join(rootdir,initdate)
path_to_plots = os.path.join(outdir,initdate)
V = range(200,2200,200)
labels = ['a)','b)','c)']

for ax,t,label in zip(axes14.flat, times,labels):
    cf = p.plot_diff_energy('sum_z','kinetic',t,runfolder,'DKE_500_'+initdate,
                                path_to_plots,plotfname,V,no_title=1,ax=ax)
    ax.text(0.05,0.85,label,transform=ax.transAxes,
            bbox={'facecolor':'white'},fontsize=15,zorder=1000)
    
fig14.tight_layout()
fig14.subplots_adjust(bottom=0.15)
cbar_ax = fig14.add_axes([0.15,0.075,0.7,0.025])
cb = fig14.colorbar(cf,cax=cbar_ax,orientation='horizontal')
cb.set_label('Different Kinetic Energy ($m^{2}s^{-2}$)')

output_path = os.path.join(path_to_plots,'fig14.png')
fig14.savefig(output_path)
plt.close(fig14)

#######################
#######################

fig15, axes15 = plt.subplots(3,1,figsize=(5,7))
initdate = '2011112700'
plotfname = 'DKE_500_'

times = [(2011,m,d,12,0,0) for m,d in zip((11,11,12),(29,30,1))]
runfolder = os.path.join(rootdir,initdate)
path_to_plots = os.path.join(outdir,initdate)
V = range(200,2200,200)
labels = ['a)','b)','c)']

for ax,t,label in zip(axes15.flat, times,labels):
    cf = p.plot_diff_energy('sum_z','kinetic',t,runfolder,'DKE_500_'+initdate,
                                path_to_plots,plotfname,V,no_title=1,ax=ax)
    ax.text(0.05,0.85,label,transform=ax.transAxes,
            bbox={'facecolor':'white'},fontsize=15,zorder=1000)
    
fig15.tight_layout()
fig15.subplots_adjust(bottom=0.15)
cbar_ax = fig15.add_axes([0.15,0.075,0.7,0.025])
cb = fig15.colorbar(cf,cax=cbar_ax,orientation='horizontal')
cb.set_label('Different Kinetic Energy ($m^{2}s^{-2}$)')

output_path = os.path.join(path_to_plots,'fig15.png')
fig15.savefig(output_path)
plt.close(fig15)


#######################
#######################

fig16, axes16 = plt.subplots(3,1,figsize=(5,7))
initdate = '2011112900'
plotfname = 'DKE_500_'

times = [(2011,m,d,12,0,0) for m,d in zip((11,11,12),(29,30,1))]
runfolder = os.path.join(rootdir,initdate)
path_to_plots = os.path.join(outdir,initdate)
V = range(200,2200,200)
labels = ['a)','b)','c)']

for ax,t,label in zip(axes16.flat, times,labels):
    cf = p.plot_diff_energy('sum_z','kinetic',t,runfolder,'DKE_500_'+initdate,
                                path_to_plots,plotfname,V,no_title=1,ax=ax)
    ax.text(0.05,0.85,label,transform=ax.transAxes,
            bbox={'facecolor':'white'},fontsize=15,zorder=1000)
    
fig16.tight_layout()
fig16.subplots_adjust(bottom=0.15)
cbar_ax = fig16.add_axes([0.15,0.075,0.7,0.025])
cb = fig16.colorbar(cf,cax=cbar_ax,orientation='horizontal')
cb.set_label('Different Kinetic Energy ($m^{2}s^{-2}$)')

output_path = os.path.join(path_to_plots,'fig16.png')
fig16.savefig(output_path)
plt.close(fig16)
