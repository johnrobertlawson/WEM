"""Create postage stamps of ensemble members.

Two views to choose:
A4/Letter publication-ready
OR
To fill a laptop screen.

"""

import sys
sys.path.append('/home/jrlawson/gitprojects/')

import matplotlib as M
M.use('Agg')
import matplotlib.pyplot as plt
import numpy as N
import pdb
import glob
import os

import WEM.utils.utils as utils
from settings import Settings

# SETTINGS
# Portrait or landscape.
portrait = 0
case = 20110419
IC = 'GEFSR2'
experiment = 'MXMP'
ICens = 'p04'
vrbl = 'cref'
MP = 'Thompson' # For SKEB

verifplot = 1 # Include verification

# FUNCTIONS
def plot_map():
	pass

# START SCRIPT
# Assign values
wy = case[:4]
if wy == 2006:
    case_str = '20060526'
    itime = (2006,5,26,0,0,0)
    ftime = (2006,5,27,12,0,0)
    int_hr = 1
elif wy == 2009:
    case_str = '20090910'
    itime = (2009,9,10,23,0,0)
    ftime = (2009,9,11,14,0,0)
    int_hr = 1
elif wy == 2011:
    case_str = '20110419'
    itime = (2011,4,19,18,0,0)
    ftime = (2011,4,20,10,30,0)
    int_hr = 1
elif wy == 2013:
    case_str = '20130815'
    itime = (2013,8,15,0,0,0)
    ftime = (2013,8,16,13,0,0)

if plot == 'cref':
    va_str = 'cref'
    lv_str = 'sfc'

    if wy == 2006 or wy == 2013:
        radar_view = 'cent_plains'
    elif wy == 2009:
        radar_view = 'north_plains'
    elif wy == 2011:
        radar_view = 'upper_missvly'

elif plot == 'shear03':
    va_str = 'shear'
    lv_str = '0to3'

elif plot == 'shear06':
    va_str = 'shear'
    lv_str = '0to6'
    times = utils.generate_times(itime,ftime,60*60*1)

elif plot == 'maxwind':
    va_str = 'strongestwind'
    lv_str = 'sfc'

elif plot == 'streamlines':
    va_str = 'streamlines'
    lv_str = 'sfc' 

if plot != 'maxwind':
    time_strs = get_times(itime,ftime,int_hr)
else:
    time_strs = ['range',] 

if plot == 'streamlines':
    int_hr = 2
else:
    int_hr = 1

fig = plt.figure(figsize=(width,height))

verifrootdir = os.path.join(config.output_root,case_str,'VERIF')
RUCrootdir = os.path.join(config.output_root,case_str,'RUC','anl','VERIF')

if ex == 'ICBC':
    panellist = ['verif'] + ens
elif ex == 'STCH':
    panellist = ['verif',MP] + experiments
else:
    panellist = ['verif','ICBC'] + experiments

if not verifplot:
    panellist.pop(0)

if len(panellist) < 13:
    rows, cols = (4,3) 
else:
    rows, cols = (5,3)
    height += 1

for time_str in time_strs:
    print(('Creating image for time {0}'.format(time_str)))
    for p,panel in enumerate(panellist):
        # caserootdir = os.path.join(config.output_root,case_str,IC)
        fname = '_'.join((va_str,lv_str,time_str)) + '.png'
        if ex == 'ICBC':
            #pdb.set_trace()
            froot = os.path.join(config.output_root,case_str,IC,panel,'ICBC')
        elif ex == 'STCH':
            if panel == MP:
                froot = os.path.join(config.output_root,case_str,IC,ens,MP)
            else:
                froot = os.path.join(config.output_root,case_str,IC,ens,MP,panel)
        else:
            froot = os.path.join(config.output_root,case_str,IC,ens,panel)
        fpath = os.path.join(froot,fname)

        # Load image
        p += 1
        ax = fig.add_subplot(rows,cols,p)
        if p == 1 and verifplot:
            if va_str == 'cref':
                verif_fname = '_'.join((radar_view,time_str+'.png'))
                verif_fpath = os.path.join(verifrootdir,verif_fname)
                veriffiles = glob.glob(verif_fpath)
                if veriffiles:
                    img = M.image.imread(veriffiles[0])
                    ax.imshow(img)
                else:
                    veriffile = download_radar(radar_view,time_str,verifrootdir)
                    fileexists = glob.glob(verif_fpath)
                    #pdb.set_trace()
                    if fileexists:
                        img = M.image.imread(veriffile)
                        ax.imshow(img)
                    else:
                        print("Verif skipped")
            elif va_str == 'streamlines':
                RUCplots = glob.glob(RUCrootdir+'/*')
                fname = 0
                for p in RUCplots:
                    if time_str in p:
                        fname = p
                        break
                if fname:
                    img = M.image.imread(fname)
                    ax.imshow(img)
                else:
                    print("RUC file not there")
            else:
                print("Verif skipped")
        elif p == 1 and not verifplot and ex is not 'ICBC':
            img = M.image.imread(fpath)
            ax.imshow(img)
        elif (p == 2) & (ex != 'ICBC'):
            img = M.image.imread(fpath)
            ax.imshow(img)
        else:
            try:
                img = M.image.imread(fpath)
            except:
                print("Skipping ", panel)
            else:
                print("Plotting ", panel)
                ax.imshow(img)
        ax.axis('off')
        plt.title(panel)

    #pdb.set_trace()
    if ex == 'STCH':
        outfiledir = os.path.join(config.output_root,case_str,IC,ens,MP)
    else:
        outfiledir = os.path.join(config.output_root,case_str,IC,ens)
            
    outfilename = '_'.join(('postage',ex,va_str,lv_str,time_str)) + '.png'
    outfilepath = os.path.join(outfiledir,outfilename)
    print(outfilepath)
    utils.trycreate(outfiledir)
    #plt.tight_layout()
    fig.savefig(outfilepath)
    fig.clf()