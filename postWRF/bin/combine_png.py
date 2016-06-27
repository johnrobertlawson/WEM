# Import PNGs from each ensemble member at a certain time
import sys
sys.path.append('/home/jrlawson/gitprojects/')

import matplotlib as M
M.use('Agg')
import matplotlib.pyplot as plt
import numpy as N
import pdb
import glob
import os

import WEM.utils as utils
from settings import Settings

#M.rc['DPI'] = 400.0
height, width = (15,10)

def download_radar(radar_view,time_str,verifrootdir):
    d = time_str[:8]
    t = time_str[8:]
    gif = '{0}_{1}{2}.gif'.format(radar_view,d,t)
    png = '{0}_{1}{2}.png'.format(radar_view,d,t)   
    gifpath = os.path.join(verifrootdir,gif)
    pngpath = os.path.join(verifrootdir,png)
    # Don't download if .gif exists, just convert.
    if glob.glob(gifpath):
       print("Radar image exists: converting existing .gif.") 
    else:
        URL = ('http://www.mmm.ucar.edu/imagearchive1/RadarComposites/'
              '{0}/{1}/{2}'.format(radar_view,d,gif))
        getimage = 'wget -P {0} {1}'.format(verifrootdir,URL)
        os.system(getimage)
        
        print("Radar image downloaded.")

    convert = 'convert {0} {1}'.format(gifpath,pngpath)
    os.system(convert)
    print("Radar image converted to .png.")
    return pngpath

def get_times(itime,ftime,int_hr):
    times = utils.generate_times(itime,ftime,60*60*int_hr)
    #times = [itime,]
    time_strs = []
    for t in times:
        time_strs.append(utils.string_from_time('output',t))
    return time_strs

config = Settings()

wy = eval(input('Which year: '))

IC = input('ICs? (GEFSR2/NAM/GFS): ')

ex = input('MXMP? ICBC? STCH? STMX?: ')

if ex == 'MXMP':
    experiments = ['WSM6_Grau','WSM6_Hail','Kessler','Ferrier','WSM5',
                   'WDM5','Lin','WDM6_Grau','WDM6_Hail','Morrison_Grau',
                   'Morrison_Hail']
    if IC == 'GEFSR2': 
        ens = input('Which ensemble member? (c00, p01 etc): ')
    else:
        ens = 'anl'

elif ex == 'STMX':
    experiments = ['WSM6_Grau','WSM6_Hail','Kessler','Ferrier','WSM5',
                   'WDM5','Lin','WDM6_Grau','WDM6_Hail','Morrison_Grau',
                   'Morrison_Hail']
    experiments = [e+'_STCH' for e in experiments]

    if IC == 'GEFSR2': 
        ens = input('Which ensemble member? (c00, p01 etc): ')
    else:
        ens = 'anl'

elif ex == 'ICBC':
    ens = ['c00'] + ['p'+"%02d" %n for n in range(1,11)]
    experiments = 'ICBC'

elif ex == 'STCH':
    experiments = ['s'+"%02d" %n for n in range(1,11)]

    MP = input('Which control microphysics scheme? (Ferrier, Morrison_Grau, etc) ')
 
    if IC == 'GEFSR2':
        ens = input('Which ensemble member? (c00, p01 etc): ')
    else:
        ens = 'anl'
 
plot = input('variable to plot? (cref/shear03/shear06/strongestwind): ')
lv_str = input('what level? ')

if wy == 2006:
    case_str = '20060526'
    itime = (2006,5,26,3,0,0)
    ftime = (2006,5,27,12,0,0)
    int_hr = 3
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
    itime = (2013,8,15,3,0,0)
    ftime = (2013,8,16,12,0,0)
    if plot == 'streamlines':
        int_hr = 2
    else:
        int_hr = 3


if wy == 2006 or wy == 2013:
    radar_view = 'cent_plains'
elif wy == 2009:
    radar_view = 'north_plains'
elif wy == 2011:
    radar_view = 'upper_missvly'

if plot == 'cref':
    va_str = 'cref'
    lv_str = 'sfc'


elif plot == 'shear03':
    va_str = 'shear'
    lv_str = '0to3'

elif plot == 'shear06':
    va_str = 'shear'
    lv_str = '0to6'
    times = utils.generate_times(itime,ftime,60*60*1)

elif plot == 'strongestwind':
    va_str = 'strongestwind'
    duration = input("What duration? ")
    # lv_str = ''

elif plot == 'streamlines':
    va_str = 'streamlines'
    lv_str = 'sfc' 

else:
    va_str = plot
    # lv_str = 'sfc'

if plot != 'strongestwind':
    time_strs = get_times(itime,ftime,int_hr)
else:
    time_strs = [duration,] 

#times = utils.generate_times(itime,ftime,60*60*int_hr)
verifplot = 1

"""
if ensembletype == 'CTRL':
    ensnames = ['c00'] + ['p'+ '%02u' %n for n in range(1,11)]
elif ensembletype == 'STCH':
    ensnames = ['s' + '%02u' %n for n in range(1,11)]
elif ensembletype == 'MXMP':
    ensnames = ['WSM6_Grau','WSM6_Hail','Kessler','Ferrier','WSM5','WDM5','Lin','WDM6_Grau','WDM6_Hail','Morrison_Grau','Morrison_Hail']
elif ensembletype == 'STCH_MXMP':
    ensnames = ['WSM6_Grau_STCH','WSM6_Hail_STCH','Kessler_STCH','Ferrier_STCH','WSM5_STCH','WDM5_STCH','Lin_STCH','WDM6_Grau_STCH','WDM6_Hail_STCH','Morrison_Grau_STCH','Morrison_Hail_STCH']
"""

#fig = plt.figure()
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
        fname = '_'.join((va_str,time_str,lv_str)) + '.png'
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
            if va_str:
            #if va_str == 'cref':
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
        # import pdb; pdb.set_trace()
        plt.title(panel)

    # pdb.set_trace()
    if ex == 'STCH':
        outfiledir = os.path.join(config.output_root,case_str,IC,ens,MP)
    elif ex == 'ICBC':
        outfiledir = os.path.join(config.output_root,case_str,IC)
    else:
        outfiledir = os.path.join(config.output_root,case_str,IC,ens)
            
    outfilename = '_'.join(('postage',ex,va_str,lv_str,time_str)) + '.png'
    outfilepath = os.path.join(outfiledir,outfilename)
    print(outfilepath)
    utils.trycreate(outfiledir)
    #plt.tight_layout()
    fig.savefig(outfilepath)
    fig.clf()
