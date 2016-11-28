import os
import WEM.utils as utils
import numpy as N
import itertools
import time
import subprocess

# Automatically run idealised jobs on SLURM
# ideal.exe will be quick (<5 sec) and not submitted
# Strip down files to a few variables
# Move them elsewhere before next run

sounddir = '/home/johnlawson/idealised/WK82_profiles_3'
wrfdir = '/home/johnlawson/WRF_v3.8/WRFV3/run'
# wrfoutdir = '/home/johnlawson/idealised/wrfout'
wrfoutdir = '/scratch/johnlawson/wrfout'
ncksdir = '/scratch/johnlawson/idealised'

vrbls = ['U','V','W','T','P','QVAPOR','PH','REFL_10CM','PB','PHB','RAINNC']

def delete_files(*args,**kwargs):
    for fpath in args:
        cmd = 'rm -f {0}'.format(fpath)
        if kwargs['dryrun']:
            print(cmd)
        else:
            os.system('rm -f {0}'.format(fpath))
    return

def ncks(fpath,vrbls):
    vrblstr = ','.join(vrbls)
    # fpath_new = fpath.replace('.nc','_ncks.nc')
    fpath_new = fpath+'_ncks.nc'
    cmnd = 'ncks -v {0} {1} {2}'.format(vrblstr,fpath,fpath_new)
    os.system(cmnd)
    return fpath_new

qv0_range = N.arange(10.0,15.8,0.2)
U_range = N.arange(0.0,52.0,2.0)
# qv0_range = (13.0,)
# U_range = (32.0,)

def gen_fname(q,u):
    qst = q*10
    ust = u*10
    fname = 'profile_{0:03.0f}_{1:03.0f}'.format(qst,ust)
    return fname

import glob
already_done = glob.glob(ncksdir+'/*')

for q,u in itertools.product(qv0_range,U_range):
    # Link sounding for idealised
    fname = gen_fname(q,u)
    ncksname = '{0}.nc'.format(fname)

    if os.path.join(ncksdir,ncksname) in already_done:
        print("Skipping {0}, already done.".format(fname))
        continue

    print("Running WRF for {0}".format(fname))
    soundpath = os.path.join(sounddir,fname)
    insoundpath = os.path.join(wrfdir,'input_sounding')
    linkcmd = 'ln -sf {0} {1}'.format(soundpath,insoundpath)
    linkcmd_p = subprocess.Popen(linkcmd,cwd=wrfdir,shell=True)
    linkcmd_p.wait()
    # os.system(linkcmd)

    # Remove wrfinput 
    wrfinputpath = os.path.join(wrfdir,'wrfinput_d01')
    delcmd = 'rm {0}'.format(wrfinputpath)
    delcmd_p = subprocess.Popen(delcmd,cwd=wrfdir,shell=True)
    delcmd_p.wait()
    
    # Run ideal.exe
    path_to_ideal = os.path.join(wrfdir,'ideal.exe')
    # idealcmd = '{0}'.format(path_to_ideal)
    ideal_p = subprocess.Popen(path_to_ideal, cwd=wrfdir,shell=True)
    ideal_p.wait()
    # os.system(idealcmd)
    # time.sleep(15)

    # Remove rsl files
    rslpaths = os.path.join(wrfdir,'rsl*')
    rslrmcmd = 'rm {0}'.format(rslpaths)
    rsldel_p = subprocess.Popen(rslrmcmd, cwd=wrfdir,shell=True)
    rsldel_p.wait()
    # os.system(rslrmcmd)

    # Submit job
    jobpath = os.path.join(wrfdir,'submit_wrf.job')
    subcmd = 'sbatch {0}'.format(jobpath) 
    os.system(subcmd)

    # Start the timer.
    startsleep = 1
    elapsed = startsleep

    # Wait five minutes then start checking tail file
    time.sleep(60*startsleep)

    path_to_rsl = os.path.join(wrfdir,'rsl.error.0000')
    tailcmd = 'tail {0}'.format(path_to_rsl)
    finished = False
    while not finished:
        tailrsl = subprocess.Popen(tailcmd,shell=True,stdout=subprocess.PIPE)
        tailout = tailrsl.stdout.read()
        if b"SUCCESS COMPLETE WRF" in tailout:
            finished = True
            print("WRF has finished; moving files")
        elif b"Timing for main" in tailout:
            elapsed += 0.5
            if elapsed > 25:
                print("Job seems to have died. Exiting.")
                raise Exception
            else:
                print("WRF still running. Trying again in 30 sec.")
                time.sleep(30)
        else:
            startsleep += 3
            if startsleep > 12*60:
                print("Job has been waiting for 12 hours. Exiting.")
                raise Exception
            else:
                print("Waiting for job to start running. Trying again in 3 min.")
                time.sleep(180)

    # Do ncks
    fpath_nc = os.path.join(wrfoutdir,'wrfout_ideal')
    # fpath_nc = os.path.join(wrfdir,'wrfout_d01_2000-01-01_00:00:00')
    fpath_ncks = ncks(fpath_nc,vrbls)
    delete_files(fpath_nc,dryrun=False)

    # exdir = fname
    # movedir = os.path.join(wrfoutdir,exdir)
    movefpath = os.path.join(ncksdir,ncksname)
    # utils.trycreate(movedir)
    mvcmd = 'mv {0} {1}'.format(fpath_ncks,movefpath)
    mvcmd_p = subprocess.Popen(mvcmd,shell=True)
    mvcmd_p.wait()
    # os.system(mvcmd)

    # Remove old rsl files
    rslpaths = os.path.join(wrfdir,'rsl*')
    rslrmcmd = 'rm {0}'.format(rslpaths)
    lastrmrls_p = subprocess.Popen(rslrmcmd,shell=True)
    lastrmrls_p.wait()
    # os.system(rslrmcmd)

    # Move to scratch?
    # pdb.set_trace()
