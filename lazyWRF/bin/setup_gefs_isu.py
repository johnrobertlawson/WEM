# Create met_em files looping over ensemble members
# Moves output between each 
# Assumes Vtables for GEFS data are from Lawson .pdf README file

# First, manually check namelist.wps
# Then remove met_em and GRIB files not related to this ensemble
# Keep the GRIB soft links if halfway through GEFS ensemble run
# Then run this script

import os
import sys
import pdb
import glob
import subprocess
from copyfilescript import copyfiles
import time

linkonly = 0 # Don't submit jobs, just link met_em files

# FUNCTIONS
def edit_namelist(old,new):
    nwps = open('namelist.wps','r').readlines()
    for idx, line in enumerate(nwps):
        if old in line:
            # Prefix for soil intermediate data filename
            nwps[idx] = new + " \n"
            nameout = open('namelist.wps','w')
            nameout.writelines(nwps)
            nameout.close()
            break

def get_datestring():
    # Then it opens the WPS namelist, copies the old one
    os.system('cp namelist.wps{,.python_backup}')
    nwps = open('namelist.wps','r').readlines()
    # Get the GEFS initialisation date from here
    for line in nwps:
        if "start_date" in line:
            datestring = line[15:19] + line[20:22] + line[23:25]
            break
    return datestring

def gefs_soil():
    # Find intermediate filename line in WPS namelist
    # Link to, and ungrib, soil files if they don't exist
    if not glob.glob('FILE_SOIL*'):
        edit_namelist("prefix"," prefix = 'FILE_SOIL'")
        os.system('./link_grib.csh ' + pathtosoildata)
        os.system('ln -sf ungrib/Variable_Tables/Vtable.GFS_soilonly Vtable')
        os.system('./ungrib.exe')

def gefs_atmos(i,nextens):
    datestring = get_datestring()
    print datestring
    # Set path to gefs data
    pathtogefsdata = gefsdir+datestring+'_'+nextens+'_f*'
        
    # Now change the namelist.wps to use the atmospheric data prefix
    #pdb.set_trace()
    edit_namelist("prefix"," prefix = 'FILE_ATMOS'")
    
    if i:
        os.system('rm GRIB* FILE_ATMOS* met_em.d*')
    os.system('./link_grib.csh ' + pathtogefsdata)
    os.system('ln -sf ungrib/Variable_Tables/Vtable.GEFSR2 Vtable')
    os.system('./ungrib.exe')

    # Combine both GEFS atmos and GFS soil data
    edit_namelist("fg_name"," fg_name = 'FILE_SOIL','FILE_ATMOS'")
    os.system('./metgrid.exe')

def submit_job(linkonly=0):
    # Soft link data netCDFs files from WPS to WRF
    os.system('ln -sf ' + pathtoWPS + 'met_em* ' + pathtoWRF)
    if not linkonly:
        p_real = subprocess.Popen('qsub -d '+pathtoWRF+' real_run.sh',cwd=pathtoWRF,shell=True,stdout=subprocess.PIPE)
        p_real.wait()
        jobid = p_real.stdout.read()[:5] # Assuming first five digits = job ID.
        # Run WRF but wait until Real has finished without errors
        print 'Now submitting wrf.exe.'
        # Again, change name of submisGsion script if needed
        p_wrf = subprocess.Popen('qsub -d '+pathtoWRF+' wrf_run.sh -W depend=afterok:'+jobid,cwd=pathtoWRF,shell=True)
        p_wrf.wait()

        time.sleep(1*60*60) # Wait an hour.
        # Makes sure real.exe has finished and wrf.exe is writing to rsl files

        finished = 0
        while not finished:
            tailrsl = subprocess.Popen('tail '+pathtoWRF+'rsl.error.0000',shell=True,stdout=subprocess.PIPE)
            tailoutput = tailrsl.stdout.read()
            if "SUCCESS COMPLETE WRF" in tailoutput:
                finished = 1
                print "WRF has finished; moving to next case."
            else:
                time.sleep(5*60) # Try again in 5 min

# SETTINGS
# Directory with GEFS data
datestr = '20060526'
pathtoWPS = '/ptmp/jrlawson/WPS/'
gefsdir = './gefsfiles/' + datestr + '/'
pathtosoildata = './gfsfiles/'+datestr+'/gfsanl*'
pathtoWRF = '/ptmp/jrlawson/WRFV3/run/'
pathtometem = '/ptmp/jrlawson/wrfout/bowecho/'
ensemblelist = ['p'+ '%02u' %n for n in range(1,11)] 
#ensemblelist = ['c00'] + ['p'+ '%02u' %n for n in range(1,11)] 

for i,nextens in enumerate(ensemblelist):
    if i==0:
        os.system('./geogrid.exe')
        gefs_soil()
    gefs_atmos(i,nextens)
    submit_job(linkonly=linkonly)
    if not linkonly:
        copyfiles(pathtoWRF,pathtometem,nextens)
