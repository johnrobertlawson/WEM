# This script goes through post-processing of WRF
# And submission of the job to a Rocks cluster if desired (edit to taste) 
# Run it ("python lazyWRF.py") from your WPS folder.

# John Lawson, Iowa State University 2013
# Email john.rob.lawson@googlemail.com

# -------------IMPORTANT NOTES---------------
# Remove met_em files if you don't want them overwritten (or hide them)
# Make sure all SETTINGS are correct
# Then run this script
# If job submission is switched on, make sure namelist.input parameterisations are correct
# This script will sync all namelist.wps settings with namelist.output

##########################
# IMPORTS
import os
import sys
import pdb
import glob
import subprocess
import calendar
import math
import datetime
###########################

# Script SETTINGS
# If switched on, existing wrfout files etc will be moved to a folder
move_wrfout = 1
pathtowrfout = './'
# If switched on, the script will submit jobs (edit as needed)
submit_job = 1
pathtoWRF = '../WRFV3/run/'

# WRF run SETTINGS
# Start,end date in YYYYMMDD_HHMM
idate = '20060526_0000'
fdate = '20060527_1200'
domains = 1
e_we = (500,) # Needs to be same length as number of domains
e_sn = (500,)
dx = 3 # In km; grid spacing of largest domain
dy = 3 # Ditto
i_start = (1,) # Same length as num of domains
j_start = (1,)
parent_grid_ratio = (1,) # Same length as num of domains; ratio of each parent to its child

# Select initial data source
# gfs, nam, gefs
init_data = 'gfs'
# Directory with initialisation data
pathtoinitdata = './gfsfiles/gfsanl'
# Intermediate file prefix
prefix = "FILE"

############################

# FUNCTIONS
def edit_namelist(old,new,incolumn=1):
    nwps = open('namelist.wps','r').readlines()
    for idx, line in enumerate(nwps):
        if old in line:
            # Prefix for soil intermediate data filename
            if incolumn==1:
                nwps[idx] = nwps[idx][:23] + new + " \n"
            else:
                nwps[idx] = ' ' + old + ' = ' + new + "\n"
            nameout = open('namelist.wps','w')
            nameout.writelines(nwps)
            nameout.close()
            break

def edit_namelist_input(old,new,pathtoWRF=pathtoWRF):
    ninput = open(pathtoWRF+'namelist.input','r').readlines()
    for idx, line in enumerate(ninput):
        if old in line:
            # Prefix for soil intermediate data filename
            ninput[idx]= ninput[idx][:39] + new + " \n"
            nameout = open(pathtoWRF+'namelist.input','w')
            nameout.writelines(ninput)
            nameout.close()
            break

############################
############################

# Open the WPS namelist; copy the old one in case of bugs
os.system('cp namelist.wps{,.python_backup}')
nwps = open('namelist.wps','r').readlines()

# Sets values depending on initialisation data
if init_data == 'gfs':
    atmos_levs = 27
    soil_levs = 4
    interval = 6 # In hours
    Vtable_suffix = 'GFS'
elif init_data == 'nam':
    atmos_levs = 40
    soil_levs = 4
    interval = 6 # In hours
    Vtable_suffix = 'NAM'
elif init_data == 'gefs':
    ens = raw_input('Which ensemble member? (c00,p01...p10) ')
    atmos_levs = 12
    soil_levs = 4 # Uses GFS soil data
    interval = 3 # Interpolation happens after 90 h
    Vtable_suffix = 'GEFSR2'

# Prepares namelist.wps
idate_s = "'"+idate[:4]+"-"+idate[4:6]+"-"+idate[6:11]+":"+idate[11:13]+":00',"
edit_namelist("start_date",idate_s * domains, incolumn=0)
fdate_s = "'"+fdate[:4]+"-"+fdate[4:6]+"-"+fdate[6:11]+":"+fdate[11:13]+":00',"
edit_namelist("end_date",fdate_s * domains, incolumn=0)
edit_namelist("max_dom",str(domains)+',',incolumn=0)
edit_namelist("interval_seconds",str(interval*3600)+',', incolumn=0)
edit_namelist("parent_grid_ratio",', '.join([str(p) for p in parent_grid_ratio])+',')
edit_namelist("i_parent_start", ', '.join([str(i) for i in i_start])+',')
edit_namelist("j_parent_start", ', '.join([str(j) for j in j_start])+',')
edit_namelist("dx",str(dx*1000),incolumn=0)
edit_namelist("dy",str(dy*1000),incolumn=0)
edit_namelist("e_we",', '.join([str(w) for w in e_we])+',')
edit_namelist("e_sn",', '.join([str(s) for s in e_sn])+',')
edit_namelist("prefix",prefix+',',incolumn=0)
edit_namelist("fg_name",prefix+',',incolumn=0)

# Add your own here if wanting to change e.g. domain location, dx, dy from here...    

# Link to, and ungrib, initialisation files
os.system('./link_grib.csh ' + pathtoinitdata)
os.system('ln -sf ungrib/Variable_Tables/Vtable.' + Vtable_suffix + ' Vtable')
os.system('./ungrib.exe')
os.system('./metgrid.exe')

# Submit jobs (edit as needed)
if submit_job == 1:
    # Soft link data netCDFs files from WPS to WRF
    os.system('ln -sf ./met_em* ' + pathtoWRF)
    
    ##### Sync namelist.input with namelist.wps
    # Copy original in case of bugs
    os.system('cp ' + pathtoWRF + 'namelist.input{,.python_backup}')
    # Compute run time
    dt_1 = calendar.timegm((int(idate[0:4]),int(idate[4:6]),int(idate[6:8]),
                            int(idate[9:11]),int(idate[11:13]),0))
    dt_2 = calendar.timegm((int(fdate[0:4]),int(fdate[4:6]),int(fdate[6:8]),
                            int(fdate[9:11]),int(fdate[11:13]),0))
    dt = datetime.timedelta(seconds=(dt_2 - dt_1))
    days = dt.days
    hrs = math.floor(dt.seconds/3600.0)
    mins = ((dt.seconds/3600.0) - hrs) * 60.0
    secs = 0 # Assumed!

    # Compute dx,dy for each domain
    dxs = [dx*1000]
    if domains != 1:
        for idx in range(1,domains):
            child_dx = dxs[idx-1] * 1.0/parent_grid_ratio[idx]
            dxs.append(child_dx)
    dys = [dy*1000]
    if domains != 1:
        for idx in range(1,domains):
            child_dy = dys[idx-1] * 1.0/parent_grid_ratio[idx]
            dys.append(child_dy)

    # If all namelist values begin on column 38:
    edit_namelist_input("run_days","%01u" %days)
    edit_namelist_input("run_hours","%01u" %hrs)
    edit_namelist_input("run_minutes","%01u" %mins)
    edit_namelist_input("run_seconds","%01u" %secs)
    edit_namelist_input("start_year", (idate[:4]+', ')*domains)
    edit_namelist_input("start_month", (idate[4:6]+', ')*domains)
    edit_namelist_input("start_day", (idate[6:8]+', ')*domains)
    edit_namelist_input("start_hour", (idate[9:11]+', ')*domains)
    edit_namelist_input("start_minute", (idate[11:13]+', ')*domains)
    edit_namelist_input("start_second", ('00, ')*domains)
    edit_namelist_input("end_year", (fdate[:4]+', ')*domains)
    edit_namelist_input("end_month", (fdate[4:6]+', ')*domains)
    edit_namelist_input("end_day", (fdate[6:8]+', ')*domains)
    edit_namelist_input("end_hour", (fdate[9:11]+', ')*domains)
    edit_namelist_input("end_minute", (fdate[11:13]+', ')*domains)
    edit_namelist_input("end_second", ('00, ')*domains)
    edit_namelist_input("interval_seconds", str(interval*3600)+',')
    edit_namelist_input("max_dom",str(domains)+',')
    edit_namelist_input("e_we", ', '.join([str(w) for w in e_we])+',')
    edit_namelist_input("e_sn", ', '.join([str(s) for s in e_sn])+',')
    edit_namelist_input("num_metgrid_levels", str(atmos_levs)+',')
    edit_namelist_input("dx", ', '.join([str(d) for d in dxs])+',')
    edit_namelist_input("dy", ', '.join([str(d) for d in dys])+',')
    edit_namelist_input("i_parent_start", ', '.join([str(i) for i in i_start])+',')
    edit_namelist_input("j_parent_start", ', '.join([str(j) for j in j_start])+',')
    edit_namelist_input("parent_grid_ratio",', '.join([str(p) for p in parent_grid_ratio])+',') 

    """
    # Run real, get ID number of job
    # Change name of submission script if needed
    p_real = subprocess.call([qsub,real_run.sh],cwd=pathtoWRF,shell=True,
                            stdout=subprocess.PIPE)
    jobid = p.stdout.read()[:5] # Assuming last five digits = job ID.
    # Run WRF but wait until Real has finished without errors
    # Again, change name of submission script if needed
    p_wrf = subprocess.call([qsub,wrf_run.sh,-W,"afterok=jobid"],
                            cwd=pathtoWRF,shell=True)
    print "real.exe and wrf.exe submitted. Exiting Python script."
    """
else:
    print "Pre-processing complete. Exiting Python script."

