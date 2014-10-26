# This script goes through processing of WPS and WRF (either, or both)
# And submission of the job to a Rocks cluster if desired (edit to taste) 
# Run it ("python lazyWRF.py") from your WPS folder.

# John Lawson, Iowa State University 2013
# Email john.rob.lawson@googlemail.com

# -------------IMPORTANT NOTES---------------
# Remove met_em files if you don't want them overwritten (or hide them)
# Make sure all SETTINGS are correct
# Make sure you've read all the comments down to "Edit above here"
# Make sure you've set other settings in both namelists that aren't covered in this script
# (This script won't remove any bogus lines in either namelist, syntax errors etc)
# Then run this script
# If job submission is switched on, make sure namelist.input parameterisations are correct
# This script will sync all namelist.wps settings with namelist.output
# Submit issues and requests to the GitHub (https://github.com/johnrobertlawson/lazyWRF)

# IMPORTS
import os
import sys
import pdb
import glob
import subprocess
import calendar
import math
import datetime
import time

######################################
### EDIT BELOW HERE ##################
######################################

##### Script SETTINGS #####
# If switched on, this will do pre-processing (WPS)
WPS = 1
# If switched on, this will do WRF processing
WRF = 1
# If switched on, the script will submit jobs (edit as needed)
submit_job = 1

# If switched on, existing wrfout files etc will be moved to a folder
move_wrfout = 0 # WORK IN PROGRESS

# Path to WRF, WPS folders (absolute) - end in a slash!
pathtoWPS = '/ptmp/jrlawson/WPS/'
pathtoWRF = '/ptmp/jrlawson/WRFV3/run/'
# Path to move wrfout* files - end in a slash!
pathtowrfout = '/ptmp/jrlawson/home/path/to/store/wrfout/'

# If you want error messages sent to an email, fill it here as a string; otherwise put '0'.
# email = 'yourname@domain.edu'
email = 0

##### WRF run SETTINGS #####
# Start and end date in (YYYY,MM,DD,H,M,S)
idate = (2006,05,26,0,0,0)
interval = 6.0 # In hours, as a float
fdate = (2006,05,27,12,0,0)
domains = 1
e_we = (500,) # Needs to be same length as number of domains
e_sn = (500,)
dx = 3 # In km; grid spacing of largest domain
dy = 3 # Ditto
i_start = (1,) # Same length as num of domains
j_start = (1,)
parent_grid_ratio = (1,) # Same length as num of domains; ratio of each parent to its child

# Select initial data source
# 'gfs' = GFS analyses
# 'nam' = NAM analyses
init_data = 'gfs'
# Directory with initialisation data (absolute, or relative to WPS) - end in a slash!
pathtoinitdata = './gfsfiles/'
# Intermediate file prefix (usually no need to change)
int_prefix = "FILE"

### NOTE:
# Any settings you want to change that aren't in this box (e.g. time step),
# you need to manually change yourself in its relevant namelist.
# Submit a github request if you think it can be automated/is commonly changed

######################################
### EDIT ABOVE HERE ##################
######################################

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
    return

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
    return

def str_from_date(date,format):
    # date = Input is tuple (yyyy,mm,dd,h,m,s)
    # format = choose 'list' or 'indiv' or...
    # ... year, month, day, hour, minute, second output
    datelist = []
    for n in date:
        datelist.append("%02u" %n)
    if format=='list':
        return datelist
    else:
        year,month,day,hour,minute,second = datelist
        if format=='indiv':
            return year,month,day,hour,minute,second
        else:
            val = eval(format)
            return val

def download_data(date,initdata,pathtoinitdata):
    # Download gfs, nam data from server for a timestamp
    if initdata == 'gfs':
        prefix = 'gfsanl'
        n = '3'
        suffix = '.grb'
    elif initdata == 'nam':
        prefix = 'namanl'
        n = '218'
        suffix = '.grb'
    command = ('wget "http://nomads.ncdc.noaa.gov/data/' + prefix + '/' + date[:6] + '/' + date[:8] + 
                '/' + prefix + '_' + n + '_' + date + '_000' + suffix + '" -P ' + pathtoinitdata)
    os.system(command)
    return

def get_init_files(initdata,idate,interval,fdate,pathtoinitdata):
    # Let's assume all initdata possibilities start at 0000 UTC and the interval is an integer factor
    if initdata=='gfs':
        prefix = 'gfsanl'
    elif initdata=='nam':
        prefix = 'namanl'

    # First, convert dates to seconds-from-epoch time
    idaten = calendar.timegm(idate)
    fdaten = calendar.timegm(fdate)
    
    # The initial and final files needed:
    ifiledate = idaten - idaten%(interval*3600)
    ffiledate = fdaten - fdaten%(interval*3600) + (interval*3600)

    # Create a range of required file dates
    required_dates = range(int(ifiledate),int(ffiledate),int(interval*3600))

    # List all files in initialisation data folder
    initfiles = glob.glob(pathtoinitdata + '*')

    # Loop through times and check to see if file exists for initialisation model
    for r in required_dates:
        # Tuple of date (9 items long starting with year, month, etc...)
        longdate = time.gmtime(r)
        fname_date = ''.join(["%02u" %x for x in longdate[0:3]]) + '_' + ''.join(["%02u" %x for x in longdate[3:5]])
        checkfiles_prefix = []
        checkfiles_date = []
        for f in initfiles:
            checkfiles_prefix.append(prefix in f)
            checkfiles_date.append(fname_date in f)
        try:
            checkfiles_prefix.index(1) + checkfiles_date.index(1)
        except ValueError:
            print 'Downloading required file for ' + fname_date
            download_data(fname_date,init_data,pathtoinitdata)
        else:     
            print 'Data for ' + fname_date + ' already exists.'
    return

# This function runs a script and checks for errors; raises exception if one exists
def intelligent_run(executable,email):
    # email = if you want email sent to an address, fill it here
    command = './' + executable + '.exe'
    os.system(command)
    logfile = open(executable + '.log').readlines() 
    if "Successful completion" in logfile[-1]:
        print '>>>>>>>> ' , executable, "has completed successfully. <<<<<<<<"
    else:
        print '!!!!!!!! ' , executable, "has failed. Exiting... !!!!!!!!"
        if email:
            os.system('tail '+logfile+' | mail -s "lazyWRF message: error in '+executable+'." '+email)
        raise Exception
    return

###############################
#### BEGINNING OF CODE ########
###############################

# Open the WPS namelist; copy the old one in case of bugs
os.system('cp namelist.wps{,.python_backup}')
nwps = open('namelist.wps','r').readlines()

# Sets values depending on initialisation data
if init_data == 'gfs':
    atmos_levs = 27
    soil_levs = 4
    Vtable_suffix = 'GFS'
    init_prefix = 'gfsanl'
elif init_data == 'nam':
    atmos_levs = 40
    soil_levs = 4
    Vtable_suffix = 'NAM'
    init_prefix = 'namanl'

# Get nice strings for namelist writing
y1,mth1,d1,h1,min1,s1 = str_from_date(idate,'indiv')
y2,mth2,d2,h2,min2,s2 = str_from_date(fdate,'indiv')

if WPS:
    # Prepares namelist.wps
    idate_s = "'"+y1+"-"+mth1+"-"+d1+'_'+h1+':'+min1+":"+s1+"',"
    edit_namelist("start_date",idate_s * domains, incolumn=0)
    fdate_s = "'"+y2+"-"+mth2+"-"+d2+'_'+h2+':'+min2+":"+s2+"',"
    edit_namelist("end_date",fdate_s * domains, incolumn=0)
    edit_namelist("max_dom",str(domains)+',',incolumn=0)
    edit_namelist("interval_seconds",str(int(interval*3600))+',', incolumn=0)
    edit_namelist("parent_grid_ratio",', '.join([str(p) for p in parent_grid_ratio])+',')
    edit_namelist("i_parent_start", ', '.join([str(i) for i in i_start])+',')
    edit_namelist("j_parent_start", ', '.join([str(j) for j in j_start])+',')
    edit_namelist("dx",str(dx*1000),incolumn=0)
    edit_namelist("dy",str(dy*1000),incolumn=0)
    edit_namelist("e_we",', '.join([str(w) for w in e_we])+',')
    edit_namelist("e_sn",', '.join([str(s) for s in e_sn])+',')
    edit_namelist("prefix",int_prefix+',',incolumn=0)
    edit_namelist("fg_name",int_prefix+',',incolumn=0)

    # Add your own here if wanting to change e.g. domain location, dx, dy from here...    

    # Run geogrid
    intelligent_run('geogrid',email)

    # Check to see if initialisation files exist
    # If they don't, download into data directory
    get_init_files(init_data,idate,interval,fdate,pathtoinitdata)

    # Link to, and ungrib, initialisation files
    os.system('./link_grib.csh ' + pathtoinitdata + init_prefix)
    os.system('ln -sf ungrib/Variable_Tables/Vtable.' + Vtable_suffix + ' Vtable')
    intelligent_run('ungrib',email)
    intelligent_run('metgrid',email)

# Submit jobs (edit as needed)
if WRF:
    # Soft link data netCDFs files from WPS to WRF
    os.system('ln -sf ' + pathtoWPS + 'met_em* ' + pathtoWRF)
    
    ##### Sync namelist.input with namelist.wps
    # Copy original in case of bugs
    os.system('cp ' + pathtoWRF + 'namelist.input{,.python_backup}')
    
    print 'met_em* linked. Now amending namelist.input.'

    # Compute run time
    dt_1 = calendar.timegm(idate)
    dt_2 = calendar.timegm(fdate)
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
    edit_namelist_input("run_days","%01u" %days + ',')
    edit_namelist_input("run_hours","%01u" %hrs + ',')
    edit_namelist_input("run_minutes","%01u" %mins + ',')
    edit_namelist_input("run_seconds","%01u" %secs + ',')
    edit_namelist_input("start_year", (y1+', ')*domains)
    edit_namelist_input("start_month", (mth1+', ')*domains)
    edit_namelist_input("start_day", (d1+', ')*domains)
    edit_namelist_input("start_hour", (h1+', ')*domains)
    edit_namelist_input("start_minute", (min1+', ')*domains)
    edit_namelist_input("start_second", (s1+', ')*domains)
    edit_namelist_input("end_year", (y2+', ')*domains)
    edit_namelist_input("end_month", (mth2+', ')*domains)
    edit_namelist_input("end_day", (d2+', ')*domains)
    edit_namelist_input("end_hour", (h2+', ')*domains)
    edit_namelist_input("end_minute", (min2+', ')*domains)
    edit_namelist_input("end_second", (s2+', ')*domains)
    edit_namelist_input("interval_seconds", str(int(interval*3600))+',')
    edit_namelist_input("max_dom",str(domains)+',')
    edit_namelist_input("e_we", ', '.join([str(w) for w in e_we])+',')
    edit_namelist_input("e_sn", ', '.join([str(s) for s in e_sn])+',')
    edit_namelist_input("num_metgrid_levels", str(atmos_levs)+',')
    edit_namelist_input("dx", ', '.join([str(d) for d in dxs])+',')
    edit_namelist_input("dy", ', '.join([str(d) for d in dys])+',')
    edit_namelist_input("i_parent_start", ', '.join([str(i) for i in i_start])+',')
    edit_namelist_input("j_parent_start", ', '.join([str(j) for j in j_start])+',')
    edit_namelist_input("parent_grid_ratio",', '.join([str(p) for p in parent_grid_ratio])+',') 

    
    if submit_job:
        print 'Namelist edited. Now submitting real.exe.'  
        # Run real, get ID number of job
        # Change name of submission script if needed
        p_real = subprocess.Popen('qsub -d '+pathtoWRF+' real_run.sh',cwd=pathtoWRF,shell=True,stdout=subprocess.PIPE)
        p_real.wait()
        jobid = p_real.stdout.read()[:5] # Assuming first five digits = job ID.
        # Run WRF but wait until Real has finished without errors
        print 'Now submitting wrf.exe.'  
        # Again, change name of submission script if needed
        p_wrf = subprocess.Popen('qsub -d '+pathtoWRF+' wrf_run.sh -W depend=afterok:'+jobid,cwd=pathtoWRF,shell=True)
        p_wrf.wait()
        print "real.exe and wrf.exe submitted. Exiting Python script."

    else:
        print "Pre-processing complete. Exiting Python script."

