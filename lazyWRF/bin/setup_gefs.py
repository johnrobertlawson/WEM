# Create met_em files 
# Uses command-line options for:
# -number where number is the ensemble member (0-10)
# -a Automatically find last member created and do next

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

# SETTINGS
# Directory with GEFS data
gefsdir = './gefsfiles/'
pathtosoildata = './gfsfiles/gfsanl*'
# If switched on, the script will submit jobs (edit as needed)
submit_job = 1
# Change if WRF job submission required
pathtoWRF = '../WRFV3/run/'

ensemblelist = ['c00'] + ['p'+str(n) for n in range(1,11)]

# The script first checks for command-line argument
try:
    arg = sys.argv[1]
except IndexError: 
    print "A command-line argument is required (-e or -a)"
    raise Exception
#finally:
#    if arg == '-a':
#        print "Automatically generating next ensemble member"
#        print "Using ensemble member ", arg[-1]

# Then it opens the WPS namelist, copies the old one
os.system('cp namelist.wps{,.python_backup}')
nwps = open('namelist.wps','r').readlines()
# Get the GEFS initialisation date from here
for line in nwps:
    if "start_date" in line:
        datestring = line[15:19] + line[20:22] + line[23:25]
        break

# Find intermediate filename line in WPS namelist
# Link to, and ungrib, soil files if they don't exist
if not glob.glob('FILE_SOIL*'):
    edit_namelist("prefix"," prefix = 'FILE_SOIL'")
    os.system('./link_grib.csh ' + pathtosoildata)
    os.system('ln -sf ungrib/Variable_Tables/Vtable.GFS_soilonly Vtable')
    os.system('./ungrib.exe')


# Next, one of two things:

# (1) if the command-line argument is automatic:
if arg == '-a':
    # Grab link_grib links to grib files
    griblinks = glob.glob('GRIB*')
    if not griblinks:
        nextens = 'c00'
        # Then we must run the control first
    else:
        # Find the last softlinked ensemble member file
        softlink = os.readlink(griblinks[0])
        # Check to see if last member was E10
        # If it is, our work here is done
        if "p10" in softlink:
            print "Last ensemble member already completed"
            raise Exception
        else:
            for idx, n in enumerate(ensemblelist):
                if n in softlink:
                    lastens = n
                    nextens = ensemblelist(idx+1)

else:
    # (2) if the command-line argument is set:
    if arg == '-0':
        nextens = 'c00'
    else:
        nextens = 'p' + str(arg[-1])

# Set path to gefs data
pathtogefsdata = gefsdir+datestring+'_'+nextens+'_f*'
    
# Now change the namelist.wps to use the atmospheric data prefix
#pdb.set_trace()
edit_namelist("prefix"," prefix = 'FILE_ATMOS'")
os.system('./link_grib.csh ' + pathtogefsdata)
os.system('ln -sf ungrib/Variable_Tables/Vtable.GEFSR2 Vtable')
os.system('./ungrib.exe')

# Combine both GEFS atmos and GFS soil data
edit_namelist("fg_name"," fg_name = 'FILE_SOIL','FILE_ATMOS'")
os.system('./metgrid.exe')

# Submit jobs (edit as needed)
if submit_job == 1:
    # Soft link data netCDFs files from WPS to WRF
    os.system('ln -sf ./met_em* ' + pathtoWRF)
    # Run real, get ID number of job
    p_real = subprocess.call([qsub,real_run.sh],cwd=pathtoWRF,shell=True,
                            stdout=subprocess.PIPE)
    jobid = p.stdout.read()[:5]
    # Run WRF but wait until Real has finished without errors
    p_wrf = subprocess.call([qsub,wrf_run.sh,-W,"afterok=jobid"],
                            cwd=pathtoWRF,shell=True)
    print "real.exe and wrf.exe submitted. Exiting Python script."
else:
    print "Pre-processing complete. Exiting Python script."

