# Run ensemble of SKEB, microphysics, or both
# We assume met_em files have all been created

# RUN THIS ON CHINOOK
# ROCKS JOBS ARE SUBMITTED VIA DERECHO (SSH)
# Files are moved to /tera9/ for storage
# Quicklook images are created after each run 
# Script is executed from /home/jrlawson/pythoncode/plotting/scripts/
# Saved to /home/jrlawson/pythoncode/plotting/output/

# Imports
import subprocess
import pdb
import os
import time
import sys
import pexpect
from sshls import ssh_command

sys.path.append('/home/jrlawson/pythoncode/plotting/scripts/')
import plot_quicklooks

# Settings
pause_to_check = 0 # To check namelists before job submission
#datestring = '2009091000' # Case
datestring = '20110419'
skipfirst = 0 # Needing to start halfway through an ensemble
#datestring = '20130815'
ICens = 'p04' # Initial conditions for ensembles
IC = 'GEFSR2' # Model for initial/boundary conditions in WRF
experiment = 'STCH' #STCH or MXMP or STCH_MXMP
pathtoWRF = '/ptmp/jrlawson/WRFV3/run/' # Run this file in WRF directory
MPn = 8 # For SKEB: pick constant MP scheme
GHn = 0 # For SKEB: pick constant graupel/hail option
MP = 'ICBC' # For SKEB: name of MP scheme

def edit_namelist_input(old,new,pathtoWRF=pathtoWRF):
    # Pad old and new with a space before and after
    # to avoid partial matches

    old = ' '+old+' '
    #new = ' '+new+' '
    ninput = open(pathtoWRF+'namelist.input','r').readlines()
    exists = 0
    for idx, line in enumerate(ninput):
        if old in line:
            exists = 1
            ninput[idx]= ninput[idx][:39] + new + " \n"
            nameout = open(pathtoWRF+'namelist.input','w')
            nameout.writelines(ninput)
            nameout.close()
            break
    if not exists:        
        print("Line %s not found" % old)
        raise Exception
    else:
        return

def create_quicklook(datestring,ICmodel,ensembletype,ICens,ens,pathtonc):
    varlist = ['cref','wind'] # Plot simulated reflectivity and 10m wind
    for v in varlist:
        plot_quicklooks.plotquick(datestring,ICmodel,ensembletype,ICens,ens,v,pathtonc)
    return 

ensembletype = experiment # Same thing, whatever

if ensembletype == 'STCH': # 10 members
    ensnames = ['s' + '%02u' %n for n in range(1,11)]
elif ensembletype == 'MXMP': # 11 members
    # See p1194 of Adams-Selin "Sensitivity of Bow-Echo Sim..." 2013 WAF
    # With adaptations to method thanks to pers. comm. 2013
    # Setting graupel_param to zero is graupel; to one, it is hail (see Fortran code for MP schemes)
    ensnames = ['WSM6_Grau','WSM6_Hail','Kessler','Ferrier','WSM5','WDM5','Lin','WDM6_Grau','WDM6_Hail','Morrison_Grau','Morrison_Hail']
    scheme_no = [6,6,1,5,4,14,2,16,16,10,10]
    graupel_param = [0,1,0,0,0,0,0,0,1,0,1]
elif ensembletype == 'STCH_MXMP': # 11 members
    ensnames = ['WSM6_Grau_STCH','WSM6_Hail_STCH','Kessler_STCH','Ferrier_STCH','WSM5_STCH','WDM5_STCH','Lin_STCH','WDM6_Grau_STCH','WDM6_Hail_STCH','Morrison_Grau_STCH','Morrison_Hail_STCH']
    scheme_no = [6,6,1,5,4,14,2,16,16,10,10]
    graupel_param = [0,1,0,0,0,0,0,0,1,0,1]

for n,ens in enumerate(ensnames):
    if n<skipfirst:
        continue
    else:
        print("Running",ens)
    # First, edit namelists
    if ensembletype == 'STCH':
        edit_namelist_input('stoch_force_opt','1')
        edit_namelist_input('nens',str(n+1)+',')
        edit_namelist_input('mp_physics',str(MPn)+',')
        edit_namelist_input('afwa_hail_opt',str(GHn)+',')
    elif ensembletype == 'MXMP':
        edit_namelist_input('stoch_force_opt','0')
        #edit_namelist_input('nens',str(n+1)+',')
        edit_namelist_input('mp_physics',str(scheme_no[n])+',')
        edit_namelist_input('afwa_hail_opt',str(graupel_param[n]))
    elif ensembletype == 'STCH_MXMP':
        edit_namelist_input('stoch_force_opt','1')
        # For this SKEB ensemble, use different seeding from STCH for uniqueness
        edit_namelist_input('nens',str(n+11)+',')
        edit_namelist_input('mp_physics',str(scheme_no[n])+',')
        edit_namelist_input('afwa_hail_opt',str(graupel_param[n]))       

    # Running on Chinook, need to ssh into derecho to submit job, and then log out

    if pause_to_check:
        pdb.set_trace()

    user = 'user'
    host = 'domain.name'
    password = 'password' # Change this to use .ssh keys!!
    jobid = '0'

    # REAL.pbs
    REALcmd = ' '.join(('qsub','-d',pathtoWRF,'real_run.sh'))
    child = ssh_command(user,host,password,REALcmd)
    child.expect(pexpect.EOF)
    jobid = child.before[2:7]
    print(child.before)

    # WRF.PBS
    WRFcmd = ' '.join(('qsub','-d',pathtoWRF,'wrf_run.sh','-W','depend=afterok:'+jobid)) 
    child = ssh_command(user,host,password,WRFcmd)
    child.expect(pexpect.EOF)
    print(child.before)

    # DONE WITH DERECHO NOW

    #p_real = subprocess.Popen('qsub -d '+pathtoWRF+' real_run.sh',cwd=pathtoWRF,shell=True,stdout=subprocess.PIPE)
    #p_real.wait()
    #jobid = p_real.stdout.read()[:5] # Assuming first five digits = job ID.

    # Run wrf.exe, but wait until real.exe has finished without errors
    #print 'Now submitting wrf.exe.'
    # Again, change name of submission script if needed
    #p_wrf = subprocess.Popen('qsub -d '+pathtoWRF+' wrf_run.sh -W depend=afterok:'+jobid,cwd=pathtoWRF,shell=True)
    #p_wrf.wait()
    #print "real.exe and wrf.exe submitted."

    # Wait an hour to make sure real.exe is complete and wrf.exe is writing to file
    time.sleep(60*60)

    # Check log file until wrf.exe has finished
    finished = 0
    while not finished:
        tailrsl = subprocess.Popen('tail '+pathtoWRF+'rsl.error.0000',shell=True,stdout=subprocess.PIPE)
        tailoutput = tailrsl.stdout.read()
        if "SUCCESS COMPLETE WRF" in tailoutput:
            finished = 1
            print "WRF has finished; moving to next case."
        else:
            time.sleep(5*60) # Try again in 5 min

    # Copy namelist.input to directory
    # Move wrfout file to directory
    # Create directory if it doesn't exist
    if experiment is 'STCH':
        wrfoutdir = os.path.join('/chinook2','jrlawson','bowecho',datestring,IC,ICens,MP,ens)
    else:
        wrfoutdir = os.path.join('/chinook2','jrlawson','bowecho',datestring,IC,ICens,ens)

    try:
        os.stat(wrfoutdir)
    except:
        os.makedirs(wrfoutdir)

    os.system('cp ' + pathtoWRF + 'namelist.input ' + wrfoutdir)
    os.system('mv ' + pathtoWRF + 'wrfout* ' + wrfoutdir)
    os.system('mv ' + pathtoWRF + 'rsl.error.0000 ' + wrfoutdir)
    os.system('rm -f ' + pathtoWRF + 'rsl*')

    # Before a loop back to top, create quicklooks
    #create_quicklook(datestring,ICmodel,ensembletype,ICens,ens,wrfoutdir)

