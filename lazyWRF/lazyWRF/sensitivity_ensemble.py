# This is an example of creating a WRF ensemble using the stochastic kinetic energy backscatter scheme
# We assume met_em files have all been created, have been linked etc

# Settings
datestring = '20090910'
ensembletype = 'STCH' # Change in future to expand to e.g. mixed microphysics
pathtoWRF = './' # Path to WRF folder where wrfout* files are generated

def edit_namelist_input(old,new,pathtoWRF=pathtoWRF):
    ninput = open(pathtoWRF+'namelist.input','r').readlines()
    for idx, line in enumerate(ninput):
        if old in line:
            ninput[idx]= ninput[idx][:39] + new + " \n"
            nameout = open(pathtoWRF+'namelist.input','w')
            nameout.writelines(ninput)
            nameout.close()
            break
    return

if ensembletype == 'STCH':
    ensnames = ['s' + '%02u' %n for n in range(1,11)] # Change the range to create n-member ensemble

for n,ens in enumerate(ensnames):
    # First, edit namelists
    if ensembletype == 'STCH':
        edit_namelist_input('nens',str(n)+',')

    # Run real.exe
    p_real = subprocess.Popen('qsub -d '+pathtoWRF+' real_run.sh',cwd=pathtoWRF,shell=True,stdout=subprocess.PIPE)
    p_real.wait()
    jobid = p_real.stdout.read()[:5] # Assuming first five digits = job ID.

    # Run wrf.exe, but wait until real.exe has finished without errors
    print 'Now submitting wrf.exe.'
    # Again, change name of submission script if needed
    p_wrf = subprocess.Popen('qsub -d '+pathtoWRF+' wrf_run.sh -W depend=afterok:'+jobid,cwd=pathtoWRF,shell=True)
    p_wrf.wait()
    print "real.exe and wrf.exe submitted."

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
    wrfoutdir = '/ptmp/jrlawson/predictability/'+datestring+'/wrfout/'+ensembletype+'/'+ens+'/'

    try:
        os.stat(wrfoutdir)
    except:
        os.makedirs(wrfoutdir)

    os.system('cp ' + pathtoWRF + 'namelist.input ' + wrfoutdir)
    os.system('mv' + pathtoWRF + 'wrfout* ' + wrfoutdir)
    os.system('rm -f ' + pathtoWRF + 'rsl*')

    # Loop back to top    
