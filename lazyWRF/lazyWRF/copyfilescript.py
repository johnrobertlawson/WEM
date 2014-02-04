### IMPORTS
import os
import pdb

def copyfiles(pathtoWRF,outdir,ensmem,confirm_switch=0):
    ### Determine values from latest namelist file
    f = open(pathtoWRF+'namelist.input')
    lines = f.readlines()

    # Number always at column 39
    yr = lines[5][39:43]
    mth = lines[6][39:41]
    day = lines[7][39:41]
    hr = lines[8][39:41]
    # Join into init_time
    init_time = yr+mth+day+hr+'/'
    # This is the folder for all GEFS forecasts on this date

    outloc = outdir+init_time+'wrfout/'+str(ensmem)+'/'
    
    files = ('wrfout*','namelistCOPY.input','wrfbdy*','wrfinput*','*.TS', 'tslist.txt','rsl.error.0000','*.PH','*.QV','*.TH','*.UU','*.VV')
    if confirm_switch:
        confirm = raw_input("Move data files to directory"+outloc+" (y/n) ? ")

        if confirm=='n':
            print "Aborting."
            raise Exception
        elif confirm=='y':
            pass
        else:
            print "Type y or n."
            raise Exception

    ### Create commands
    commandlist = []
    #outputdir = rootdir + small_dom + model + init_time
    for f in files:
        cmnd = 'mv ' + pathtoWRF + f + ' ' + outloc
        #if model == 'GEFS/':
        #    cmnd += 'ensemble_no'
        commandlist.append(cmnd)

    os.system('cp '+pathtoWRF+'namelist.input ' + files[1])
    os.system('cp '+pathtoWRF+'tslist ' + files[5])

    for cInd, c in enumerate(commandlist):
        dir = os.path.dirname(outloc)
        print "Checking for directory for command #", cInd
        try:
            os.stat(dir)
        except:
            os.makedirs(dir)
            print "Creating directory."
        os.system(c)
        print 'Completing command #', cInd

    # Wipe rsl.error and rsl.out files
    os.system('rm -f rsl.error* rsl.out*')

"""
pathtoWRF = '/ptmp/jrlawson/WRFV3/run/'
outdir = '/ptmp/jrlawson/predictability/'
ensmem = raw_input("Type the ensemble member and number, 'E#', or 'C0' for control.")
copyfiles(pathtoWRF,outdir,ensmem)
"""
