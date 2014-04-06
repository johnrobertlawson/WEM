import os 
import glob
import pdb
import time
import subprocess 

import WEM.utils as utils

class Lazy:
    def __init__(self,config):
        self.C = config
    
    def go(self,casestr,IC,experiment,ensnames,**kwargs):
        """
        Inputs: (all folder names)
        casestr     :   string of case study initialisation date/time
        IC          :   initial condition model
        experiment  :   dictionary. Key: ensemble type (ICBC,STCH,MXMP)
                            -initial condition/boundary condition 
                            -stochastic kinetic energy backscatter
                            -mixed model parameterisations
                        Value...
                            -model configuration (CTRL) for ICBC
                            -initial condition model/ens member for others
        ensnames    :   list of ensemble member names
                            - e.g. c00,p01,p02

        **kwargs include:
        WPS_only    :   stop after linking met_em files to WRF folder
                            
        """
        self.casestr = casestr
        self.IC = IC
        self.experiment = experiment.keys()[0]
        self.control = experiment.values()[0]
        self.ensnames = ensnames
        
        # self.GO = {'GEFSR2':go_GEFSR2,'NAMANL':go_NAMANL,
                #   'NAMFCST':go_NAMFCST, 'GFSANL':go_GFSANL,
                #   'GFSFCST':go_GFSFCST}
                
        self.GO = {'GEFSR2':self.go_GEFSR2}
                    
        """
        self.enstype is a list of ensemble models or types.
        self.enslist is a list of each model's member names.
        Both are used to name runs, and folders for output etc
        
        Lookup table GO contains the methods to run for each 
        ensemble type.
        """
        
        self.GO[IC](self.ensnames,**kwargs)
        
    def go_GEFSR2(self,ensns,**kwargs):    
        
        """ 
        Runs WPS, WRF for one set of initial conditions
        and settings based on the GEFSR2 model.
        
        Inputs:
        ensns   :   names of ensemble member
        """
        # Creating backup!
        self.copy_namelist('wps')
        
        # Warning: deleting old data files first.
        # Back these up if important

        files = ('met_em*','GEFSR2*','geo_em.d*','SOIL*','GRIB*')
        for f in files:
            if len(glob.glob(f)):
                #command = 'rm -f {0}'.format(f)
                command = 'rm -f %s' %f # < 2.6 edit
                os.system(command)

        # Starting loop
        for n,e in enumerate(ensns):
            # e is the next ensemble member to run
            
            if n==0:
                # First time, generate soil data from GFS analyses
                # and set up geogrid.

                self.run_exe('geogrid.exe')
    
                # Create soil data intermediate files
                
                self.edit_namelist('wps',"prefix"," prefix = 'SOIL'")
                self.link_to_soil_data()
                self.link_to_soil_Vtable()
                self.run_exe('ungrib.exe')
            else:
                # Soil data already exists
                # Remove old met_em files and links
                os.system('rm GRIB* GEFSR2* met_em*')
    
            # Create atmos intermediate files
            self.edit_namelist('wps',"prefix"," prefix = 'GEFSR2'")
            self.link_to_IC_data('GEFSR2',e)
            self.link_to_IC_Vtable('GEFSR2')
            self.run_exe('ungrib.exe')
            
            # Combine both intermediate files
            self.edit_namelist('wps',"fg_name"," fg_name = 'GEFSR2','SOIL' ")
            self.run_exe('metgrid.exe')

            if 'WPS_only' in kwargs:
                continue
            # This is where the magic happens etc etc
            #self.submit_job()[0] <--- why was this [0] here?
            self.submit_job()
            
            # Move files to storage before looping back
            to_folder = os.path.join(self.casestr,'GEFSR2',e,self.experiment)
            self.copy_files(to_folder)      
            os.system('rm -f rsl.error* rsl.out*')
            
                
    def copy_files(self,tofolder):
        """ 
        Move wrfout* files to folder.
        Create folder if it doesn't exist

        Move *.TS files if they exist
        Copy namelist.input to that folder.
        Copy rsl.error.0000 to the folder.
        
        Input(s):
        args = names of folder tree, in order of depth.
        """
        root = self.C.path_to_storage
        topath = os.path.join(root,tofolder)
        
        utils.trycreate(topath)
        
        files = {'wrfout_d0*':'mv','namelist.input':'cp',
                    'rsl.error.0000':'cp'}

        if len(glob.glob('*.TS')):
            # hi-res time series files
            files['*.TS'] = 'mv'
            files['tslist'] = 'cp'
        
        for f,transfer in files.iteritems():
            fs = os.path.join(self.C.path_to_WRF,f)
            command = '%s %s %s' %(transfer,fs,topath)
            os.system(command)
            del command

        # Finally the namelist.wps.
        path_to_namelistwps = os.path.join(self.C.path_to_WPS,'namelist.wps')
        'cp %s %s' %(path_to_namelistwps,topath)

    def submit_job(self):
        # Soft link data netCDFs files from WPS to WRF
        self.link_to_met_em()
        
        print("Submitting real.exe.")
        real_cmd = 'qsub -d %s real_run.sh' %(self.C.path_to_WRF)
        p_real = subprocess.Popen(real_cmd,cwd=self.C.path_to_WRF,shell=True,stdout=subprocess.PIPE)
        p_real.wait()
        jobid = p_real.stdout.read()[:5] # Assuming first five digits = job ID.
        
        # Run WRF but wait until real.exe has finished without errors
        print 'Now submitting wrf.exe.'
        wrf_cmd = 'qsub -d %s wrf_run.sh -W depend=afterok:%s' %(self.C.path_to_WRF,jobid)
        p_wrf = subprocess.Popen(wrf_cmd,cwd=self.C.path_to_WRF,shell=True)
        p_wrf.wait()

        time.sleep(self.C.roughguesshr*60*60) # Wait specified hours
        
        # Makes sure real.exe has finished and wrf.exe is writing to rsl files
        finished = 0
        while not finished:
            path_to_rsl = os.path.join(self.C.path_to_WRF,'rsl.error.0000')
            tail_cmd = 'tail %s' %(path_to_rsl)
            tailrsl = subprocess.Popen(tail_cmd,shell=True,stdout=subprocess.PIPE)
            tailoutput = tailrsl.stdout.read()
            if "SUCCESS COMPLETE WRF" in tailoutput:
                finished = 1
                print "WRF has finished; moving to next case."
            else:
                # Need to check if job has died! If so, kill script, warn user
                time.sleep(5*60) # Try again in 5 min
        
    def link_to_met_em(self):
        path_to_met_em = os.path.join(self.C.path_to_WPS,'met_em*')
        command = 'ln -sf %s %s' %(path_to_met_em,self.C.path_to_WRF)
        os.system(command)
             
    def link_to_IC_data(self,IC,*args):
        """
        Inputs:
        *args   :   e.g. ensemble member
        """
        if IC == 'GEFSR2':
            """
            Assumes files are within a folder named casestr (YYYYMMDD)
            """
            csh = './link_grib.csh'
            nextens = args[0]
            gribfiles = '_'.join((self.casestr,nextens,'f*'))
            gribpath = os.path.join(self.C.path_to_GEFSR2,self.casestr,gribfiles)
            command = ' '.join((csh,gribpath))
       
        #pdb.set_trace() 
        os.system(command)

    def link_to_IC_Vtable(self,IC):
        if IC == 'GEFSR2':
            path = os.path.join(self.C.path_to_WPS,'ungrib/Variable_Tables',
                            self.C.GEFSR2_Vtable)
            command = 'ln -sf %s Vtable' %(path)
        
        os.system(command)

    def link_to_soil_data(self):
        csh = './link_grib.csh'
        command = ' '.join((csh,self.C.path_to_soil))
        os.system(command)
        
    def link_to_soil_Vtable(self):
        path = os.path.join(self.C.path_to_WPS,'ungrib/Variable_Tables',
                            self.C.soil_Vtable)
        command = 'ln -sf %s Vtable' %(path)
        os.system(command)
        
    def edit_namelist(self,suffix,sett,newval,maxdom=1):
        """ Method edits namelist.wps or namelist.input.
        
        Inputs:
        suffix  :   which namelist needs changing
        sett    :   setting that needs changing
        newval  :   its new value -> currently replaces whole line
        maxdom  :   number of domains to edit
                    (this is relevant for multiple columns?)
                    
        No outputs, just changes the file.
        """
        if suffix == 'wps':
            f = os.path.join(self.C.path_to_WPS,'namelist.wps')
        elif suffix == 'input':
            f = os.path.join(self.C.path_to_WRF,'namelist.input')
        flines = open(f,'r').readlines()
        for idx, line in enumerate(flines):
            if sett in line:
                # Prefix for soil intermediate data filename
                flines[idx] = newval + " \n"
                nameout = open(f,'w')
                nameout.writelines(flines)
                nameout.close()
                break

    def run_exe(self,exe):
        """Run WPS executables, then check to see if it failed.
        If not, return True to proceed.
        
        Input:
        exe     :   .exe file name.
        """
        
        #if not exe.endswith('.exe'):
        #    f,suffix = exe.split('.')            
        
        command = os.path.join('./',self.C.path_to_WPS,exe)
        print command
        os.system(command)
        
        # Wait until complete, then check tail file
        name,suffix = exe.split('.')
        log = os.path.join(self.C.path_to_WPS,name + '.log')
        l = open(log,'r').readlines()
        lastline = l[-1]
        if 'Successful completion' in lastline:
            pass
            #return True
        else:
            print ('Running %s has failed. Check %s.'%(exe,log))
            raise Exception

    def generate_date(self,date,outstyle='wps'):
            """ Creates date string for namelist files.
            
            Input:
            """
            pass
        
    def copy_namelist(self,suffix):
            """Appends current time to namelist to create backup.
            """
            t = time.strftime("%Y%m%d_%H%M")
            if suffix == 'wps':
                f = os.path.join(self.C.path_to_WPS,'namelist.wps') # Original 
            elif suffix == 'input':
                f = os.path.join(self.C.path_to_WRF,'namelist.input') # Original    
                
            f2 = '_'.join((f,t)) # Backup file
            # pdb.set_trace()
            command = 'cp %s %s' %(f,f2)
            os.system(command)
            print("Backed up namelist.%s." %(suffix))
            
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
