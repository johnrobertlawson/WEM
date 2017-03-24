import os 
import glob
import pdb
import time
import subprocess 

import WEM.utils as utils

class Lazy:
    def __init__(self,sched,path_to_WRF,path_to_wrfout,submit_fname,
                    submit_ideal_fname=False,path_to_storage=False,
                    submit_real_fname=False,
                    path_to_WPS=False,path_to_atmos=False,path_to_soil=False):
        self.sched = sched
        self.path_to_WRF = path_to_WRF
        self.path_to_wrfout = path_to_wrfout
        self.path_to_WPS = path_to_WPS
        if path_to_WPS:
            self.doWPS = True
            self.init_WPS()
        else:
            self.doWPS = False
        self.path_to_storage = path_to_storage
        self.path_to_atmos = path_to_atmos
        self.path_to_soil = path_to_soil
        self.submit_fname = submit_fname
        self.submit_fpath = os.path.join(path_to_WRF,submit_fname)
        do_firstsubmit = 1
        if submit_ideal_fname:
            self.dofirst = 'ideal'
            self.submit_first_fname = submit_ideal_fname
        elif submit_real_fname:   
            self.dofirst = 'real'
            self.submit_first_fname = submit_real_fname
        else:
            do_firstsubmit = 0
        if do_firstsubmit:
            self.submit_first_fpath = os.path.join(path_to_WRF,self.submit_first_fname)

    
    def init_WPS(self,soilVtable_fname='Vtable.GFS_soilonly'):
        self.path_to_soilVtable = os.path.join(
                self.path_to_WPS,'ungrib/Variable_tables',soilVtable_fname)

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
        self.experiment = list(experiment.keys())[0]
        self.control = list(experiment.values())[0]
        self.ensnames = ensnames
        
        # self.GO = {'GEFSR2':go_GEFSR2,'NAMANL':go_NAMANL,
                #   'NAMFCST':go_NAMFCST, 'GFSANL':go_GFSANL,
                #   'GFSFCST':go_GFSFCST}
                
        self.GO = {'GEFSR2':self.go_GEFSR2,
                    'em_quarter_ss':self.go_quarter_ss,}
                    
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
            
                
    def copy_files(self,fromfolder,tofolder,matchfs=False):
        """Move wrfout files to folder.
        Create folder if it doesn't exist

        Move .TS files if they exist
        Copy namelist.input to that folder.
        Copy rsl.error.0000 to the folder.
        
        Input(s):
        args = names of folder tree, in order of depth.

        """
        root = self.path_to_storage
        # topath = os.path.join(root,tofolder)
        
        utils.trycreate(tofolder)
        
        if not matchfs:
            files = {'wrfout_d0*':'mv','namelist.input':'cp',
                    'rsl.error.0000':'cp'}
        else:
            files = matchfs

        # if len(glob.glob('*.TS')):
            # files['*.TS'] = 'mv'
            # files['tslist'] = 'cp'
        
        for f,transfer in files.items():
            fs = os.path.join(fromfolder,f)
            fout = os.path.join(tofolder,f)
            command = '{0} {1} {2}'.format(transfer,fs,fout)
            os.system(command)
            del command

        if self.doWPS:
            # Finally the namelist.wps.
            path_to_namelistwps = os.path.join(self.path_to_WPS,'namelist.wps')
            fout = os.path.join(tofolder,'namelist.wps')
            'cp {0} {1}'.format(path_to_namelistwps,fout)

    def submit_job(self,waitmin=10,sleepmin=5,wrf_only=False,first_only=False):
        if self.doWPS:
            # Soft link data netCDFs files from WPS to WRF
            self.link_to_met_em()
        
        if not wrf_only:
            if self.dofirst is 'real':
                print("Submitting real.exe.")
            else:
                print("Submitting ideal.exe.")

            if self.sched is 'rocks':
                scmd = 'qsub -d {0}'.format(self.path_to_WRF)
            elif self.sched is 'slurm':
                scmd = 'sbatch'
                real_cmd = '{0} {1}'.format(scmd,self.submit_first_fpath)
            p_real = subprocess.Popen(real_cmd,cwd=self.path_to_WRF,shell=True,stdout=subprocess.PIPE)
            p_real.wait()
            jobid = p_real.stdout.read()[:5] # Assuming first five digits = job ID.
        # pdb.set_trace()
        
        if not first_only:
            print("Submitting wrf.exe")
            if not wrf_only:
                aftercmd = 'depend=afterok:{0}'.format(jobid)
            else:
                aftercmd = ''

            # Run WRF but wait until real.exe has finished without errors
            if self.sched is 'rocks':
                wrf_cmd = 'qsub -d {0} {1} -W {2}' %(self.path_to_WRF,
                                    self.submit_fpath,aftercmd)
            else:
                wrf_cmd = 'sbatch {0} {1}'.format(self.submit_fpath,aftercmd)

            p_wrf = subprocess.Popen(wrf_cmd,cwd=self.path_to_WRF,shell=True)
            p_wrf.wait()

            time.sleep(waitmin*60) # Wait specified mins
            
            # Makes sure real.exe has finished and wrf.exe is writing to rsl files
            finished = 0
            while not finished:
                path_to_rsl = os.path.join(self.path_to_WRF,'rsl.error.0000')
                tail_cmd = 'tail {0}'.format(path_to_rsl)
                tailrsl = subprocess.Popen(tail_cmd,shell=True,stdout=subprocess.PIPE)
                tailoutput = tailrsl.stdout.read()
                if b"SUCCESS COMPLETE WRF" in tailoutput:
                    finished = 1
                    print("WRF has finished; moving to next case.")
                else:
                    # Need to check if job has died! If so, kill script, warn user
                    time.sleep(sleepmin*60) # Try again in x min
            
    def link_to_met_em(self):
        path_to_met_em = os.path.join(self.path_to_WPS,'met_em*')
        command = 'ln -sf %s %s' %(path_to_met_em,self.path_to_WRF)
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
            gribpath = os.path.join(self.path_to_GEFSR2,self.casestr,gribfiles)
            command = ' '.join((csh,gribpath))
       
        #pdb.set_trace() 
        os.system(command)

    def link_to_IC_Vtable(self,IC):
        if IC == 'GEFSR2':
            path = os.path.join(self.path_to_WPS,'ungrib/Variable_Tables',
                            self.GEFSR2_Vtable)
            command = 'ln -sf %s Vtable' %(path)
        
        os.system(command)

    def link_to_soil_data(self):
        csh = './link_grib.csh'
        command = ' '.join((csh,self.path_to_soil))
        os.system(command)
        
    def link_to_soil_Vtable(self):
        path = os.path.join(self.path_to_WPS,'ungrib/Variable_Tables',
                            self.soil_Vtable)
        command = 'ln -sf %s Vtable' %(path)
        os.system(command)
        
    def add_new_namelist(self,suffix,section,sett,newval,quoted):
        """Adds new namelist setting at end of section, before
        the backslash."""

        if quoted:
            newval = "'{0}'".format(newval)

        f = self.return_namelist_path(suffix)

        flines = open(f,'r').readlines()
        sidx = 'None'
        for idx, line in enumerate(flines):
            if section in line:
                sidx = idx
            if isinstance(sidx,int) and '/' in line[:4]:
                spaces = 36
                flines[idx] = " {0: <{sp}}= {1}, \n / \n ".format(sett,newval,sp=spaces)
                nameout = open(f,'w')
                nameout.writelines(flines)
                nameout.close()
                break
                
    def remove_namelist_line(self,suffix,sett):
        f = self.return_namelist_path(suffix)
        flines = open(f,'r').readlines()
        for idx, line in enumerate(flines):
            if sett in line:
                flines[idx] = "\n"
                nameout = open(f,'w')
                nameout.writelines(flines)
                nameout.close()
                break


    def return_namelist_path(self,suffix):
        """Gives path to namelist depending on suffix argument.
        """
        if suffix == 'wps':
            f = os.path.join(self.path_to_WPS,'namelist.wps')
        elif suffix == 'input':
            f = os.path.join(self.path_to_WRF,'namelist.input')
        return f


    def edit_namelist(self,suffix,sett,newval,maxdom=1,quoted=False):
        """Method edits namelist.wps or namelist.input.
        
        Args:
            suffix : which namelist needs changing
            sett : setting that needs changing
            newval : its new value -> currently replaces whole line
            maxdom : number of domains to edit
                (this is relevant for multiple columns?)
        
        Returns:
            None.
        """
        if quoted:
            newval = "'{0}'".format(newval)
        f = self.return_namelist_path(suffix)
        utils.edit_namelist(f,sett,newval)
        return


    def run_exe(self,exe):
        """Run WPS executables, then check to see if it failed.
        If not, return True to proceed.
        
        Input:
        exe     :   .exe file name.
        """
        
        #if not exe.endswith('.exe'):
        #    f,suffix = exe.split('.')            
        
        command = os.path.join('./',self.path_to_WPS,exe)
        print(command)
        os.system(command)
        
        # Wait until complete, then check tail file
        name,suffix = exe.split('.')
        log = os.path.join(self.path_to_WPS,name + '.log')
        l = open(log,'r').readlines()
        lastline = l[-1]
        if 'Successful completion' in lastline:
            pass
            #return True
        else:
            print(('Running %s has failed. Check %s.'%(exe,log)))
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
                f = os.path.join(self.path_to_WPS,'namelist.wps') # Original 
            elif suffix == 'input':
                f = os.path.join(self.path_to_WRF,'namelist.input') # Original    
                
            f2 = '_'.join((f,t)) # Backup file
            # pdb.set_trace()
            command = 'cp %s %s' %(f,f2)
            os.system(command)
            print(("Backed up namelist.%s." %(suffix)))
            
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
