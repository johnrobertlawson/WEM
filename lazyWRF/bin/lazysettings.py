import os

class LazySettings:
    def __init__(self,case):
        """
        Input:
        case    :   YYYYMMDD of event for folder string
        """
        
        self.path_to_WPS = '/ptmp/jrlawson/WPS'
        self.path_to_WRF = '/ptmp/jrlawson/WRFV3/run'
        self.path_to_GEFSR2 = os.path.join(self.path_to_WPS,'gefsfiles')
        self.path_to_soil =  os.path.join(self.path_to_WPS,'gfsfiles',case,'gfsanl*')
        self.GEFSR2_Vtable = 'Vtable.GEFSR2'
        self.soil_Vtable = 'Vtable.GFS_soilonly'
        self.roughguesshr = 1 # Minimum time one WRF run would take
        self.path_to_storage = '/chinook2/jrlawson/bowecho/'
 
 
        # selected namelist.wps settings
        #self.maxdom = 1
        #self.dx =      

        # selected namelist.input settings
        #self.time_step = 
