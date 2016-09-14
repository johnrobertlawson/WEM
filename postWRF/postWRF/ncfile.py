from netCDF4 import Dataset

from WEM.postWRF.postWRF.datafile import DataFile

class NC(DataFile):
    def __init__(self,fpath):
        """Generic netCDF import.

        Subclass of generic DataFile class.
        """
        super().__init__(fpath)
        self.nc = Dataset(fpath,'r')
