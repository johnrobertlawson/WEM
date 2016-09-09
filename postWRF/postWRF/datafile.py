class DataFile(object):
    def __init__(self,fpath):
        """Generic superclass for data file. Could be netCDF, grib1,
            grib2...
        """
        self.fpath = fpath
        
