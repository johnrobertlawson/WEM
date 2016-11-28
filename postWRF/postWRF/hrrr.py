from .gribfile import GribFile

class HRRR(GribFile):
    def __init__(self,fpath):
        """Initialise HRRR object, a child of WRFOut,
        grandchild of NC, great-grandchild of DataFile.
        """
        super().__init__(fpath)

