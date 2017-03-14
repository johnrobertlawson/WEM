from .gribfile import GribFile

class HRRR(GribFile):
    def __init__(self,fpath):
        """Initialise HRRR object, a child of GribFile,
        grandchild of DataFile.
        """
        super().__init__(fpath)

    def lookup_vrbl(self,vrbl):
        LOOKUP = {}
        LOOKUP['accum_precip'] = {'key':'Total Precipitation','idx':1}

        return LOOKUP[vrbl]['key'], LOOKUP[vrbl]['idx']
