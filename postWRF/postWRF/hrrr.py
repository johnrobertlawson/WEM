from .wrfout import WRFOut

class HRRR(WRFOut):
    def __init__(self,fpath):
        """Initialise HRRR object, a child of WRFOut,
        grandchild of NC, great-grandchild of DataFile.
        """
        super().__init__(fpath)
