"""
Scripts related to plotting of observed data, such as surface observations,
radar reflectivity, severe weather reports.
"""

import scipy.ndimage
import pdb

class Obs(object):
    """
    An instance represents a data set.
    """
    def __init__(self,fpath):
        self.fpath = fpath

class Radar(Obs):
    def __init__(self,datapath,wldpath):
        """
        Composite radar archive data from mesonet.agron.iastate.edu.

        :param datapath:        Absolute path to .png file
        :type datapath:         str
        :param wldpath:         Absolute path to .wld file
        :type wldpath:          str
        """
        # Data
        self.data = scipy.ndimage(datapath)

        # Metadata
        f = open(wldpath,'r').readlines()

        # pixel size in the x-direction in map units/pixel
        self.xpixel = f[0]

        # rotation about y-axis
        self.roty = f[1]

        # rotation about x-axis
        self.rotx = f[2]

        # pixel size in the y-direction in map units,
        self.ypixel = f[3]

        # x-coordinate of the center of the upper left pixel
        self.ulx = f[4]

        # y-coordinate of the center of the upper left pixel
        self.uly = f[5]
        pdb.set_trace()
        
    def generate_basemap(self):
        """
        Generate basemap object
        """
