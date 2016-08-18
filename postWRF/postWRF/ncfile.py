"""Compute or load data from netCDF file.

Dimensions of 4D variable X are X.dimensions:
(Time,bottom_top,south_north,west_east_stag)
Time, levels, latitude, longitude
"""

from netCDF4 import Dataset
import sys
import os
import numpy as N
import calendar
import pdb
import scipy.ndimage
import collections
import scipy.interpolate
import datetime

import WEM.utils as utils
from WEM.utils import metconstants as mc

class NC(object):

    """
    General netCDF file import.
    """
    def __init__(self,fpath):
        """
        Initialisation fetches and computes basic user-friendly
        variables that are most oftenly accessed.

        :param fpath:   absolute path to netCDF4 (wrfout) file
        :type fpath:    str

        """

        self.path = fpath
        self.nc = Dataset(fpath,'r')
        return
