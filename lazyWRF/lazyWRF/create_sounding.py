import csv
import numpy as N
from WEM.utils import metconstants as mc

"""
ROWS:
line 1 is surface information
lines 2 onwards is a point in profile

COLUMNS:
column 1 is surface pressure (hPa) for row 1
--- set this as 1000.00.
otherwise it is height (m) for rows 2 onwards
--- increasing in height with row number

column 2 is potential temperature (K)
--- set row 1 as 300.0 K as reference, probably

column 3 is vapour mixing ratio (g/kg)
--- set row 1 as 14.00 g/kg, for instance

column 4 is U wind (m/s)
--- no row 1

column 5 is V wind (m/s)
--- no row 1

METHODS FOR PROFILE GENERATION
Temp:
    - Weisman and Klemp
    etc
Wind:
    - Blah

Dewpoint:
    - Blah

"""

class Profile(Object):
    def __init__(self,method='MW01'):
        pass

    def buoyancy(self,E,m,z,H=14500):
        """
        Parcel buoyancy profile, or Eq. (A1) in MW01.

        Parameters
        ----------
        E   :   int, float
            Specified CAPE
        m   :   int, float
            Profile compression parameter
        z   :   int, float
            Altitude above ground level (metres)
        H   :   int, float, optional
            Vertical scale

        Returns
        -------

        profile :   array_like
            Atmospheric profile.
        """"


    def calc_theta(self,z,ztr=12000.0,theta0=300.0,thetatr=343,Ttr=213):
        if z <= ztr:
            theta = theta0 + (thetatr - theta0) * (z/ztr)**1.25
        else:
            theta = thetatr * N.exp((mc.g/(mc.cp*Ttr))*(z-ztr))
        return theta

    def calc_RH(self,z,ztr=12000.0):
        if z <= ztr:
            RH = 1-(0.75*(z/ztr)**1.25
        else:
            RH = 0.25
        return RH

    def calc_U(self,z,us,zs=3000):
        U = us * (N.tanh(z)/zs)
        return U

