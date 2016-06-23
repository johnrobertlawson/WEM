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
"""

def calc_theta(z,ztr=12000.0,theta0=300.0,thetatr=343,Ttr=213):
    if z <= ztr:
        theta = theta0 + (thetatr - theta0) * (z/ztr)**1.25
    else:
        theta = thetatr * N.exp((mc.g/(mc.cp*Ttr))*(z-ztr))
    return theta

def calc_RH(z,ztr=12000.0):
    if z <= ztr:
        RH = 1-(0.75*(z/ztr)**1.25
    else:
        RH = 0.25
    return RH

def calc_U(z,us,zs=3000):
    U = us * (N.tanh(z)/zs)
    return U

heights = N.hstack((array([10,35]),N.arange(100,20100,100),
                    N.arange(20000,30500,500)))
qs = N.arange(11,16.25,0.25)
