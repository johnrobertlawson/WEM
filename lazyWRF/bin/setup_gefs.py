"""
Might want to put L.go into __init__ or 
even just put arguments inside the constructor 
"""
import pdb
import sys
sys.path.append('/home/jrlawson/gitprojects/')

from WEM.lazyWRF.lazyWRF import Lazy
from lazysettings import LazySettings

# This is where you mnight run getgefs, getgfs etc
case = '20110419'

config = LazySettings(case)
L = Lazy(config)


# Create ensemble
IC = 'GEFSR2' 
experiment = {'ICBC':'CTRL'}
#ensnames = ['c00'] + ['p'+"{0:02d}".format(n) for n in range(1,11)]
ensnames = ['c00'] + ['p'+"{0:02d}".format(n) for n in range(4,11)]

L.go(case,IC,experiment,ensnames)
