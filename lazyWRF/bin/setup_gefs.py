"""
Might want to put L.go into __init__ or 
even just put arguments inside the constructor 
"""

import sys
sys.path.append('/home/jrlawson/gitprojects/WEM/')

from WEM.lazyWRF import Lazy
from lazysettings import LazySettings

# This is where you mnight run getgefs, getgfs etc

config = LazySettings()
L = Lazy(organise)

case = '20110419'

# Create ensemble
IC = 'GEFSR2' 
experiment = {'ICBC','CTRL'}
ensnames = ['c00'] + ['p'+"{0:02d}".format(n) for n in range(1,11)]}

L.go(case,IC,experiment,ensnames)
