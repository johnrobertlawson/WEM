"""Edit this file to taste.

The class PyWRFSettings (in settings.py, same folder) is used to create config, containing all settings
"""

from PyWRFPlus import PyWRFEnv
from settings import PyWRFSettings

# Initialise configuration
config = PyWRFSettings()
# Initialise plotting environment
p = PyWRFEnv(config)

## PLOTTING
# Time for first plots
config.time = (2006,5,26,0,0,0)
# Run some plotting scripts
p.plot_shear(0,3)

# Create a tuple of times for iterating over
timelist = [(2006,5,27,n,0,0) for n in range(1,7)]
for t in timelist:
    config.time = t
    config.plottype = 'contourf'
    p.plot_var('sfc','dewpoint')

