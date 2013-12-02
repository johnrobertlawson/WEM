"""Edit this file to taste.

The class PyWRFSettings is used to create config, containing all settings
"""

from PyWRFPlus import PyWRFPlusEnv
from settings import PyWRFSettings

config = PyWRFSettings()
PyWRFPlusEnv(config)
