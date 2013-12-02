"""Edit this file to taste.

The class PyWRFSettings (in settings.py, same folder) is used to create config, containing all settings
"""

from PyWRFPlus import PyWRFEnv
from settings import PyWRFSettings

config = PyWRFSettings()
PyWRFEnv(config)
