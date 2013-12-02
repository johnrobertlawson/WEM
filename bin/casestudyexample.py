"""Edit this file to taste.

The class PyWRFSettings is used to create config, containing all settings
"""

from PyWRFPlus import PyWRFPlusEnv

class PyWRFSettings:
    def __init__(self):
        # Settings go here
        self.DPI = 250.0
        self.output_rootdir = '/tera9/jrlawson/test/'
        self.wrfout_rootdir = '/tera9/jrlawson/bowecho/'

config = PyWRFSettings()
PyWRFPlusEnv(config)
