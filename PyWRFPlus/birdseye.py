"""Collection of x-y cross-section classes.

Describe each class here.

"""

# Imports
import numpy as N
import matplotlib as M

# Custom imports
import meteogeneral

class BirdsEye:
    def __init__(self,config):
        pass
    
    def plot2D(self,config):
        # This looks a horrible way of doing this...
        plot_args = 1 # How do I get plot_args?
        eval('fig.' + plottype + '(' + plot_args + ')')
