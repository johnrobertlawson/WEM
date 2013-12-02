"""Create cross-sections through WRF data.

This can be time-height or distance-height. 
The height can be pressure, model, geometric, or geopotential

The output can be saved to a pickle file.
This can be useful for creating composites

"""

# Imports
import numpy as N

class getCrossSection(config):
    """
    Create cross-section as specified in config (settings).
    """

    def __init__(self):
        
     
