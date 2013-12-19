"""This script shows examples of using the package to create arrays
of data stored to disc. This is then plotted using the package.
"""

from DKE_settings import Settings

# Initialise settings and environment
config = Settings()
p = PyWRFEnv(config)

# User settings


# Produce .npy data files with DKE data


# Plot these .npy files
p.plot2D('DKE_2D',path_to_data,path_to_plots)
