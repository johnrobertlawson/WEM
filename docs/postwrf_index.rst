postWRF: post-processing
========================

Create basic plots from your WRF runs.

First steps
-----------
You first need to create a script that contains your general settings. Name it something like `settings.py`. Include these four lines at the top.::

	class Settings:
    	def __init__(self):
        	self.output_root = '/home/user/images/path/'
        	self.wrfout_root = '/home/user/data/path'

The `output_root` is the location you'd like .png images to be saved to. The `wrfout_root` specifies the location of the netCDF wrfout files. Both are only roots, and you can modify subfolders and filenames later on-the-fly. Next, create a script that calls the plotting functions. Perhaps call it `plot.py`.::

	from settings import Settings
	from WEM.postWRF import WRFEnviron

	config = Settings()
	p = WRFEnviron(config)

This creates a configuration that gets passed into the WRF Environment (called `p` for plotting in this case).

Dictionaries
------------

Whereas your `settings.py` file contains constant settings, more flexible settings go into a dictionary. Let's create one in `plot.py`. Each variable we want plotting should be a nested dictionary too, as so::

	variables = {'cref':{}, 'wind':{}}

We now specify the required level(s) and time(s) to plot. A bounding box can be specified but this is covered later. Here's an example::

	variables['cref'] = {'lv':2000, 'pt':(2011,,4,19,18,0,0)}

This will plot simulated composite reflectivity at the surface (2000) for the time specified by `pt`.

 