WEM package
===========

Tutorial
--------

This will lead you through an example for automating WRF runs, 
creating statistics from an ensemble, and plotting data.

Installation
------------

Ensure you have `git` installed on your system or server. Then execute ``git
clone https://github.com/johnrobertlawson/WEM.git``. The example scripts are
located in ``WEM/postWRF/bin/``. You can copy a `.py` file from there into your
own personal scripts folder. WEM works best when you don't interact directly
with the codebase, but only change the top-level script.

Let's look at the bare minimum to get plotting. First, you should have the
following:

.. code-block:: python

    import sys
    sys.append('path/to/WEM')

Make sure you change the path to where you have downloaded the WEM codebase
from GitHub. Next:

.. code-block:: python

    from WEM.postWRF import WRFEnviron
    p = WRFEnviron()

This creates an instance of the environment. Now you can use postWRF functions
to generate data and plot figures by calling, for example, ``p.plot2D()``.
Before this, though, define the location of the netCDF data files you use, and
the location to which you want figures saving.

.. code-block:: python

    outdir = '/absolute/path/to/figures/'
    ncdir = '/absolute/path/to/data'

    # If there is more than one netCDF file in the folder,
    # choose one of the following ways to make the selection
    # unambiguous:

    # Time of initiation
    nct = (2006,5,10,12,0,0)
    # Or filename
    ncf = 'wrfout_do1...'

You can also generate a sequence of times. This is useful for iterating plots
over numerous plot times. Don't forget you can iterate over levels, contour
level settings, etc., with the basic Python loops.

.. code-block:: python

    itime = (2006,5,10,18,0,0)
    ftime = (2006,5,11,6,0,0)
    hourly = 3
    times = p.generate_times(itime,ftime,hourly*60*60)

Now here are some example of plots:

.. code-block:: python
    
    # This plots simulated composite reflectivity 
    # Ignore level argument (it is set to False if not specified)
    # as cref does not have a level.
    p.plot2D('cref',utc=itime,outdir=outdir,ncdir=ncdir,ncf=ncf,
                nct=nct,legend=True)

    p.plotstreamlines()

All that's left is executing the script with ``python script.py``, where
`script.py` is your file's name.

More information on the various plots and statistics can be found in the API
section for :class:`WEM.postWRF.postWRF.main`.

Examples
--------

Here are some other useful functions. First, to plot simulated composite
reflectivity for a given time and domain, and then save a second figure showing
verification composite reflectivity (over the US CONUS) on the same domain and
projection, with the same colourbar, use the following:

.. code-block:: python

    p.plot_radar()


To plot accumulated rainfall (combined grid-scale and cumulus
parameterisation), amassed over a number of hours, try this:    

.. code-block:: python

    p.plot_accum_rain(utc,accum_hr,ncdir=ncdir,outdir=outdir)

To plot Difference Kinetic Energy, integrated up to 500 hPa, over a domain,
every six hours, you would first compute the fields (as it is time-consuming,
and it makes sense to save data to file first, in case of reuse), and next plot
this data.

.. code-block:: python

    p.)
    p.

Many functions can accept and return matplotlib figure/axis objects, in case
you want to use WEM's processing capability, but use your own plots. Here's an
example where frontogenesis fields form a four-panel plot (suitable for
publication, for instance):

.. code-block:: python
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2,2)

You can pass a dictionary of locations and their latitude/longitude, and have
these places plotted on the map. It currently works for ``plot2D()`` and
``plot_accum_rain()``. This is an example usage:

.. code-block:: python
    # locs = {'label':(latitude,longitude),etc}
    locs = {'Norman':(35.22,-97.44),'Topeka':(39.06,-95.69)}
    p.plot2D('RAINNC',utc,ncdir=wrf_sd,outdir=out_sd,locations=locs,clvs=N.arange(1,100,2))

