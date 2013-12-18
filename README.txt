===
WEM
===

WRF Ensemble Management can help you create WRF ensembles from GEFS reanalysis
data via submodule "lazyWRF". It can also post-process the data (such as 
computing ensemble means, creating postage-stamp plots of all members, etc)
via submodule "postWRF".

Where do I start?
=================

```./lazyWRF/``` contains scripts that form the basis of automating your WRF 
ensemble runs. ```./postWRF/bin/``` contains examples of post-processing that
you may like to perform with the module. The other essential file you will need 
to personalise for post-processing is /bin/settings.py. The class therein contains
all the settings for loading data, saving output, etc. Almost all settings can be
left as default (by not specifying a setting), other than essentials like 
the path to your WRF data, the path to output figures, etc.

Contributors & Attributions
===========================

Some files or methods contain attributions to other programmers whose
code has been refactored to fit this project. In summary, thanks to:

HootPy/PulsatrixWx project
--------------

URL: http://www.atmos.washington.edu/~lmadaus/pyscripts.html

* David-John Gagne
* Tim Supinie
* Luke Madaus

PyWRF project (Monash)
----------------------

URL: http://code.google.com/p/pywrf/
URL: https://github.com/scaine1/pyWRF/

