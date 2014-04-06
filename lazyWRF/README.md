The lazyWRF project aims to simplify the WPS/WRF process for a few reasons:
  (1) It makes life easier 
  (2) It reduces silly mistakes that overwrite old output files (notwithstanding bugs in this code!)
  (3) It can maximise efficiency when submitting jobs to a server (i.e. it can do it overnight)
  
Your main file is lazyWRF.py. This is the template for other clever ideas like looping, etc.

It does the following:

    (1) Runs geogrid.exe, ungrib.exe, metgrid.exe, real.exe, and wrf.exe automatically.
    (2) Syncs your namelist.wps and namelist.input so you only have to enter settings once (at the top of the script)
        (i) This includes not having to duplicate everything for each domain
        (ii) And you don't have to work out run_hours etc for the namelist.input
    (3) Automatically downloaded GFS, NAM data you need for your run (others coming soon...)
    (4) Intelligently check your logs to alert you via email of errors (and kill the script at this point)

It will eventually:

    (*) Move existing wrfout*, namelist.input etc files from your previous run into a specific directory
    (*) Be suitable for looping through dates, parameterisation permutations, etc etc
    (*) Merge with other projects (creating ensembles with GEFS data; PyWRFPlus plotting package)

This is a work in progress, and comments are highly encouraged. The final goal is a portable and bug-free collection of standalone code that anyone can download, assuming basic Python package installs.

Please report suggestions, issues, etc to me via GitHub and I will improve and fix the package as we go along. Hopefully there is something useful to come out of this, potentially to be presented at the American Meteorological Society Annual Meeting 2014 in Atlanta, Georgia.

Many thanks, 
John Lawson
Iowa State University
PhD Candidate
September 2013
