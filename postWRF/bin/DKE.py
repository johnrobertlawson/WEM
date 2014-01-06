"""This script shows examples of using the package to create arrays
of data stored to disc. This is then plotted using the package.
"""

from DKE_settings import Settings
from ../postWRF import WRFEnviron

# Initialise settings and environment
config = Settings()
p = WRFEnviron(config)

# User settings
init_time = p.string_from_time('dir',(2009,9,10,0,0,0),strlen='hour')
rootdir = '/uufs/chpc.utah.edu/common/home/horel-group2/lawson2/'
outdir = '/uufs/chpc.utah.edu/common/home/u0737349/public_html/paper2/'

for rundate in ('25','27','29'):
    foldername = '201111' + rundate + '00'
    runfolder = os.path.join(rootdir,foldername)
    path_to_wrfouts = p.wrfout_files_in(wrfout_dir,dom=1)

    path_to_plots = os.path.join(outdir,runfolder)

    # Produce .npy data files with DKE data
    p.compute_diff_energy('sum_z','kinetic',path_to_wrfouts,times,upper=500,
                          d_save=runfolder, d_return=0,d_fname='DKE_'+foldername)
    # Plot these .npy files
    p.plot_diff_energy('sum_z',times,runfolder,path_to_plots)
