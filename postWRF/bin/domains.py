import os
import pdb
import sys
sys.path.append('/home/jrlawson/gitprojects/')

from WEM.postWRF.postWRF import WRFEnviron
from settings import Settings
import WEM.utils as utils
# from WEM.postWRF.postWRF.rucplot import RUCPlot

config = Settings()
p = WRFEnviron()

cases = ('20060526','20130815')
# cases = ('20060526','20090910','20110419','20130815')
labels = [c[:4] for c in cases]
# colours = ('green','blue','red','orange')
# colours = ['black'] * len(labels)

IC = 'GEFSR2'
ens = 'p09'
ex = 'ICBC'

folders = []
for case in cases:
    folders.append(os.path.join(config.wrfout_root,case,IC,ens,ex))

Nlim = 47.0
Elim = -90.0
Slim = 31.0
Wlim = -112.0
outdir = '/home/jrlawson/public_html/bowecho'

wrfouts = utils.wrfout_files_in(folders,descend=0)
p.plot_domains(wrfouts,labels,outdir,Nlim,Elim,Slim,Wlim)

