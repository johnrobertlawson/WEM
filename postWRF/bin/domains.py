import os
import pdb
import sys
sys.path.append('/home/jrlawson/gitprojects/')
import matplotlib as M
M.use('agg')
import matplotlib.pyplot as plt

from WEM.postWRF.postWRF import WRFEnviron
import WEM.utils as utils
# from WEM.postWRF.postWRF.rucplot import RUCPlot

p = WRFEnviron()

cases = {'20130815_hires':'KSOK13'}

labels = ('3km','1km')
# labels = ('ILIN11 3km','ILIN11 1km','KSOK13 3km','KSOK13 1km')

labpos = ('lc','lc')
# labpos = ('lr','lr','ll','ll')

# wrfouts = ('/chinook2/jrlawson/bowecho/20110419_hires/s21/wrfout_d01_2011-04-19_00:00:00',
        # '/chinook2/jrlawson/bowecho/20110419_hires/s21/wrfout_d02_2011-04-19_00:00:00',
        # '/chinook2/jrlawson/bowecho/20130815_hires/wrfout_d01_2013-08-15_00:00:00',
        # '/chinook2/jrlawson/bowecho/20130815_hires/wrfout_d02_2013-08-15_00:00:00',)

wrfouts = ('/chinook2/jrlawson/bowecho/20130815_hires/wrfout_d01_2013-08-15_00:00:00',
        '/chinook2/jrlawson/bowecho/20130815_hires/wrfout_d02_2013-08-15_00:00:00',)

cols = ('red',)*2
landc = 'darkgrey'
waterc = 'lightgrey'
# cols = ('midnightblue','mediumblue','darkgreen','forestgreen')

# Nlim = 50.0
# Elim = -77.0
# Slim = 30.0
# Wlim = -112.0

Nlim = 50.0
Elim = -86.0
Slim = 28.0
Wlim = -115.0

outdir = '/home/jrlawson/public_html/bowecho/paper2'
fname = 'domains_hires.png'

p.plot_domains(wrfouts,labels,outdir,fname,Nlim,Elim,Slim,Wlim,fill_land=landc,
                labpos=labpos,colours=cols,fill_water=waterc)

