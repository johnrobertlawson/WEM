import numpy as N
import matplotlib as M
M.use('Agg')
import matplotlib.pyplot as plt
import pdb
import math
import sys

sys.path.append('/uufs/chpc.utah.edu/common/home/u0737349/lawsonpython/')
import gridded_data

# Constants
rE = 6378100 # radius of Earth in metres

# Settings
plt.rc('text',usetex=True)
fonts = {'family':'Computer Modern','size':16}
plt.rc('font',**fonts)
height, width = (9,17)

outdir = '/uufs/chpc.utah.edu/common/home/u0737349/public_html/thesis/topoxs/'

# Functions
# First, if topography data is not in memory, load it
try:
    dataloaded
except NameError:
    topodata, lats, lons = gridded_data.gettopo()
    dataloaded = 1

def get_map(Nlim,Elim,Slim,Wlim):
    ymax,xmax = gridded_data.getXY(lats,lons,Nlim,Elim) #Here, x is lat, y is lon
    ymin,xmin = gridded_data.getXY(lats,lons,Slim,Wlim) # Not sure why!
    terrain = topodata[xmin:xmax,ymin:ymax]
    xlat = lats[xmin:xmax]
    xlon = lons[ymin:ymax]
    #pdb.set_trace()
    return terrain,xlat,xlon

def get_cross_section(Alat, Alon, Blat, Blon):
    # Now find cross-sections
    Ax, Ay = gridded_data.getXY(lats,lons,Alat,Alon)
    Bx, By = gridded_data.getXY(lats,lons,Blat,Blon)

    # Number of points along cross-section
    xspt = int(N.hypot(Bx-Ax,By-Ay))
    xx = N.linspace(Ay,By,xspt).astype(int)
    yy = N.linspace(Ax,Bx,xspt).astype(int)

    # Get terrain heights along this transect
    heights = topodata[xx,yy]

    # Work out distance along this cross-section in m
    xsdistance = gridded_data.xs_distance(Alat,Alon,Blat,Blon)

    xvals = N.linspace(0,xsdistance,xspt)
    xlabels = ['%3.1f' %(x/1000) for x in xvals]
    # Now plot cross-sections
    fig = plt.figure(figsize=(width,height))
    plt.plot(xvals,heights)
    delta = xspt/10
    plt.xticks(xvals[::delta],xlabels[::delta])
    plt.xlabel('Distance along cross-section (km)')
    fname = 'test1.png'
    plt.savefig(outdir+fname,bbox_inches='tight',pad_inches=0.3)
    plt.clf()
