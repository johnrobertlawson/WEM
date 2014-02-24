import numpy as N
import pdb

# This returns the x and y grid point for a lat/lon.
# lats/lons are from the model output
# ptlat/ptlon are desired

def getXY(lats,lons,ptlat,ptlon):
    # Find closest lat/lon in array
    minlat = abs(lats-ptlat).min()
    minlon = abs(lons-ptlon).min()
    # Find where these are in the grid
    wherelat = N.where(abs(lats-ptlat) == minlat)
    wherelon = N.where(abs(lons-ptlon) == minlon)
    y = N.where(lats==lats[wherelat])[0][0]
    x = N.where(lons==lons[wherelon])[0][0]
    exactlat = lats[wherelat]
    exactlon = lons[wherelon]
    return x,y, exactlat, exactlon

def gettopo():
    fname = '/uufs/chpc.utah.edu/common/home/u0737349/dsws/topodata/globe30.bin'
    f = open(fname,'r')
    fdata = N.fromfile(f,dtype='int16')
    # Transposes and reshapes to a lat-lon grid
    # Changes negative values to 0 (sea level)
    xnum = 43200.0
    ynum = 18000.0
    topodata = N.flipud(N.reshape(fdata,(ynum,xnum))).clip(0)
    #topodata = ((N.reshape(fdata,(xnum,ynum))).clip(0))
    f.close(); del fdata
    # Define size of pixels
    xpixel = 360/xnum
    ypixel = 150/ynum # Note only 150 degrees!
    # Create lat/lon grid
    lats = N.arange(-60,90,ypixel)#[::-1]
    lons = N.arange(-180,180,xpixel)#[::-1]
    print 'Topographic data has been loaded. Everest is but a mere pixel.'
    return topodata, lats, lons

def xs_distance(Alat, Alon, Blat, Blon):
    phi1 = N.radians(90.0-Alat)
    phi2 = N.radians(90.0-Blat)
    theta1 = N.radians(Alon)
    theta2 = N.radians(Blon)
    arc = math.acos(math.sin(phi1)*math.sin(phi2)*math.cos(theta1-theta2) +
                    math.cos(phi1)*math.cos(phi2))
    xsdistance = rE * arc
    return xsdistance     

# This dstacks arrays, unless it's the first time through, in which case it initialises the variable
def dstack_loop(data, Dict, Key):
    # Try evaluating dict[key]. If it doesn't exist, then initialise it
    # If it does exist, stack data
    try:
        Dict[Key]
    except KeyError:
        stack = data
        #Dict[Key] = data
    else:
        stack = N.dstack((Dict[Key],data))
    return stack
    pass

# Create thinned pressure levels for skew T barb plotting
def thinned_barbs(pres):
    levels = N.arange(20000.0,105000.0,5000.0)
    plocs = []
    for l in levels:
        ploc = N.where(abs(pres-l)==(abs(pres-l).min()))[0][0]
        plocs.append(ploc)
    thin_locs = N.array(plocs)
    return thin_locs # Locations of data at thinned levels

