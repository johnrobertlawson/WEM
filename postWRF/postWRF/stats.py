"""
Scripts involving statistics or data manipulation here.
"""

import numpy as N
import pdb
import scipy
import scipy.signal

from wrfout import WRFOut

def std(ncfiles,va,tidx,lvidx):
    """
    Find standard deviation in along axis of ensemble
    members. Returns matrix x-y for plotting
    """
    for n, nc in enumerate(ncfiles):
        W = WRFOut(nc)
        slices = {'lv':lvidx, 't':tidx}
        va_array = W.get(va,slices)
        dims = va_array.shape

        if n==0:
            all_members = N.zeros([len(ncfiles),1,1,dims[-2],dims[-1]])
        all_members[n,0,0,:,:] = va_array[...]

    std = N.std(all_members,axis=0).reshape([dims[-2],dims[-1]])
    # pdb.set_trace()
    return std

def gauss_kern(size, sizey=None):
    """
    Taken from scipy cookbook online.
    Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = scipy.mgrid[-size:size+1, -sizey:sizey+1]
    g = scipy.exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()

def gauss_smooth(data, n, ny=None, pad=1, pad_values=0) :
    """
    Taken from scipy cookbook online.
    Blur the data by convolving with a gaussian kernel of typical
    size n. The optional keyword argument ny allows for a different
    size in the y direction.

    :param data:    data (2D only?)
    :type data:     N.ndarray
    :param pad:     put zeros on edge of length n so that output
                    array equals input array size.
    :type pad:      bool
    :param pad_values:  if pad, then use this value to fill edges.
    :type pad_values:   int,float

    """
    # Create list from fill values
    if pad_values == 'nan':
        constant_values = [0,]
    else:
        constant_values = [pad_values,]

    g = gauss_kern(n, sizey=ny)
    dataproc = scipy.signal.convolve(data,g, mode='valid')
    if pad:
        dataproc = N.pad(dataproc,n,'constant',constant_values=constant_values)
        if pad_values=='nan':
            dataproc[dataproc==0] = N.nan
    return(dataproc)

def compute_diff_energy(ptype,energy,files,times,upper=None,lower=None,
                        d_save=True,d_return=True,d_fname='diff_energy_data'):
    """
    This method computes difference kinetic energy (DKE)
    or different total energy (DTE, including temp)
    between WRFout files for a given depth of the
    atmosphere, at given time intervals

    :param ptype:   'sum_z' or 'sum_xyz'.
                    'sum_z' integrates vertically between lower and
                    upper hPa and creates a time series.
                    'sum_xyz' integrates over the 3D space (again between
                    the upper and lower bounds) and creates 2D arrays.
    :param energy:   'kinetic' or 'total'
    :param upper:   upper limit of vertical integration
    :param lower:   lower limit of vertical integration
    :param files:   abs paths to all wrfout files
    :param times:   times for computations - tuple format
    :param d_save:   save dictionary to folder (path to folder)
    :param d_return:   return dictionary (True or False)
    :param d_fname:   custom filename

    :returns: N.ndarray -- time series or list of 2D arrays

    """
    if d_save and not isinstance(d_save,basestring):
        d_save = os.environ['HOME']

    # First, save or output? Can't be neither!
    if not d_save and not d_return:
        print("Pick save or output, otherwise it's a waste of computer"
                "power")
        raise Exception

    print("Saving pickle file to {0}".format(d_save))
    # Look up the method to use depending on type of plot
    PLOTS = {'sum_z':self.DE_z, 'sum_xyz':self.DE_xyz}

    print('Get sequence of time')
    # Creates sequence of times
    ts = utils.get_sequence(times)

    # Dictionary of data
    DATA = {}

    print('Get permutations')
    # Get all permutations of files
    nperm = len(list(itertools.combinations(files,2)))
    print('Start loop')
    # pdb.set_trace()
    for n, perm in enumerate(itertools.combinations(files,2)):
        print("No. {0} from {1} permutations".format(n,nperm))
        perm_start = time.time()
        DATA[str(n)] = {}
        f1, f2 = perm
        W1 = WRFOut(f1)
        W2 = WRFOut(f2)
        print('WRFOuts loaded.')
        #pdb.set_trace()
        # Make sure times are the same in both files
        if not N.all(N.array(W1.wrf_times) == N.array(W2.wrf_times)):
            print("Times are not identical between input files.")
            raise Exception
        else:
            print("Passed check for identical timestamps between "
                  "NetCDF files")

        # Find indices of each time
        print('Finding time indices')
        t_idx = []
        for t in ts:
            t_idx.append(W1.get_time_idx(t))

        print("Calculating values now...")
        DATA[str(n)]['times'] = ts
        DATA[str(n)]['values'] = []
        for t in t_idx:
            DATA[str(n)]['values'].append(PLOTS[ptype](W1.nc,W2.nc,t,
                                            energy,lower,upper))
        DATA[str(n)]['file1'] = f1
        DATA[str(n)]['file2'] = f2

        print "Calculation #{0} took {1:2.2f} seconds.".format(n,time.time()-perm_start)

    if d_return and not d_save:
        return DATA
    elif d_save and not d_return:
        #self.save_data(DATA,d_save,d_fname)
        self.save_data(DATA,d_save,d_fname)
        #self.json_data(DATA,d_save,d_fname)
        return
    elif d_return and d_save:
        #self.save_data(DATA,d_save,d_fname)
        self.save_data(DATA,d_save,d_fname)
        #self.json_data(DATA,d_save,d_fname)
        return DATA

def DE_xyz(self,nc0,nc1,t_idx,energy,*args):
    """
    Computation for difference kinetic energy (DKE).
    Sums DKE over the 3D space, returns a time series.

    Destaggering is not enabled as it introduces
    computational cost that is of miniscule value considering
    the magnitudes of output values.

    Inputs:

    nc0     :   netCDF file
    nc1     :   netCDF file
    t_idx   :   times indices to difference
    energy  :   kinetic or total
    *args   :   to catch lower/upper boundary which isn't relevant here

    Outputs:

    data    :   time series.
    """
    # Wind data
    U0 = nc0.variables['U']
    V0 = nc0.variables['V']
    U1 = nc1.variables['U']
    V1 = nc1.variables['V']

    if energy=='total':
        T0 = nc0.variables['T']
        T1 = nc1.variables['T']
        R = 287.0 # Universal gas constant (J / deg K * kg)
        Cp = 1004.0 # Specific heat of dry air at constant pressure (J / deg K * kg)
        kappa = (R/Cp)

    xlen = U0.shape[2]

    DKE = []
    for n,t in enumerate(t_idx):
        print("Finding DKE at time {0} of {1}.".format(n,len(t)))
        DKE_hr = 0   # Sum up all DKE for the 3D space
        for i in range(xlen):
            if energy=='kinetic':
                DKE_hr += N.sum(0.5*((U0[t,:,:,i]-U1[t,:,:,i])**2 +
                            (V0[t,:,:-1,i]-V1[t,:,:-1,i])**2))
            elif energy=='total':
                DKE_hr += N.sum(0.5*((U0[t,:,:,i]-U1[t,:,:,i])**2 +
                            (V0[t,:,:-1,i]-V1[t,:,:-1,i])**2 +
                            kappa*(T0[t,:,:,i]-T1[t,:,:,i])**2))
        print("DTE at this time: {0}".format(DKE_hr))
        DKE.append(DKE_hr)
    return DKE

def DE_z(self,nc0,nc1,t,energy,lower,upper):
    """
    Computation for difference kinetic energy (DKE).
    Sums DKE over all levels between lower and upper,
    for each grid point, and returns a 2D array.

    Destaggering is not enabled as it introduces
    computational cost that is of miniscule value considering
    the magnitudes of output values.

    Method finds levels nearest lower/upper hPa and sums between
    them inclusively.

    Inputs:

    nc0     :   netCDF file
    nc1     :   netCDF file
    t       :   times index to difference
    energy  :   kinetic or total
    lower   :   lowest level, hPa
    upper   :   highest level, hPa

    Outputs:

    data    :   2D array.
    """

    # Speed up script by only referencing data, not
    # loading it to a variable yet

    # WIND
    U0 = nc0.variables['U'][t,...]
    U1 = nc1.variables['U'][t,...]
    Ud = U0 - U1
    #del U0, U1

    V0 = nc0.variables['V'][t,...]
    V1 = nc1.variables['V'][t,...]
    Vd = V0 - V1
    #del V0, V1

    # PERT and BASE PRESSURE
    if lower or upper:
        P0 = nc0.variables['P'][t,...]
        PB0 = nc0.variables['PB'][t,...]
        Pr = P0 + PB0
        #del P0, PB1
        # Here we assume pressure columns are
        # roughly the same between the two...

    if energy=='total':
        T0 = nc0.variables['T'][t,...]
        T1 = nc1.variables['T'][t,...]
        Td = T0 - T1
        #del T0, T1

        R = 287.0 # Universal gas constant (J / deg K * kg)
        Cp = 1004.0 # Specific heat of dry air at constant pressure (J / deg K * kg)
        kappa = R/Cp

    xlen = Ud.shape[1] # 1 less than in V
    ylen = Vd.shape[2] # 1 less than in U
    zlen = Ud.shape[0] # identical in U & V

    # Generator for lat/lon points
    def latlon(nlats,nlons):
        for i in range(nlats): # y-axis
            for j in range(nlons): # x-axis
                yield i,j

    DKE = []
    DKE2D = N.zeros((xlen,ylen))
    print_time = ''.join((nc0.variables['Times'][t]))
    print("Calculating 2D grid for time {0}...".format(print_time))
    gridpts = latlon(xlen,ylen)
    for gridpt in gridpts:
        i,j = gridpt
        # Find closest level to 'lower', 'upper'
        if lower or upper:
            P_col = Pr[:,j,i]
        if lower:
            low_idx = utils.closest(P_col,lower*100.0)
        else:
            low_idx = None
        if upper:
            upp_idx = utils.closest(P_col,upper*100.0)+1
        else:
            upp_idx = None

        zidx = slice(low_idx,upp_idx)

        if energy=='kinetic':
            DKE2D[j,i] = N.sum(0.5*((Ud[zidx,j,i])**2 +
                                (Vd[zidx,j,i])**2))
        elif energy=='total':
            DKE2D[j,i] = N.sum(0.5*((Ud[zidx,j,i])**2 +
                                (Vd[zidx,j,i])**2 +
                                kappa*(Td[zidx,j,i])**2))

    DKE.append(DKE2D)

    return DKE
