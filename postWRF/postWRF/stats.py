import numpy as N
import pdb

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
