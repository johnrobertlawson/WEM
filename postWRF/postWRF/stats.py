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

def blur_image(im, n, ny=None, pad=0) :
    """ 
    Taken from scipy cookbook online.
    Blur the image by convolving with a gaussian kernel of typical
    size n. The optional keyword argument ny allows for a different
    size in the y direction.

    Pad     :   put zeros on edge of length n so that output
                array equals input array size.
    """
    g = gauss_kern(n, sizey=ny)
    improc = scipy.signal.convolve(im,g, mode='valid')
    if pad:
        improc = N.pad(improc,n,'constant')
    return(improc)
