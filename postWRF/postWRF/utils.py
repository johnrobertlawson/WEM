import numpy as N
import os
import time

""" A collection of useful utilities.
"""

def trycreate(loc):
    try:
        os.stat(loc)
    except:
        os.makedirs(loc)
        
def padded_times(timeseq):
    padded = ['{0:04d}'.format(t) for t in timeseq]
    return padded

def string_from_time(usage,t,dom=0,strlen=0,conven=0,tupleformat=1):
        #if not tupleformat:
        if isinstance(t,int):
            # In this case, time is in datenum. Get it into tuple format.
            t = time.gmtime(t)
        if usage == 'title':
            # Generates string for titles
            str = '{3:02d}:{4:02d}Z on {2:02d}/{1:02d}/{0:04d}'.format(*t)
        elif usage == 'wrfout':
            # Generates string for wrfout file finding
            # Needs dom
            if not dom:
                print("No domain specified; using domain #1.")
                dom = 1
            str = ('wrfout_d0' + str(dom) +
                   '{0:04d}-{1:02d}-{2:02d}_{3:02d}:{4:02d}:{5:02d}'.format(*t))
        elif usage == 'output':
            if not conven:
                # No convention set, assume DD/MM (I'm biased)
                conven = 'DM'
            # Generates string for output file creation
            if conven == 'DM':
                str = '{2:02d}{1:02d}_{3:02d}{4:02d}'.format(*t)
            elif conven == 'MD':
                str = '{1:02d}{2:02d}_{3:02d}{4:02d}'.format(*t)
            else:
                print("Set convention for date format: DM or MD.")
        elif usage == 'dir':
            # Generates string for directory names
            # Needs strlen which sets smallest scope of time for string
            if not strlen:
                 print("No timescope strlen set; using hour as minimum.")
                 strlen = 'hour'
            n = lookup_time(strlen)
            str = "{0:04d}".format(t[0]) + ''.join(
                    ["{0:02d}".format(a) for a in t[1:n+1]])
        else:
            print("Usage for string not valid.")
            raise Exception
        return str

def lookup_time(str):
    D = {'year':0, 'month':1, 'day':2, 'hour':3, 'minute':4, 'second':5}
    return D[str]

def level_type(lv):
    """ Check to see what type of level is requested by user.
        
    """
    if lv.endswith('K'):
        return 'isentropic'
    elif lv < 1500:
        return 'isobaric'
    elif lv == 2000:
        return 'surface'
    elif lv.endswith('PVU'):
        return 'PV-surface'
    elif lv.endswith('km'):
        return 'geometric'
        
def closest(arr2D,val):
    """
    Inputs:
    val     :   required value
    arr2D     :   2D array of values
    
    Output:
    
    idx     :   index of closest value
    
    """
    idx = N.argmin(N.abs(arr2D - val))
    return idx
    
def dstack_loop(data, obj):
    """
    Tries to stack numpy array (data) into object (obj).
    If obj doesn't exist, then initialise it
    If obj does exist, stack data.
    """
    try:
        print obj
    except NameError:
        stack = data
    else:
        stack = N.dstack((obj,data))
        
    return stack