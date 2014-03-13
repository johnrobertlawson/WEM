import numpy as N
import os
import time
import calendar
import pdb

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

def string_from_time(usage,t,dom=0,strlen=0,conven=0,**kwargs):
    """
    conven  :   convection of MM/DD versus DD/MM
    """


    if isinstance(t,str):
        if usage == 'output':
            usage = 'skip' # Time is already a string
        elif usage == 'title':
            pass
        #    if kwargs['itime']: # For averages or maxima over time
        #        itime = kwargs['itime']
        #        ftime = kwargs['ftime']
        #    else:
        #        pass
        else:
            raise Exception
    elif isinstance(t,int):
        # In this case, time is in datenum. Get it into tuple format.
        t = time.gmtime(t)
    else:
        pass

    if usage == 'title':
        # Generates string for titles
        if not 'itime' in kwargs: # i.e. for specific times
        #if not hasattr(kwargs,'itime'): # i.e. for specific times
            strg = '{3:02d}:{4:02d}Z on {2:02d}/{1:02d}/{0:04d}'.format(*t)
        else: # i.e. for ranges (average over time)
            s1 = '{3:02d}:{4:02d}Z to '.format(*kwargs['itime'])
            s2 = '{3:02d}:{4:02d}Z'.format(*kwargs['ftime'])
            strg = s1 + s2
    elif usage == 'wrfout':
        # Generates string for wrfout file finding
        # Needs dom
        if not dom:
            print("No domain specified; using domain #1.")
            dom = 1
        strg = ('wrfout_d0' + str(dom) +
               '{0:04d}-{1:02d}-{2:02d}_{3:02d}:{4:02d}:{5:02d}'.format(*t))
    elif usage == 'output':
        if not conven:
            # No convention set, assume DD/MM (I'm biased)
            conven = 'full'
        # Generates string for output file creation
        if conven == 'DM':
            strg = '{2:02d}{1:02d}_{3:02d}{4:02d}'.format(*t)
        elif conven == 'MD':
            strg = '{1:02d}{2:02d}_{3:02d}{4:02d}'.format(*t)
        elif conven == 'full':
            strg = '{0:04d}{1:02d}{2:02d}{3:02d}{4:02d}'.format(*t)
        else:
            print("Set convention for date format: DM or MD.")
    elif usage == 'dir':
        # Generates string for directory names
        # Needs strlen which sets smallest scope of time for string
        if not strlen:
             print("No timescope strlen set; using hour as minimum.")
             strlen = 'hour'
        n = lookup_time(strlen)
        strg = "{0:04d}".format(t[0]) + ''.join(
                ["{0:02d}".format(a) for a in t[1:n+1]])
    elif usage == 'skip':
        strg = t
    else:
        print("Usage for string not valid.")
        raise Exception
    return strg

def lookup_time(str):
    D = {'year':0, 'month':1, 'day':2, 'hour':3, 'minute':4, 'second':5}
    return D[str]

def get_level_naming(lv,va,vardict):
    if lv < 1500:
        return str(lv)+'hPa'
    elif lv == 2000:
        return 'sfc'
    elif lv.endswith('K'):
        return lv
    elif lv.endswith('PVU'):
        return lv
    elif lv.endswith('km'):
        return lv
    elif lv == 'all':
        if va == 'shear':
            name = '{0}to{1}'.format(vardict['bottom'],vardict['top'])
            return name
        else:
            return 'all_lev'


def level_type(lv):
    """ Check to see what type of level is requested by user.
        
    """
    if lv < 1500:
        return 'isobaric'
    elif lv == 2000:
        return 'surface'
    elif lv.endswith('K'):
        return 'isentropic'
    elif lv.endswith('PVU'):
        return 'PV-surface'
    elif lv.endswith('km'):
        return 'geometric'
    elif lv == 'all':
        return 'eta'
    else:
        print('Unknown vertical coordinate.')
        raise Exception
        
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

def generate_times(idate,fdate,interval):
    """
    Interval in seconds
    """
    i = calendar.timegm(idate)
    f = calendar.timegm(fdate)
    times = range(i,f,interval)
    return times

