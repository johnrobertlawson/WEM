""" A collection of useful utilities.
"""

def trycreate(loc):
    try:
        os.stat(loc)
    except:
        os.makedirs(loc)
        
def padded_times(obj):
    padded = ['{0:04d}'.format(t) for t in obj.timeseq]
    obj.yr, obj.mth, obj.day, obj.hr, obj.min, obj.sec = padded        