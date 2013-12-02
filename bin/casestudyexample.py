"""Edit this file to taste.

The function create_config returns config, a dictionary of settings
"""

def create_config()
    # Settings
    config = {} # This is the dictionary to be passed to all plotting 

    # Switches to create plots
    config['xsection'] = True
    config['birdseye'] = True
    config['ts'] = True

    # Plotting
    config['plot'] = True
    config['dpi'] = 239.0 
    config['font'] = {'family':'sans-serif','sans-serif':['Liberation Sans'],
                      'weight':'normal','size':14}
    config['latex'] = False

    # Dates, tuple format (YYYY,MM,DD,h,m,s).
    config['casedate'] = (2006,5,26,0,0,0) 
    config['initialtime'] = (2006,5,26,22,0,0)
    config['endtime'] = (2006,5,27,12,0,0)
    config['interval'] = 3.0 # In hours

    # Output pickle files?
    config['pickle'] = False

    return config

PyWRFPlus(config)
