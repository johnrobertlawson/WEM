# Scripts to help with directory, file, etc issues

import os

# Create folder if it doesn't exist, along with its subdirectories
def createfolder(dir):
    try:
        os.stat(dir)
    except:
        os.makedirs(dir)
        print 'Creating directory',dir
    
