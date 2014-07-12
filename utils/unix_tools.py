"""
Utility scripts to help with directory, file, etc issues
"""

import os

def dir_from_naming(self,root,*args):
	"""
	Generate file path from arguments

	Inputs:
	root	:	file path base
	args 	:	list of arguments to join as separate folders
	"""
    l = [str(a) for a in args]
    path = os.path.join(root,*l)
    return path

