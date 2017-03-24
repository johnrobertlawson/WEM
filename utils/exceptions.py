"""Custom exceptions.
"""

class FormatError(Exception):
    """Data is the wrong format or shape."""
    def __init__(self,error='Format error.'):
        print(error)

class QCError(Exception):
    """Data has failed a quality control check.
    """
    def __init__(self,error='Quality Control error.',pass_idx=False):
        """
        Optional:
        Error (str) -   Message to send to user   
        pass_idx (tuple, list, N.ndarray)   -   Location in data of bad vals.
        """
        print(error)
        if pass_idx is not False:
            if isinstance(pass_idx,(list,tuple)):
                print("Bad data is here: \n")
                [print(p) for p in pass_idx]
            else:
                print(pass_idx)


