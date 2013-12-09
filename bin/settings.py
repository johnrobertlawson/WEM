class PyWRFSettings:
    def __init__(self):
        # Required settings:
        self.output_rootdir = '/tera9/jrlawson/test'
        self.wrfout_rootdir = '/tera9/jrlawson/bowecho'
        self.output_dir = '/home/jrlawson/public_html/test'
        self.wrfout_prefix = 'wrfout'
        # Optional settings:
        self.DPI = 250.0
        #self.domain = 1 
