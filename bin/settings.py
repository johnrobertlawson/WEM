class PyWRFSettings:
    def __init__(self):
        # Required settings:
        self.output_dir = '/home/jrlawson/public_html/test'
        
        # Optional settings:
        self.DPI = 250.0
        self.plot_titles = False
        self.basemap = True
        self.terrain = False
        if self.terrain:
            self.terrain_data_path = '/path/to/terrain/data'