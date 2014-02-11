class Settings:
    def __init__(self):
        # Required settings:
        self.output_root = '/home/jrlawson/public_html'
        self.wrfout_root = '/tera9/jrlawson/' 
        # Optional settings:
        self.DPI = 120.0
        self.plot_titles = True
        self.basemap = True
        self.terrain = False
        if self.terrain:
            self.terrain_data_path = '/path/to/terrain/data'
        self.scales = {'wind':(5,65,5)}
        self.colorbar = True
