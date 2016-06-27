""" Create time series of variable(s) for certain period at location.

Can use .TS files (?)
Can use model output.

"""
from .wrfout import WRFOut
import matplotlib.pyplot as plt
import os

class TimeSeries:
    def __init__(self,ensemble,latlon,locname):
        # This is a list of instances
        self.ensemble = ensemble
        # lat/lon of location
        self.lat, self.lon = latlon
        self.locname = locname

    def meteogram(self,vrbl,utc=False,outdir=False,ncf=False,
                        nct=False,dom=1):
        fig, ax = plt.subplots()

        for W in self.ensemble:
            # W = self.get_netcdf(enspath,ncf=ncf,nct=nct,dom=dom)
            # W = WRFOut(enspath)
            times = W.utc
            ts = W.get(vrbl,utc=False,lats=self.lat,lons=self.lon)[:,0,0,0]
            if vrbl == 'T2':
                ts -= 273.15
            # import pdb; pdb.set_trace()
            ax.plot(ts)
            print("Plotting ensemble members.")

        fname = 'meteogram_{0}_{1}.png'.format(vrbl,self.locname)
        fpath = os.path.join(outdir,fname)
        fig.savefig(fpath)
        print(("Saved meteogram to {0}".format(fpath)))
