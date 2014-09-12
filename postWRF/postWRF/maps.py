# from figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import collections
from wrfout import WRFOut
import os

def plot_domains(wrfouts,labels,latlons,outpath,colour=0):
    """
    wrfouts     :   list of wrfout file paths
    latlons     :   dictionary of Nlim,Elim,Slim,Wlim
                    for plot
    """

    fig, ax = plt.subplots(1)

    # Create basemap first of all
    #basemap_res = getattr(self.C,'basemap_res',self.D.basemap_res)
    basemap_res = 'h'

    m = Basemap(
        projection='merc',
        llcrnrlon=latlons['Wlim'],llcrnrlat=latlons['Slim'],
        urcrnrlon=latlons['Elim'],urcrnrlat=latlons['Nlim'],
        lat_0=latlons['lat0'],lon_0=latlons['lon0'],
        resolution=basemap_res,area_thresh=500,
        ax=ax)

    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()

    if not isinstance(colour,collections.Sequence):
        colours = ['k',] * len(wrfouts)
    else:
        colours = colour
    # Get corners of each domain
    for gridlabel,fpath,colour in zip(labels,wrfouts,colours):
        W = WRFOut(fpath)
        print("Plotting domain {0} for {1}".format(gridlabel,fpath))
        #Nlim, Elim, Slim, Wlim = W.get_limits()
        x,y = m(W.lons,W.lats)
        xl = len(x[0,:])
        midpt = len(y[0,:])/2         
        ax.annotate(gridlabel,color=colour,fontsize=10,xy=(x[0,-(0.12*xl)],y[0,midpt]),
                     bbox=dict(fc='white'),alpha=1,va='center',ha='left')    
        m.plot(x[0,:],y[0,:],colour,lw=2)
        ax.plot(x[:,0],y[:,0],colour,lw=2) 
        ax.plot(x[len(y)-1,:],y[len(y)-1,:],colour,lw=2)     
        ax.plot(x[:,len(x)-1],y[:,len(x)-1],colour,lw=2)    

    # fpath = os.path.join(self.C.output_root,'domains.png')
    fname = 'domains.png'
    fpath = os.path.join(outpath,fname)
    fig.savefig(fpath)
    print("Saved to "+fpath)
