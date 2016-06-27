# from figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import collections
from .wrfout import WRFOut
import os

def plot_domains(wrfouts,labels,outpath,Nlim,Elim,
                    Slim,Wlim,colours='k',fname=False,
                    fill_land=False,labpos=False,fill_water=False):
    """
    wrfouts     :   list of wrfout file paths
    """

    if not labpos:
        labpos = ['lr',]*len(labels)
    fig, ax = plt.subplots(1,figsize=(5,5))

    # Create basemap first of all
    #basemap_res = getattr(self.C,'basemap_res',self.D.basemap_res)
    m = Basemap(projection='merc',
                llcrnrlat=Slim,
                llcrnrlon=Wlim,
                urcrnrlat=Nlim,
                urcrnrlon=Elim,
                lat_ts=(Nlim-Slim)/2.0,
                resolution='l',
                ax=ax)
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    if fill_land and not fill_water:
        m.fillcontinents(color=fill_land,lake_color='white')
    elif fill_water and not fill_land:
        m.fillcontinents(color='white',lake_color=fill_water)
        m.drawmapboundary(fill_color=fill_water)
    elif fill_land and fill_water:
        m.fillcontinents(color=fill_land,lake_color=fill_water)
        m.drawmapboundary(fill_color=fill_water)

    if isinstance(colours,(list,tuple)):
        if not len(wrfouts) == len(colours):
            print("Not the correct number of colours")
            raise Exception
    elif isinstance(colours,str):
        colours = [colours,] * len(wrfouts)
    # Get corners of each domain
    for gridlabel,fpath,col,lp in zip(labels,wrfouts,colours,labpos):
        W = WRFOut(fpath)
        print(("Plotting domain {0} for {1}".format(gridlabel,fpath)))
        #Nlim, Elim, Slim, Wlim = W.get_limits()
        x,y = m(W.lons,W.lats)
        xl = len(x[0,:])
        midx = len(x[0,:])/2  
        yl = len(y[0,:])
        midy = len(y[0,:])/2         
        
        if lp == 'lr':
            xylab = (x[0,-(0.2*xl)],y[0,midy])
            halab = 'left'
        elif lp == 'ur':
            xylab = (x[0,-(0.2*xl)],y[-1,midy])
            halab = 'left'
        elif lp == 'll':
            xylab = (x[0,(0.2*xl)],y[0,midy])
            halab = 'right'
        elif lp == 'ul':
            xylab = (x[0,(0.2*xl)],y[-1,midy])
            halab = 'right'
        elif lp == 'lc':
            xylab = (x[0,midx],y[0,midy])
            halab = 'center'
        else:
            raise Exception("Label position needs to be ll, lr, ul, ur.")


        ax.annotate(gridlabel,color=col,fontsize=8,xy=xylab,
                     bbox=dict(fc='white'),alpha=1,va='center',ha=halab)    
        m.plot(x[0,:],y[0,:],col,lw=1.2)
        ax.plot(x[:,0],y[:,0],col,lw=1.2) 
        ax.plot(x[len(y)-1,:],y[len(y)-1,:],col,lw=1.2)     
        ax.plot(x[:,len(x)-1],y[:,len(x)-1],col,lw=1.2)    

    # fpath = os.path.join(self.C.output_root,'domains.png')
    if not fname:
        fname = 'domains.png'
    fpath = os.path.join(outpath,fname)
    fig.tight_layout()
    fig.savefig(fpath)
    print(("Saved to "+fpath))
