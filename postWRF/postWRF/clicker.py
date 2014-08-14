import matplotlib.pyplot as plt
import numpy as N

# from figure import Figure

class Clicker(object):
    # def __init__(self,config,wrfout,ax=0):
    def __init__(self,data=0,fig=0,ax=0):
        # super(Clicker,self).__init__(config,wrfout,ax=ax)
        if ax and fig:
            self.ax = ax
            self.fig = fig
        elif ax or fig:
            raise Exception
        else:
            self.fig, self.ax = plt.subplots(1,figsize=(5,5))
        
        if isinstance(data,N.ndarray):
            self.overlay_data(data)
        
    def click_x_y(self):
        self.fig.canvas.mpl_connect('pick_event',self.onpick)
        plt.show(self.fig)
        
    def onpick(self,event):
        artist = event.artist
        mouseevent = event.mouseevent
        self.x = mouseevent.xdata
        self.y = mouseevent.ydata
        
    def overlay_data(self,data,cmap='jet'):
        xlen = data.shape[1]
        ylen = data.shape[0]
        self.ax.imshow(data,cmap=cmap,extent=(0,xlen,0,ylen),
                        picker=5)
        return