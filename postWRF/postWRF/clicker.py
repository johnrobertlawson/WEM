import matplotlib as M
M.use('gtkagg')

import matplotlib.pyplot as plt
import numpy as N
import collections
import pdb

# from figure import Figure
import colourtables as ct
from scales import Scales
from figure import Figure
from defaults import Defaults

class Clicker(Figure):
    # def __init__(self,config,wrfout,ax=0):
    def __init__(self,config,wrfout,data=0,fig=0,ax=0):
        # import pdb; pdb.set_trace()
        self.C = config
        self.D = Defaults()
        self.W = wrfout

        if isinstance(fig,M.figure.Figure):
            self.fig = fig
            self.ax = ax
        else:
            super(Clicker,self).__init__(config,wrfout,fig=fig,ax=ax)
        
        self.bmap,self.x,self.y = self.basemap_setup()
        if isinstance(data,N.ndarray):
            # Lazily assuming it's reflectivity
            S = Scales('cref',2000)
            self.overlay_data(data,V=S.clvs,cmap=S.cm)
        
    def click_x_y(self,plotpoint=0):
        """
        plotpoint       :   boolean. If yes, plot point.
        """
        # self.plotpoint = plotpoint
        # self.fig.canvas.mpl_connect('pick_event',self.onpick)
        # self.point = M.patches.Circle((0,0),radius=1, color='g')
        # self.ax.add_patch(self.point)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release_point)
        plt.show(self.fig)

    def draw_box(self):
        self.rect = M.patches.Rectangle((0,0),1,1)
        self.ax.add_patch(self.rect)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release_box)
        plt.show(self.fig)

    def draw_line(self):
        self.line = M.lines.Line2D((0,0),(1,1))
        self.ax.add_line(self.line)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release_line)
        plt.show(self.fig)
        
    def on_press(self, event):
        print 'press'
        self.x0 = event.xdata
        self.y0 = event.ydata

    def on_release_point(self,event):
        # self.point.set_xy((self.x0,self.y0))
        if hasattr(self,'scatt'):
            if isinstance(self.scatt, M.collections.PathCollection):
                self.scatt.remove()
        self.scatt = self.ax.scatter(self.x0,self.y0,marker='x')
        self.ax.figure.canvas.draw()

    def on_release_box(self, event):
        print 'release'
        self.x1 = event.xdata
        self.y1 = event.ydata
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy((self.x0, self.y0))
        self.ax.figure.canvas.draw()

    def on_release_line(self, event):
        print 'release'
        self.x1 = event.xdata
        self.y1 = event.ydata
        # self.rect.set_width(self.x1 - self.x0)
        # self.rect.set_height(self.y1 - self.y0)
        self.line.set_data((self.x0, self.x1),(self.y0,self.y1))
        self.ax.figure.canvas.draw()
        
    def onpick(self,event):
        artist = event.artist
        mouseevent = event.mouseevent
        self.x = mouseevent.xdata
        self.y = mouseevent.ydata
        
    def overlay_data(self,data,V=0,cmap=0):
        xlen = data.shape[1]
        ylen = data.shape[0]
        kwargs = {}
        
        if isinstance(V,N.ndarray):
            kwargs['levels'] = V
            
        kwargs['cmap'] = cmap
        kwargs['extent'] = (0,xlen,0,ylen)
        kwargs['picker'] = 5

        self.cf = self.bmap.contourf(self.x,self.y,data,**kwargs)
        # self.fig.colorbar(cf,ax=self.ax,shrink=0.5,orientation='horizontal')
        # pdb.set_trace()
        
        return
    
    def set_box_width(self,X):
        """
        Ask user to specify a width that is normal to the
        cross-section X. The plot will show with the box displayed.
        If the user is not happy, they can try again.
        """
        plt.show(self.fig)
        user_is_happy = 0
        while not user_is_happy:
            self.km = int(raw_input("Specify line-normal width (km): "))
            if not isinstance(self.km,int):
                print("Value is not integer.")
                raise Exception
                
            self.rect = M.patches.Rectangle((self.x0,self.y0),X.hyp_pts,X.angle)
            self.ax.add_patch(self.rect)
            self.ax.figure.canvas.draw()

            plt.show(self.fig)
            
            while True:
                doesitwork = raw_input("Does this work? (y/n/x): ")
                if doesitwork == 'y':
                    user_is_happy = 1
                    break
                elif doesitwork == 'n':
                    break
                elif doesitwork == 'x':
                    raise Exception
                else:
                    print("Try again.")
                    
