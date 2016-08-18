import numpy as N
import os
import matplotlib as M
import matplotlib.pyplot as plt
import pdb
import scipy
import astropy.convolution

from .figure import Figure

class HeatMap(Figure):
    def __init__(self,data,ax=False,fig=False,figsize=(6,6)):
        self.matrix = data
        super(HeatMap,self).__init__(figsize=figsize)

    def interpolate_nans(self,arr1d):
        num = -N.isnan(arr1d)
        xp = num.nonzero()[0]
        fp = arr1d[-N.isnan(arr1d)]
        x = N.isnan(arr1d).nonzero()[0]
        arr1d[N.isnan(arr1d)] = N.interp(x,xp,fp)
        return arr1d

    def plot(self,outdir,flipsign=False,cmap=M.cm.Reds,alpha=0.99,
                xlabels=False,ylabels=False,x2labels=False,
                fname='heatmap.png',blank_nan=False,
                tickx_top=False,invert_y=False,
                local_std=False):
        """Plot heatmap.

        Args:
            outdir (str): absolute path to output directory
            flipsign (bool): multiplies data by -1.
            cmap (matplotlib.colormap object): colour scheme
            alpha (float): transparency of heatmap
            xlabels, ylabels (list,tuple): strings for labelling axes
            x2labels (list,tuple): enables second x axis for labelling
            fname (str): file name for output image
            local_std (int): if zero, do not overlay local standard deviation.
                             if positive integer, this is the kernel size
                             to compute local std.
        Returns:
            (None)
        """
        
        if not blank_nan:
            matrix = self.interpolate_nans(self.matrix)
            matrix_norm = ((matrix-N.mean(matrix))/(matrix.max()-matrix.min()))
            # +matrix.min()
        else:
            matrix = self.matrix
            matrix_norm = ((matrix-N.nanmean(matrix))/(N.nanmax(matrix)-N.nanmin(matrix)))
            matrix_norm += -N.nanmin(matrix_norm)

        # Normalise to 0 to 1 across whole climatology

        if flipsign:
            matrix_norm = matrix_norm*-1

        # Now plot
        # fig, ax = plt.subplots(dpi=500)
        
        # heatmap = self.ax.pcolor(matrix_norm, cmap=cmap, alpha=alpha)

        # if blank_nan:
        # marr = N.ma.array (matrix_norm, mask=N.isnan(matrix_norm))
        marr = N.ma.masked_invalid(matrix_norm)
        cmap2 = plt.get_cmap(cmap)
        cmap2.set_bad('black')
        heatmap = self.ax.pcolormesh(marr,cmap=cmap2,alpha=alpha)
        
        # Put ticks into centre of each row/column
        self.ax.set_yticks(N.arange(matrix_norm.shape[0]) + 0.5, minor=False)
        self.ax.set_xticks(N.arange(matrix_norm.shape[1]) + 0.5, minor=False)
        if invert_y:
            self.ax.invert_yaxis()
        if tickx_top:
            self.ax.xaxis.tick_top()

        if xlabels is not False:
            # if isinstance(xlabels[0],str):
            self.ax.set_xticklabels(xlabels, minor=False)
        if ylabels is not False:
            self.ax.set_yticklabels(ylabels,minor=False)

        # Make grid prettier
        self.ax.grid(False)

        for tk in self.ax.xaxis.get_major_ticks():
            tk.tick10n = False
            tk.tick20n = False
        for t in self.ax.yaxis.get_major_ticks():
            tk.tick10n = False
            tk.tick20n = False


        if x2labels:
            self.ax2 = self.ax.twinx()
            self.ax2.invert_yaxis()
            self.ax2.set_yticks(N.arange(matrix_norm.shape[0]) + 0.5, minor=False)
            self.ax2.set_xticks(N.arange(matrix_norm.shape[1]) + 0.5, minor=False)
            self.ax2.set_yticklabels([c[0].upper() for c in casetypes],minor=False)
            self.ax2.tick_params(axis='both',which='both',bottom='off',
                    top='off',left='off',right='off')
        # plt.gca().set_axis_direction(left='right')

        self.ax.tick_params(axis='both',which='both',bottom='off',
                    top='off',left='off',right='off')
        self.ax.set_aspect('equal')


        # self.ax.set_xlim(0,19)
        # self.ax2.set_ylim(14,0)
        if local_std:
            conv = self.std_convoluted(matrix_norm,local_std)
            xx = N.arange(0,matrix_norm.shape[1])#[7:]
            yy = N.arange(0,matrix_norm.shape[0])#[:-2]
            # ct = self.ax.contourf(xx,yy,conv,levels=N.arange(0,1.02,0.02),alpha=0.5)
            ct = self.ax.contour(xx,yy,conv,levels=N.arange(0,1.02,0.02),colors='k')
            plt.clabel(ct,inline=1,fontsize=8)


        self.save(outdir,fname)
        # outfpath = os.path.join(outdir,fname)


        # self.fig.tight_layout()
        # self.fig.savefig(outfpath)
        # print(("Saved figure to {0}".format(outfpath)))
        self.matrix_norm = matrix_norm

    def std_convoluted(self, image, nsq):
        # hack!
        # image = image[:-2,7:]

        im = N.array(image, dtype=float)
        im2 = im**2
        ones = N.ones(im.shape)

        kernel = N.ones((2*nsq+1, 2*nsq+1))
        # s = scipy.signal.convolve2d(im, kernel, mode="same")
        s = astropy.convolution.convolve(im, kernel)
        # s2 = scipy.signal.convolve2d(im2, kernel, mode="same")
        s2 = astropy.convolution.convolve(im2, kernel)
        # ns = scipy.signal.convolve2d(ones, kernel, mode="same")
        ns = astropy.convolution.convolve(ones, kernel)

        outarr = N.sqrt((s2 - s**2 / ns) / ns)
        # pdb.set_trace()
        return outarr
