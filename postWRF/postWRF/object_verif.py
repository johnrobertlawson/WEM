import os
import numpy as N
import pdb
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt

from .sal import SAL

class ObjVerif(object):
    def __init__(self,data,thresh,footprint):
        self.data = data
        self.footprint = footprint
        self.thresh = thresh

    def identify(self,):
        # nsize = self.footprint

        mask = N.copy(self.data)
        mask[self.data<self.thresh] = False
        mask[self.data>=self.thresh] = True

        labeled, num_objects = ndimage.label(mask)

        sizes = ndimage.sum(mask, labeled, list(range(num_objects+1)))

        masksize = sizes < self.footprint
        remove_pixel = masksize[labeled]
        labeled[remove_pixel] = 0

        labels = N.unique(labeled)
        self.label_im = N.searchsorted(labels, labeled)


