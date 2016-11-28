import os
import pdb

import numpy as N

import pygrib
from .datafile import DataFile

class GribFile(DataFile):
    def __init__(self,fpath):
        self.fpath = fpath
        self.G = pygrib.open(self.fpath)
        self.available_records = [gg for gg in self.G]
        self.G.seek(0) # Screw you GRIB!
        self.available_fields_list = [str(gg).split(':') for gg in self.G]
        # pdb.set_trace()
        self.available_fields_array = N.array(self.available_fields_list)
        self.available_fields = N.unique(self.available_fields_array[:,1])
        self.projection()

    def get_record(self,vrbl,idx=0):
        self.G.seek(0)
        ngg = len(self.G.select(name=vrbl))
        print("There are {0} entries for the variable {1}.".format(ngg,vrbl))
        gg = self.G.select(name=vrbl)[idx]
        return gg

    def get(self,vrbl,idx=0):
        # TODO: look up level in available fields array, return index
        gg = self.get_record(vrbl,idx=idx)
        arr = gg.values
        return arr

    def return_latlon(self):
        gg = self.arbitrary_pick()
        latlon = gg.latlons()
        lats, lons = latlon
        return lats,lons

    def projection(self):
        # self.m = Basemap(projection='npstere',lon_0=-105.0,#lat_1=60.0,
                # llcrnrlon=lllon,llcrnrlat=lllat,urcrnrlon=urlon,urcrnrlat=urlat,
                            # boundinglat=24.701632)
        self.lats, self.lons = self.return_latlon()
        # self.xx, self.yy = self.m(self.lons,self.lats)
        # pdb.set_trace()
        # self.mx, self.my = N.meshgrid(self.xx,self.yy)

        # lllon = -119.023 
        self.lllon = self.lons[0,0]
        # lllat = 23.117 
        self.lllat = self.lats[0,0]
        # urlon = -59.9044 
        self.urlon = self.lons[-1,-1]
        # urlat = 45.6147234 
        self.urlat = self.lats[-1,-1]

        self.shape = self.lats.shape
        assert self.lats.shape == self.lons.shape

    def arbitrary_pick(self):
        vrbl= self.available_fields[0]
        gg = self.get_record(vrbl,idx=0)
        return gg

    def return_vrbl_records(self,vrbl):
        vidx = N.where(self.available_fields_array[:,1]==vrbl)[0]
        for idx in vidx:
            print(self.available_fields_array[idx,:])
        return

