import string
import os
import pdb
import datetime

import numpy as N

"""This module contains the Ensemble class only.

Example:
    Usage here.

Todo:
    * One
    * Two
"""

class Ensemble(object):
    def __init__(self,rootdir,initutc,doms=1,ctrl='ctrl',aux=False,
        model='wrf',fmt='em_real',f_prefix=None,
        output_t=False,history_sec=None,loadobj=True):
        """Class containing all ensemble members. Default is a
            deterministic forecast (i.e. ensemble of one control member).
            Each ensemble member needs to have a separate folder (named
            as the ensemble member's name), but different domains can be
            within a single member's folder.

        TODO: Option to point only to file paths rather than load
        WRFOut, due to processing time for massive ensembles.

        Args:
            rootdir (str):  Directory at root of datafiles
            initutc (datetime.datetime): Initialization time

            doms (int, optional): Number of domains
            ctrl (bool, optional): Whether ensemble has control member
            aux (bool, dict, optional): Dictionary lists, per domain, data
                files that contain additional variables and each
                file's prefix. Default is False (no auxiliary files).
                Not implemented yet.
            enstype (str, optional): Type of ensemble. Default is for
                Weather Research and Forecast (WRF) model.
            fmt (str, optional): The type of simulations. Default is
                real-world simulations (em_real from WRF).
            f_prefix (tuple, optional): Tuple of prefixes for each
                ensemble member's main data files. Must be length /doms/.
                Default is None, which then uses a method to determine
                the file name using default outputs from e.g. WRF.
            output_t (list, optional): Future extension that allows multiple data
                files for each time.
            history_sec (int, optional): Difference between history output
                intervals. If all data is contained in one file, use None
                (default). Methods will keep looking for history output
                files until a file error is raised.
        """
        self.ctrl = ctrl
        self.model = string.lower(model)
        self.rootdir = rootdir
        self.initutc = initutc
        self.doms = doms
        self.fmt = fmt
        self.dt = history_sec
        self.loadobj = loadobj
        #self.aux = aux

        self.isaux = True if isinstance(self.aux,dict) else False
        if len(f_prefix) is not doms:
            raise Exception("Length of main datafile prefixes must "
                                "match number of domains.")
        self.isctrl = True if ctrl else False
        self.nperts = len(pertnames) if pertnames is not False else 0
        self.nmems = self.nperts + self.isctrl
        self.ndoms = len(self.doms)
        self.member_names = []
        self.members = self.get_members()

    def get_members(self,):
        """Create a dictionary with all data.

        Format is:
        members[member][domain][time][data]
        """
        members = {}
        for dom in range(1,self.ndoms+1):
            # Get file name for initialisation time and domain
            main_fname = self.get_data_fname(dom=dom,prod='main')
            if dom in self.aux:
                aux_fname = self.get_data_fname(dom=dom,prod='aux')
            # Each ensemble member has a domain
            for dirname,subdirs,files in os.walk(self.rootdir):
                # If ensemble size = 1, there will be no subdirs.
                if main_fname in files:
                    member = dirname.split('/')[-1]
                    if member not in members:
                        members[member] = {dom:{}}
                    if dom==1:
                        self.member_names.append(member)
                    t = self.initt
                    while True:
                        # Check for history output time
                        fpath = os.path.join(self.rootdir,dirname,main_fname)
                        try:
                            dataobj = self.datafile_object(fpath)
                        except IOError:
                            # All wrfout files have been found
                            break
                        else:
                            members[member][dom][t] = {'dataobj':dataobj,
                                                'fpath':fpath,
                                                'control': (member is self.ctrl)}
                        if dom in self.aux:
                            # TODO: implement
                            fpath = os.path.join(self.rootdir,dirname,aux_fname)
                            dataobj = self.datafile_object(fpath)
                            members[member][dom][t]['auxdataobj'] = dataobj
                            members[member][dom][t]['auxfpath'] = fpath
                            members[member][dom][t]['control'] = member is self.ctrl
                        # Move to next time
                        t = utc + datetime.timedelta(seconds=self.dt)
        return members

    def datafile_object(self,fpath,**kwargs):
        #Extend to include other files (GEFS, RUC etc)
        #TODO: Implement auxiliary wrfout files
        if self.loadobj:
            ops = {'wrf':WRFOut,'aux':AuxWRFOut}
            answer = ops[self.model](fpath,**kwargs)
        else:
            answer = False
        return answer

    def get_gefs_ensemble(self):
        """
        All gefs data files should be in the same folder.
        Each forecast time is a different file.
        """
        members = {}
        # This will break with subset of members, times, a/b grib files...
        allmembers = ['c00',] + ['p{0:02d}'.format(n) for n in range(1,21)]
        allfiles = glob.glob(os.path.join(self.ncroot,'ge*'))
        alltimes = N.arange(0,390,6)

        for ens in allmembers:
            self.memnames.append(ens)
            for t in alltimes:
                utc = self.initt + datetime.timedelta(hours=int(t))
                fname = 'ge{0}.t{1:02d}z.pgrb2f{2:02d}.nc'.format(
                            ens,self.initt.hour,t)
                if os.path.join(self.rootdir,fname) in allfiles:
                    if ens not in members.keys():
                        members[ens] = {}
                    members[ens][utc] = {'data':GEFS(os.path.join(self.rootdir,
                                fname)),'control':ens is 'c00'}
        return members

    def get_prob_threshold(self,vrbl,level,overunder,threshold,itime=False,
                            ftime=False,fcsttime=False,Nlim=False,
                            Elim=False,Slim=False,Wlim=False):
        """
        Return probability of exceeding or reaching a threshold.
        vrbl        :   variable or field
        overunder   :   'over' or 'under'
        threshold   :   the threshold in SI units
        itime       :   (list,tuple) - initial time
        ftime       :   (list,tuple) - final time
        Nlim etc    :   bounding box
        """
        all_ens_data = self.members_array(vrbl,level,itime=itime,ftime=ftime,
                            fcsttime=fcsttime,Nlim=Nlim,Elim=Elim,
                            Slim=Slim,Wlim=Wlim)
        if Nlim:
            all_ens_data,lats,lons = all_ens_data

        if overunder is 'over':
            # True/False if member meets condition
            bool_arr = N.where(all_ens_data > threshold,1,0)
            # Find maximum for all times
            max_arr = N.amax(bool_arr,axis=1)
            # Count members that exceed the threshold
            # And convert to percentage
            count_arr = N.sum(max_arr,axis=0)
            percent_arr = 100*(count_arr/self.nmems)
        elif overunder is 'under':
            # True/False if member meets condition
            bool_arr = N.where(all_ens_data < threshold,1,0)
            # Find minimum for all times
            min_arr = N.amin(bool_arr,axis=1)
            # Count members that exceed the threshold
            # And convert to percentage
            count_arr = N.sum(min_arr,axis=0)
            percent_arr = 100*(count_arr/self.nmems)
        else:
            raise Exception("Pick over or under for threshold comparison.")

        if Nlim:
            return percent_arr[0,:,:],lats,lons
        else:
            return percent_arr[0,:,:]


    def closest_to_mean(self,vrbl,level,fcsttime,Nlim=False,Elim=False,
                            Slim=False,Wlim=False,):
        """
        Find closest member to the mean for given variable
        Passing latlon/box allows calculation over given area
        (Box is in grid spaces)
        """
        all_ens_data = self.members_array(vrbl,level,fcsttime=fcsttime,
                                          Nlim=Nlim,Elim=Elim,
                                          Slim=Slim,Wlim=Wlim)
        if Nlim:
            all_ens_data,lats,lons = all_ens_data
        mean = N.mean(all_ens_data,axis=0)[0,0,:,:]
        diff = N.abs(N.sum((all_ens_data[:,0,0,:,:]-mean),axis=(1,2)))
        ensidx = N.argmin(diff,axis=0)

        return self.members_names[ensidx]

    def ensemble_array(self,vrbl,level=False,itime=False,ftime=False,
                        fcsttime=False,Nlim=False,Elim=False,
                        Slim=False,Wlim=False):
        """
        Returns 5D array
        Ordered in descending order on pert. members
        First dimension is ensemble members.
        Can chop down lat/lon box.

        TODO: lat/lon box is in the correct projection?
        """

        if self.ctrl:
            npert = self.nmems-1
        else:
            npert = self.nmems

        enscount = 0
        for ens in self.members_names:
            if ens is self.ctrl:
                continue
            else:
                enscount += 1

                if itime and ftime:
                    tidx = self.members[ens]['data'].return_tidx_range(
                                                                itime,ftime)
                else:
                    tidx = fcsttime

                if Nlim:
                    data = self.members[ens][tidx]['data'].get(
                                                        vrbl,level=level)
                    ens_data,lats,lons = utils.return_subdomain(
                                                    data,self.examplenc.lats1D,
                                                    self.examplenc.lons1D,Nlim,
                                                    Elim,Slim,Wlim,fmt='latlon')
                else:
                    ens_data = self.members[ens][tidx]['data'].get(
                                                        vrbl,level=level,
                                                        lons=False,lats=False)


                if enscount == 1:
                    w,x,y,z = ens_data.shape
                    all_ens_data = N.zeros((npert,w,x,y,z))
                    del w,x,y,z
                all_ens_data[enscount-1,...] = ens_data

        if Nlim:
            return all_ens_data,lats,lons
        else:
            return all_ens_data

    def mean(self,vrbl,fcsttime,level=False,Nlim=False,Elim=False,
             Slim=False,Wlim=False):
        """
        Returns mean.
        """
        all_ens_data = self.members_array(vrbl,level=level,fcsttime=fcsttime,
                                    Nlim=Nlim,Elim=Elim,Slim=Slim,Wlim=Wlim)
        if Nlim:
            all_ens_data, lats, lons = all_ens_data

        mean = N.mean(all_ens_data,axis=0)

        if Nlim:
            return mean, lats, lons
        else:
            return mean

    def std(self,vrbl,fcsttime,level=False,Nlim=False,Elim=False,Slim=False,Wlim=False):
        """Return standard devation
        """
        all_ens_data = self.members_array(vrbl,level=level,fcsttime=fcsttime,
                                    Nlim=Nlim,Elim=Elim,Slim=Slim,Wlim=Wlim)
        if Nlim:
            all_ens_data, lats, lons = all_ens_data

        std = N.std(all_ens_data,axis=0)

        if Nlim:
            return std, lats, lons
        else:
            return std
