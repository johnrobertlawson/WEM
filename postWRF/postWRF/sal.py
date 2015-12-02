import numpy as N
import pdb
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt

from wrfout import WRFOut
from obs import Radar

class SAL(object):
    def __init__(self,Wctrl_fpath,Wmod_fpath,vrbl,utc,lv=False,
                    accum_hr=False,radar_datadir=False):
        self.C = {}
        self.M = {}

        self.M['WRFOut'] = WRFOut(Wmod_fpath)
        self.dx = self.M['WRFOut'].dx
        self.compute_d(self.M['WRFOut'])

        if Wctrl_fpath is False and (vrbl=='REFL_comp' or vrbl=='cref'):
            use_radar_obs = True
            self.C['data'] = self.get_radar_verif(utc,radar_datadir)
        else:
            use_radar_obs = False
            Wctrl = WRFOut(Wctrl_fpath)

        # Get 2D grids for ctrl and model
        if vrbl == 'accum_precip':
            if not accum_hr:
                raise Exception("Need to set accumulation hours.")
            self.C['data'] = Wctrl.compute_accum_rain(utc,accum_hr)[0,0,:,:]
            self.M['data'] = self.M['WRFOut'].compute_accum_rain(utc,accum_hr)[0,0,:,:]
        else:
            self.M['data'] = self.M['WRFOut'].get(vrbl,level=lv,utc=utc)[0,0,:,:]
            if not use_radar_obs:
                self.C['data'] = Wctrl.get(vrbl,level=lv,utc=utc)[0,0,:,:]

        # Set negative values to 0 
        # if vrbl == 'REFL_comp':
        # import pdb; pdb.set_trace()
        self.C['data'][self.C['data']<0] = 0
        self.M['data'][self.M['data']<0] = 0
        self.vrbl = vrbl

        self.identify_objects()
        self.compute_amplitude()
        self.compute_location()
        self.compute_structure()

        print("S = {0}    A = {1}     L = {2}".format(self.S,self.A,self.L))

    def get_radar_verif(self,utc,datapath):
        RADAR = Radar(utc,datapath)
        Nlim, Elim, Slim, Wlim = self.M['WRFOut'].get_limits() 
        wlats = self.M['WRFOut'].lats1D
        wlons = self.M['WRFOut'].lons1D
        data, lats, lons = RADAR.get_subdomain(Nlim,Elim,Slim,Wlim)
        dBZ = RADAR.get_dBZ(data)
        dBZ_flip = N.flipud(dBZ)
        from scipy.interpolate import RectBivariateSpline as RBS
        rbs = RBS(lats[::-1],lons,dBZ_flip) 
        dBZ_interp = rbs(wlats,wlons,)#grid=True)
        # import pdb; pdb.set_trace()
        # fig, ax = plt.subplots(1)
        # ax.imshow(dBZ_interp)
        # ax.invert_yaxis()
        # fig.tight_layout()
        # fig.savefig('/home/jrlawson/public_html/bowecho/hires/SAL/dBZ_output.png')
        return dBZ_interp

    def compute_d(self,W):
        side = W.dx * W.x_dim
        self.d = N.sqrt(side**2 + side**2)
        return

    def identify_objects(self,):
        self.f = 1/15.0 # Used in Wernli et al 2008
        self.C['Rmax'] = N.max(self.C['data'])
        self.M['Rmax'] = N.max(self.M['data'])
        if (self.vrbl == 'REFL_comp') or (self.vrbl=='cref'):
            self.M['Rstar'] = 40.0
            self.C['Rstar'] = 40.0
        else:
            self.C['Rstar'] = self.f * self.C['Rmax']
            self.M['Rstar'] = self.f * self.M['Rmax']
        # Find grid points of local precip max > R*
        # self.C['objects'] = {}

        self.object_operators(self.M)
        self.object_operators(self.C)

    def compute_amplitude(self,):
        self.A = (N.mean(self.M['data']) - N.mean(self.C['data']))/(
                0.5*(N.mean(self.M['data']) + N.mean(self.C['data'])))
        # import pdb; pdb.set_trace()

    def compute_location(self,):
        L1 = self.compute_L1()
        L2 = self.compute_L2()
        self.L = L1 + L2

    def compute_L1(self,):
        # vector subtraction
        dist_km = self.vector_diff_km(self.M['x_CoM'],self.C['x_CoM'])
        L1 = dist_km/self.d
        return L1

    def vector_diff_km(self,v1,v2):
        # From grid coords to km difference
        dist_gp = N.subtract(v1,v2)
        dist_km = self.dx * N.sqrt(dist_gp[0]**2 + dist_gp[1]**2)
        return dist_km

    def compute_L2(self,):
        r_ctrl = self.compute_r(self.C)
        r_mod = self.compute_r(self.M)
        L2 = 2*(N.abs(r_ctrl-r_mod)/self.d)
        return L2
    
    def compute_r(self,dic):
        Rn_sum = 0
        for k,v in dic['objects'].items():
            Rn_sum += v['Rn'] * self.vector_diff_km(dic['x_CoM'],v['CoM']) 
        try:
            r = Rn_sum / dic['R_tot']
        except ZeroDivisionError:
            r = 0
        return r

    def object_operators(self,dic):
        nsize = 50
        thresh = dic['Rstar']
        data = dic['data']

        mask = N.copy(data)
        mask[data<thresh] = False
        mask[data>=thresh] = True

        labeled, num_objects = ndimage.label(mask)

        sizes = ndimage.sum(mask, labeled, range(num_objects+1))

        masksize = sizes < 25
        remove_pixel = masksize[labeled]
        labeled[remove_pixel] = 0

        labels = N.unique(labeled)
        label_im = N.searchsorted(labels, labeled)

        dic['objects'] = {}

        # Total R for objects
        R_objs_count = 0

        for ln,l in enumerate(labels):
            cy, cx = ndimage.measurements.center_of_mass(data,labeled,l)
            # First record is for whole field
            if ln == 0:
                dic['x_CoM'] = (cx,cy)
            else:
                dic['objects'][l] = {}
                dic['objects'][l]['CoM'] = (cx,cy)
                dic['objects'][l]['Rn'] = ndimage.sum(data,labeled,l)
                dic['objects'][l]['RnMax'] = ndimage.maximum(data,labeled,l)
                dic['objects'][l]['Vn'] = dic['objects'][l]['Rn']/dic['objects'][l]['RnMax']
                R_objs_count += dic['objects'][l]['Rn']

        dic['R_tot'] = R_objs_count

        bow_radius = False
        if bow_radius:
            sizes2 = ndimage.sum(mask,label_im, range(len(labels+1)))
            bigidx = N.where(sizes2==N.sort(sizes2)[-1])[0][0]
            # import pdb; pdb.set_trace()
            slicex, slicey = ndimage.find_objects(label_im==bigidx)[0]
            roi_data = data[slicex,slicey]
            roi = label_im[slicex,slicey]
            # markers = N.array([int(slicex.start),int(slicey.start)])
            filt = ndimage.filters.median_filter(roi,size=15.0,mode='constant')
            # filt2 = ndimage.morphology.binary_closing(filt,iterations=1)
            fig,ax = plt.subplots(3,1,figsize=(4,6))
            # ax.imshow(roi)
            # ax.flat[0].imshow(label_im)
            ax.flat[0].imshow(data)
            ax.flat[0].invert_yaxis()
            ax.flat[1].imshow(roi)
            # ax.imshow(label_im)
            ax.flat[1].invert_yaxis()
            ax.flat[2].imshow(filt)
            ax.flat[2].invert_yaxis()
            fig.tight_layout()
            fig.savefig('/home/jrlawson/public_html/bowecho/radiustest.png')
            import pdb; pdb.set_trace()

    def compute_structure(self):
        V_mod = self.compute_V(self.M)
        V_ctrl =  self.compute_V(self.C)
        try:
            self.S = (V_mod - V_ctrl)/(0.5*(V_mod+V_ctrl))
        except ZeroDivisionError:
            self.S = 0

    def compute_V(self,dic):
        Vn_sum = 0
        for k,v in dic['objects'].items():
            Vn_sum += v['Rn'] * v['Vn']
        try:
            V = Vn_sum / dic['R_tot']
        except ZeroDivisionError:
            V = 0
        return V
