import numpy as N
import pdb
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt

from .wrfout import WRFOut
from .obs import Radar

class SAL(object):
    def __init__(self,Wctrl_fpath,Wmod_fpath,vrbl=False,utc=False,lv=False,
                    accum_hr=False,radar_datadir=False,thresh=False,
                    footprint=500,ctrl_fmt='obs',mod_fmt='WRF',dx=False,dy=False,
                    f=1/15.0,datafmt=False):
        # Set data formats for both inputs to override defaults
        if datafmt:
            ctrl_fmt = datafmt
            mod_fmt = datafmt
        self.utc = utc
        self.C = {}
        self.M = {}
        self.thresh = thresh
        self.footprint = footprint
        self.f = f

        if (vrbl == 'accum_precip') and not accum_hr:
            raise Exception("Need to set accumulation hours.")

        # First, set diagonal length
        # And load model data

        if mod_fmt == 'WRF':
            self.M['WRFOut'] = WRFOut(Wmod_fpath)
            self.dx = self.M['WRFOut'].dx
            self.dy = self.M['WRFOut'].dy
            self.x_dim = self.M['WRFOut'].x_dim
            self.y_dim = self.M['WRFOut'].y_dim
            self.compute_d()

            if vrbl == 'accum_precip':
                self.M['data'] = self.M['WRFOut'].compute_accum_rain(utc,accum_hr)[0,0,:,:]
            else:
                self.M['data'] = self.M['WRFOut'].get(vrbl,level=lv,utc=utc)[0,0,:,:]

        elif mod_fmt == 'array':
            self.M['data'] = Wmod_fpath
            self.dx = dx
            self.dy = dy
            self.x_dim, self.y_dim = self.M['data'].shape
            self.compute_d()
        
        # Load verification data
        if ctrl_fmt == 'array':
            self.C['data'] = Wctrl_fpath
        elif (ctrl_fmt == 'obs') and (vrbl=='REFL_comp' or vrbl=='cref'):
            self.C['data'] = self.get_radar_verif(utc,radar_datadir)
        elif ctrl_fmt == "WRF":
            self.C['WRFOut'] = WRFOut(Wctrl_fpath)
            if vrbl == 'accum_precip':
                self.C['data'] = self.C['WRFOut'].compute_accum_rain(utc,accum_hr)[0,0,:,:]
            else:
                self.C['data'] = self.C['WRFOut'].get(vrbl,level=lv,utc=utc)[0,0,:,:]

        self.C['data'][self.C['data']<0] = 0
        self.M['data'][self.M['data']<0] = 0
        self.vrbl = vrbl
       
        """
        fig, ax = plt.subplots(1)
        ax.pcolor(self.C['data'])
        fig.savefig('/home/jrlawson/public_html/bowecho/SALtests/obs_pcolor.png')
        plt.close(fig)
        fig, ax = plt.subplots(1)
        ax.pcolor(self.M['data'])
        fig.savefig('/home/jrlawson/public_html/bowecho/SALtests/mod_pcolor.png')
        plt.close(fig)
        import pdb; pdb.set_trace()
        """

        self.identify_objects()
        self.compute_amplitude()
        self.compute_location()
        self.compute_structure()

        print(("S = {0}    A = {1}     L = {2}".format(self.S,self.A,self.L)))

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

    def compute_d(self):
        xside = self.dx * self.x_dim
        yside = self.dy * self.y_dim
        self.d = N.sqrt(xside**2 + yside**2)
        return

    def identify_objects(self,):
        # self.f = 1/15.0 used in Wernli et al 2008
        self.C['Rmax'] = N.max(self.C['data'])
        self.M['Rmax'] = N.max(self.M['data'])
        if (self.vrbl == 'REFL_comp') or (self.vrbl=='cref'):
            self.M['Rstar'] = float(self.thresh)
            self.C['Rstar'] = float(self.thresh)
        else:
            self.C['Rstar'] = self.f * self.C['Rmax']
            self.M['Rstar'] = self.f * self.M['Rmax']
        # Find grid points of local precip max > R*
        # self.C['objects'] = {}

        self.object_operators(self.M,typ='M')
        self.object_operators(self.C,typ='C')

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
        print(("L1 = {0}".format(L1)))
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
        print(("L2 = {0}".format(L2)))
        return L2
    
    def compute_r(self,dic):
        Rn_sum = 0
        for k,v in list(dic['objects'].items()):
            Rn_sum += v['Rn'] * self.vector_diff_km(dic['x_CoM'],v['CoM']) 
        try:
            r = Rn_sum / dic['R_tot']
        except ZeroDivisionError:
            r = 0
        return r

    def object_operators(self,dic,typ=False):
        nsize = self.footprint
        thresh = dic['Rstar']
        data = dic['data']

        mask = N.copy(data)
        mask[data<thresh] = False
        mask[data>=thresh] = True
        # import pdb; pdb.set_trace()
        labeled, num_objects = ndimage.label(mask)

        sizes = ndimage.sum(mask, labeled, list(range(num_objects+1)))

        masksize = sizes < nsize
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
        dic['obj_array'] = labeled

        """
        fig, ax = plt.subplots(1)
        ccc = ax.pcolor(label_im)
        plt.colorbar(ccc)
        fig.savefig('/home/jrlawson/public_html/bowecho/SALtests/{0}_objects_{1}.png'.format(typ,self.utc))
        plt.close(fig)
        if typ == 'M':
            fig, ax = plt.subplots(1)
            cb1 = ax.pcolormesh(self.M['data'])
            plt.colorbar(cb1)
            fig.savefig('/home/jrlawson/public_html/bowecho/SALtests/mod_pcolor.png')
            plt.close(fig)
        # import pdb; pdb.set_trace()
        else:
            fig, ax = plt.subplots(1)
            cb2 = ax.pcolormesh(self.C['data'])
            plt.colorbar(cb2)
            fig.savefig('/home/jrlawson/public_html/bowecho/SALtests/ctrl_pcolor.png')
            plt.close(fig)
        """

        bow_radius = False
        if bow_radius:
            sizes2 = ndimage.sum(mask,label_im, list(range(len(labels+1))))
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
        # self.obj_dic = dic

    def compute_structure(self):
        V_mod = self.compute_V(self.M)
        V_ctrl =  self.compute_V(self.C)
        try:
            self.S = (V_mod - V_ctrl)/(0.5*(V_mod+V_ctrl))
        except ZeroDivisionError:
            self.S = 0

    def compute_V(self,dic):
        Vn_sum = 0
        for k,v in list(dic['objects'].items()):
            Vn_sum += v['Rn'] * v['Vn']
        try:
            V = Vn_sum / dic['R_tot']
        except ZeroDivisionError:
            V = 0
        return V

    def active_px(self,data='ctrl',fmt='pc'):
        """
        Return number of pixels included in chosen dataset's objects.
        data is set to 'ctrl' or 'mod'.
        Expressed as percentage.
        """
        # import pdb; pdb.set_trace()
        if data == 'ctrl':
            dd = self.C['obj_array']
        elif data == 'mod':
            dd = self.M['obj_array']
        active_px = N.count_nonzero(dd)
        tot_px = dd.size
        # import pdb; pdb.set_trace()
        if fmt == 'pc':
            return (active_px/(tot_px*1.0))*100.0
        else:
            return active_px, tot_px
