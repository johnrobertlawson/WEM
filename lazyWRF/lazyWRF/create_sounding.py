import csv
import os
import copy
import pdb

import numpy as N
N.set_printoptions(precision=4,suppress=True)
import scipy.integrate as integrate

import WEM.utils as utils
from WEM.utils import metconstants as mc
import WEM.postWRF.postWRF.skewt2 as sT

"""
ROWS:
line 1 is surface information
lines 2 onwards is a point in profile

COLUMNS:
column 1 is surface pressure (hPa) for row 1
--- set this as 1000.00.
otherwise it is height (m) for rows 2 onwards
--- increasing in height with row number

column 2 is potential temperature (K)
--- set row 1 as 300.0 K as reference, probably

column 3 is vapour mixing ratio (g/kg)
--- set row 1 as 14.00 g/kg, for instance

column 4 is U wind (m/s)
--- no row 1

column 5 is V wind (m/s)
--- no row 1

METHODS FOR PROFILE GENERATION
Temp:
    - Weisman and Klemp
    etc
Wind:
    - Blah

Dewpoint:
    - Blah

"""

class Profile(object):
    def __init__(self,z,fdir,zL=False,E=2000,m=4.0,hodotype='curved',
                    n=1,V=12.0,PW=30,thT=343,tT=213,th0=300,qv0=16E-3,Us=25,
                    fname='profile',H=14500,zT=14000,RH=0.9,zs=3000,
                    method='MW01',k2P=96500,k2T=296.65,k2Td=296.15,
                    k2z=250,offset_spd=0,offset_dir=0):
        """Create profile text file.

        Parameters
        ----------
        z   :   array-like
            Altitude above ground level (metres)
        zL  :   int, float
            LCL (metres AGL)
        E   :   int, float
            Specified CAPE
        m   :   int, float
            Profile compression parameter
        hodotype    :   {'straight','curved'}
            Set hodograph type
        n   :   int
            Curvature parameter
        V   :   int, float
            Maximum wind (m/s); overall vert. shear
        fdir    :   str
            Directory for output file
        PW  :   float,int,optional
            Precip. water (mm)
        fname   :   str, optional
            Filename for output
        H   :   int, float, optional
            Vertical scale parameter
        zT  :   int, float, optional
            Height of the tropopause (m). Must be in z
        RH  :   float, optional
            Relative humidity above LCL (as a decimal)
        method  :   {'MW01,}
            Paper from which to create profile
        k2P    :   int, float, optional
            Pressure at "model level 2", Pa
        k2T, k2Td   :   int, float, optional
            Drybumb, dewpoint at "mdel level 2", C
        offset_spd, offset_dir  :   int, float
            Speed and direction of background flow
       

        """
        # if zT not in z:
            # raise Exception("Height of the tropopause must be"
                                # "in array of altitudes.")

        self.fname = fname

        if method is 'WK82':
            self.z = z
            self.fdir = fdir
            self.th0 = th0
            self.thT = thT
            self.tT = tT 
            self.zT = zT
            self.qv0 = qv0
            self.Us = Us
            self.zs = zs
            self.offset_spd = offset_spd
            self.offset_dir = offset_dir
            self.profile_th, self.profile_rv = self.MWtemphum()
            self.profile_u, self.profile_v = self.MWwind()
        
        elif method is 'MW01':

            self.test_old_b = test_old_b

            # pdb.set_trace()
            if zL not in z:
                insidx = N.argmax(z>zL)
                z = N.insert(z,insidx,zL)

            if k2z not in z:
                insidx = N.argmax(z>k2z)
                z = N.insert(z,insidx,k2z)

            self.z = z
            self.zL = zL
            self.E = E
            self.m = m
            self.hodotype = hodotype
            self.n = n
            self.V = V
            self.PW = PW
            self.fdir = fdir
            self.fname = fname
            self.H = H
            self.zT = zT
            self.RH = RH
            self.k2P = k2P
            self.k2T = k2T
            self.k2Td = k2Td
            self.k2z = k2z
            self.profile_th, self.profile_rv = self.buoyancy()
            self.profile_u, self.profile_v = self.wind()

        else:
            raise Exception("Choose method.")

        self.create_textfile()
        # self.plot_skewT()

    def buoyancy(self,):
        """ Parcel buoyancy profile, or Eq. (A1) in MW01.

        Returns
        -------

        b   :   array_like
            Atmospheric profile of buoyancy.
        qv  :   array_like
            Atmospheric profile of mixing ratio
        """
        z = self.z
        E = self.E
        m = self.m
        zL = self.zL
        H = self.H
        zT = self.zT
        RH = self.RH

        # z' in Eq (1)
        dz = z-zL

        # Index of LCL, k2, and tropopause
        Lidx = N.where(z==zL)[0][0]
        k2idx = N.where(z==self.k2z)[0][0]
        Tidx = N.where(z==zT)[0][0]

        # Height above LCL of max buoyancy
        self.Zb = H/m

        # Buoyancy profile
        b = ((E*dz*m**2)/(H**2))*N.exp((m/H)*-dz)
        
        # Set buoyancy to zero at and above tropopause
        # troposphere = N.where(z>=zT)
        # bT = copy.copy(b)
        # bT[troposphere] = 0

        # Generating other arrays 
        P = N.zeros_like(z,dtype=N.dtype('f8'))
        Theta = N.zeros_like(P)
        Rv = N.zeros_like(P)
        DryBulb = N.zeros_like(P)

        # Stats at k2
        P[k2idx] = self.k2P        
        Rv[k2idx] = self.compute_rv(self.k2P,Td=self.k2Td)
        self.k2RH = self.compute_RH(self.k2Td,self.k2T)
        Theta[k2idx] = self.compute_theta_2(self.k2P,self.k2T)
        DryBulb[k2idx] = self.k2T

        # Temperature at the LCL
        TLCL = self.compute_TLCL(self.k2Td,self.k2T)
        # self.k2the = self.compute_thetae_4(DryBulb[k2idx],P[k2idx],
                            # Rv[k2idx],TLCL)
        self.k2the = self.compute_thetaep(P[k2idx],Rv[k2idx],self.k2T,self.k2RH)

        # self.k2the = self.compute_thetae_3(Theta[k2idx],self.k2P,self.k2Td, self.k2T,Rv[k2idx],TL=TLCL)
        # self.k2the = self.compute_thetae(self.k2T,self.k2rv,self.k2P,self.k2Td)
        # self.k2the = self.compute_thetae_2(self.k2th,self.k2rv,self.k2T)+4

        # Get to next z level
        # dZ = zdiff[znext]
        # dP = self.compute_P_at_z(self.k2P,self.k2T,dZ)
        # P[znext] = self.k2P + dP
        # DryBulb[znext] = self.compute_T_with_LCL(P[znext],self.k2P,self.k2T)
        # DryBulb[znext] = self.compute_T_invert_thetae(self.k2the,P[znext],self.k2rv,self.k2T)
        # Rv[znext] =  self.compute_rv(P[znext],T=DryBulb[znext],RH=self.RH)
        # Theta-e is conserved, and use T of LCL
        # Theta[znext] = self.compute_theta(self.k2the,Rv[znext],self.k2T)
        # Theta[znext] = self.compute_theta_2(P[znext],DryBulb[znext])



        # Now iterate up sounding from k2 thru LCL to top of model
        for zidx in range(k2idx+1,len(z)):
            zbot = zidx-1
            dZ = z[zidx]-z[zbot]
            dP = self.compute_P_at_z(P[zbot],DryBulb[zbot]-(6.5*dZ*10**-3),dZ)
            # dP = self.compute_P_at_z(P[zbot],DryBulb[zbot],dZ)
            P[zidx] = P[zbot] + dP

            if zidx < Lidx:
                Rv[zidx] = Rv[k2idx]
                # DryBulb[zidx] = self.compute_T_with_LCL(P[zidx],self.k2P,self.k2T)
                Theta[zidx] = self.compute_theta(self.k2the,Rv[k2idx],TLCL)
                DryBulb[zidx] = self.compute_drybulb(Theta[zidx],P[zidx])

            elif zidx == Lidx:
                # Using LCL RH = 0.9. Could try 1?
                DryBulb[zidx] = TLCL
                Theta[zidx] = self.compute_theta_2(P[zidx],TLCL)
                Rv[zidx] = self.compute_rv(P[zidx],T=DryBulb[zidx],RH=self.RH)
                TdLCL = self.compute_Td(DryBulb[zidx],self.RH)
                # theLCL = self.compute_thetae_4(DryBulb[zidx],P[zidx],
                            # Rv[zidx],TLCL)
                theLCL = self.compute_thetaep(P[zidx],Rv[zidx],DryBulb[zidx],self.RH)
                # theLCL = self.compute_thetae_3(Theta[zidx],P[zidx],TdLCL,TLCL,Rv[zidx])

            elif zidx > Lidx:
                T2 = self.compute_T_with_LCL(P[zidx],P[Lidx],TLCL)
                Th2 = self.compute_theta_2(P[zidx],T2)
                Rv2 = self.compute_rv(P[zidx],T=T2,RH=self.RH)
                # Th1 = self.compute_theta(theLCL,Rv[zbot],TLCL)
                # T1 = self.compute_drybulb(Th1,P[zidx])
                # T3 = self.compute_T_invert_thetae(theLCL,P[zidx],Rv[zbot],TLCL)
                # Th3 = self.compute_theta_2(P[zidx],T3)
                Theta[zidx] = Th2
                DryBulb[zidx] = T2
                Rv[zidx] = Rv2
                # pdb.set_trace()
        pdb.set_trace()

        # Now iterate down from LCL to surface
        for zidx in range(0,Lidx+1)[::-1]:
            ztop = zidx+1

            dZ = z[zidx]-z[ztop]
            dP = self.compute_P_at_z(P[ztop],DryBulb[ztop]+(6.5*dZ*10**-3),dZ)
            # pdb.set_trace()
            P[zidx] = P[ztop] + dP
            # DryBulb[zidx] = self.compute_T_invert_thetae(self.k2the,P[zidx],Rv[zbot],self.k2T)
            DryBulb[zidx] = self.compute_T_with_LCL(P[zidx],self.k2P,self.k2T)
            # Rv[zidx] = self.compute_rv(P[zidx],T=DryBulb[zidx],RH=self.RH)
            Rv[zidx] = Rv[Lidx]
            Theta[zidx] = self.compute_theta(self.k2the,Rv[zidx],self.k2T)

            # MIGHT NEED RICHARDSON CONSTRAINT
            
        # Create isothermal layer above tropopause
        isotherm = DryBulb[Tidx]
        
        # for zidx in range(Tidx+1,len(z)):
        # Theta = isothermC * (100000/P)**(mc.R/mc.cp)
        TRO = slice(Tidx+1,len(z))
        Theta[TRO] = self.compute_theta_2(P[TRO],isotherm)
        DryBulb[TRO] = self.compute_drybulb(Theta[TRO],P[TRO])
        # ttt = Theta/( (100000/P)**(mc.R/mc.cp))
        # Enforce 90% again
        # Rv[zidx] = self.compute_rv(p=P[zidx],T=ttt, RH=self.RH)
        Rv[TRO] = self.compute_rv(p=P[TRO],T=DryBulb[TRO], RH=self.RH)

        # Integrate CAPE between LCL and model top after isothermal calc
        Et = integrate.trapz(b[Lidx:],x=z[Lidx:])

        # Multiply buoyancy profile by CAPE ratio
        B = b * (E/Et)

        # Calc virtual temp perturbations from buoyancy
        # Tpert = (B*Theta)/mc.g
        Tpert = self.compute_thetapert(B,DryBulb,Rv)

        # Change theta, sfc/LCL to tropopause, to create CAPE==E in sounding
        THETA = Theta.copy()
        modify_bLCL = False
        if not modify_bLCL:
            # Only change LCL upward
            THETA[Lidx:Tidx] = Theta[Lidx:Tidx]-Tpert[Lidx:Tidx]
        else:
            # Change near surface too
            THETA[:Tidx] = Theta[:Tidx]-Tpert[:Tidx]

        # Enforce RH = 90% for new theta values
        DRYBULB = self.compute_drybulb(THETA,P)
        SL = slice(Lidx,Tidx)
        RV = self.compute_rv(P,T=DRYBULB,RH=self.RH)
        # RV[SL] = self.compute_rv(P[SL],T=DRYBULB[SL],RH=self.RH)

        # Change PW
        # Substract ~8C from theta and dewpoint
        # T in C; RH in 0-100 %
        # Td = utils.dewpoint(T,RH) 
        # Modified RH

        # Extra fields for plotting skew-T
        ES = self.compute_es(DRYBULB)
        RVs = 0.622*(ES/(P-ES))
        RHs = RV/RVs
        T_D = self.compute_Td(DRYBULB,RHs)
        sT.plot_profile(P/100,DRYBULB-273.15,T_D-273.15,self.fdir)
        # sT.plot_profile(pres,oldtemps,dewpoints,self.fdir,fname='skewT_oldtemps.png')

        pdb.set_trace() 
        return THETA, RV 

    def compute_std_P(self,z):
        """Compute pressure in the tropopause as a function
        of height, for a standard atmosphere.
        """

        P = 101290*((288.14-(0.00649*z))/288.08)**5.256
        return P
    
    def compute_drybulb(self,theta,P):
        T = theta/((100000/P)**(mc.R/mc.cp))
        return T

    def compute_RH(self,Td,T):
        RH = N.exp(0.073*(Td-T))
        return RH

    def P_to_z(self,T,Pa,refPa=100000):
        """Convert pressure to altitude.

        Parameters
        ----------
        T   :   int, float, array_like
            Dry-bulb temperature. If array, must be identical to Pa in size
        Pa  :   int, float, array_like
            Pressure (Pa). If array, must be identical to T in size
        refPa   :   int, float, optional
            Reference pressure (Pa).

        Returns
        -------
        z   :   float, array_like
            Height(s) above ground level (m), same size as Pa, T.
        """

        z = -1*((mc.R*T)/mc.g)*N.log(Pa/refPa)
        return z

    def z_to_P(self,T,z,refPa=100000):
        """Convert altitude to pressure.
        """

        Pa = refPa*N.exp((-1*mc.g*z)/(T*mc.R))
        return Pa

    def compute_es(self,T):
        """Approximation for saturation water pressure of water.
        """
        TC = T-273.15
        es = 6.1078*N.exp((19.8*TC)/(273+TC))    
        return es*100       # This is in Pa

    def compute_e(self,Td=None,es=None,RH=None):
        """RH is decimal.
        """
        if Td is not None:
            Td = Td-273.15
            e = 6.1*N.exp(0.073*Td)
        else:
            eshPa = es/100
            e = RH*eshPa
        e = e * 100 # into Pa
        return e

    def compute_thetae(self,T,r,P,Td=None,RH=None):
        # r is mixing ratio here
        TC = T-273.15
        PhPa = P/100
        pkap = 0.2854
        if not Td:
            Td = self.compute_Td(T,RH)
        if not r:
            es = self.compute_es(T)
            e = self.compute_e(Td)
            r = e/es

        thetae = TC*(1000/PhPa)**(pkap*(1-0.28*r)) * N.exp(
                        (3.376/mc.R-0.00254)*1000.0*r*(1+0.81*r))

        # Tstar = self.compute_Tstar(TK,e)
        # pdb.set_trace()
        # thetae = T*((100000/P)**(0.2854*(1-(0.28*r))))*N.exp(
                    # r*(1+(0.81*r))*((3376/Tstar)-2.54))
        # thetae = (T + (r * mc.Lv/mc.cp))*(100000/P)**mc.kappa
        # pdb.set_trace()
        return thetae + 273.15

    def compute_thetae_2(self,theta,r,TLCL):
        R = r #*1000
        # TLCL = TLCL-273.15
        thetae = theta * N.exp(
                ((3.376/TLCL)-0.00254)*
                (R*(1+(0.81*R*10**-3))))
        # pdb.set_trace()
        return thetae

    def compute_thetae_3(self,theta,P,Td,T,r,TL=False):
        P = P/100
        e = self.compute_e(Td=Td)/100
        # Td = Td-273.15

        if not TL:
            TL = self.compute_TLCL(Td,T)
        thL = T*((1000/(P-e))**0.2854)*((T/TL)**(0.28*r))
        thetae = thL * N.exp(((3036/TL)-1.78)*r*(1+(0.448*r)))
        pdb.set_trace()
        return thetae

    def compute_thetae_4(self,T,p,r,TL):
        p = p/100
        # r = r*1000

        thetae = T*((1000/p)**(0.2854*(1-(r*0.28E-3))))*N.exp(
                    (((3.376/TL)-0.00254)*r*(1+(r*0.81E-3))))
        return thetae

    def compute_thetaep(self,p,r,T,RH):
        p = p/100
        es = self.compute_es(T)
        e = self.compute_e(es=es,RH=RH)
        Tstar = self.compute_Tstar(T,e)
        thetaep = T*((1000/p)**(0.2854*(1-(0.28*r))))*N.exp(
                    r*(1+(r*0.81))*((3376/Tstar)-2.54))
        return thetaep

    def compute_TLCL(self,Td,T):
        TL = 56 + 1/((1/(Td-56)) + (N.log(T/Td)/800))
        return TL

    def compute_Tstar(self,TK,e):
        T = TK-273.15
        Tstar = (2840/(3.5*N.log(TK) - N.log(e) - 4.805)) + 55
        return Tstar

    def compute_theta(self,thetae,r,TLCL):
        # L = mc.Lv
        TLCL = TLCL-273.15
        # L = self.compute_Lv(TLCL)
        R = r * 1000
        # R = r
        # es = self.compute_es(T)
        # qs = self.compute_qs(es,P)
        # theta = thetae/(
                # 1+((L*qs)/(mc.cp*TLCL)))
                # (1+(L*qs))/(mc.cp*TLCL))

        theta = thetae/N.exp(
                ((3.376/TLCL)-0.00254)*
                (R*(1+(0.81*R*10**-3))))

                # 1.0723 * 10E-6 * (L/TLCL_C) *
                # r*(1+(0.81*r*10**-3)))
        # pdb.set_trace()
        return theta+273.15

    def compute_Lv(self,T):
        TC = T-273.15
        Lv = (2.501 - (0.00237*TC))*10**6
        return Lv
    
    def compute_qs(self,es,p):
        hPa = p/100
        eshPa = es/100
        qs = (eshPa*mc.R) / (hPa*mc.Rv)
        return qs

    def compute_thetae_profile(self,TK,p,r,RH):
        """Bolton's theta E profile.

        Parameters
        ----------
        TK  :   int, float
            Temp at initial level (K)
        p   :   int, float
            Pressure at initial level (Pa)
        r   :   int, float
            Mixing ratio at initial level (g/kg?)
        RH  :   int, float
            RH...where? (percent)
        """

        p = p/100
        TL = self.compute_TL(TK,RH)
        thetaE = (TK*(1000/p)**(0.2854*((1-0.28)*(10**-3)*r))*
                    N.exp(((3.376/TL)-0.00254)*
                    r*(1+0.81*(10**-3)*r)))
        return thetaE

    def compute_TL(self,TK,rh):
        RH = rh * 100
        TL = 55 + (1/((1/(TK-55))-(N.log(RH/100)/2840)))
        return TL

    def compute_w(self,T,P):
        # t = T-273.15
        e = self.compute_e(T)
        w = ((0.622*e)/(P-e))
        return w

    def wind(self):
        """Wind profiles.

        Parameters
        ----------

        Returns
        -------
        u, v    :   array-like
            Arrays of u and v profiles
        """
        n = self.n
        V = self.V
        z = self.z
        zL = self.zL
        Zv = self.Zb
        H = self.H

        dz = z-zL
        cn = (n**-1)*N.exp(1)
        vc = cn*V*((n**2)/(H**2))*dz*N.exp(-dz*(n/H))
        if hodoyupe is 'straight':
            v = V*N.arcsin(vc/V)
            u = dz*0
        elif hodotype is 'curved':
            u = N.sign((dz-Zv))*((V**2)-vc**2)**0.5
            v = vc
        else:
            raise Exception("Specify correct hodotype.")
        return u,v

    def compute_P_levels(self,zz,theta):
        Plist = []
        P0 = 100000
        K = mc.R/mc.cp
        th0 = theta[0]
        for th,z in zip(theta,zz):
            PL = N.exp(((K-1)**-1)*(N.log(((mc.g*P0**K)/mc.R)*((th**-1)-(th0**-1)))+
                                N.log(P0**(K-1))))
            Plist.append(PL)
        Plist[0] = P0
        pdb.set_trace()
        return N.array(Plist)

    def calc_theta(self,z,ztr=12000.0,theta0=300.0,thetatr=343,Ttr=213):
        if z <= ztr:
            theta = theta0 + (thetatr - theta0) * (z/ztr)**1.25
        else:
            theta = thetatr * N.exp((mc.g/(mc.cp*Ttr))*(z-ztr))
        return theta

    def calc_RH(self,z,ztr=12000.0):
        if z <= ztr:
            RH = 1-(0.75*(z/ztr)**1.25)
        else:
            RH = 0.25
        return RH

    def calc_U(self,z,us,zs=3000):
        U = us * (N.tanh(z)/zs)
        return U

    def create_textfile(self):
        # fpath = os.path.join(self.fdir,self.fname)
        pass

    def plot_skewT(self):
        # fname = self.fname + '_skewT.png'
        # fpath = os.path.join(self.fdir,fname)
        
        pass
    
    def compute_Td(self,T,RH):
        TC = T-273.15
        Td = (N.log(RH)/0.073) + TC
        return Td+273.15

    def compute_rv(self,p,Td=None,RH=None,T=None):
        PhPa = p/100
        if Td is not None:
            e = self.compute_e(Td=Td)
        else:
            es = self.compute_es(T)
            e = self.compute_e(es=es,RH=RH)
        ehPa = e/100
        # rv = (mc.Rd/mc.Rv) * (ehPa/(PhPa-ehPa))
        rv = (0.622*e)/(p-e)
        # pdb.set_trace()
        return rv

    def RH_invert_RV(self,P,T,RV):
        es = self.compute_es(T)
        
        P = P/100
        es = es/100

        # RH = (P*(RV+0.622))/(0.622*es)
        RH = 1/( ((0.622*es)/P) * ( (1/RV)+(1/0.622)))
        return RH

    def compute_P_at_z(self,Pbot,T,r,dz):
        # dP = (-dz*P*mc.g)/(mc.R*T)
        Tv = self.compute_Tv(T,r)
        H = (mc.Rd * Tv)/mc.g
        Ptop = Pbot * N.exp(-1*(dz/H))
        # pdb.set_trace()
        return Ptop 
        # return dP

    def compute_T_with_LCL(self,P,PLCL,TLCL):
        TLCL = TLCL - 273.15
        T = TLCL * (PLCL/P)**(-mc.R/mc.cp)
        return T + 273.15

    def compute_T_invert_thetae(self,thetae,P,r,TLCL):
        R = r #* 1000
        p = P/100
        power = 0.2854*(1-(0.28*R*10**-3))
        T = thetae/(
                ((1000/p)**power)*
                N.exp(((3.376/TLCL)-0.00254)*
                R*(1+(R*0.81*10**-3))))
        # pdb.set_trace()
        return T

    def compute_theta_2(self,P,T):
        TC = T - 273.15
        # TC = T
        theta = TC * ((100000/P)**(mc.R/mc.cp))
        return theta+273.15

    def compute_Tv(self,T,r):
        TC = T-273.15
        R = r
        Tv = TC * ((1+(R/0.622))/(1+R))
        return Tv+273.15

    def compute_thetapert(self,B,T,r):
        Tv_env = self.compute_Tv(T,r)
        Tv_pert = (B*Tv_env)/mc.g
        return Tv_pert

    def MWtemphum(self):

        Theta = N.zeros_like(self.z,dtype=N.dtype('f8'))
        RH = N.zeros_like(Theta)
        P = N.zeros_like(Theta)
        # P = self.get_stdP(self.z)
        RV = N.zeros_like(Theta)
        T = N.zeros_like(Theta)

        # P[0] = 101325
        P[0] = 100000
        # P[0] = 99500
        T[0] = self.th0

        for idx,z in enumerate(self.z):
            if z <= self.zT:
                Theta[idx] = self.th0 + ((self.thT - self.th0)*(
                                            (z/self.zT)**1.25))
                RH[idx] = 1 - (0.75*((z/self.zT)**1.25))
            else:
                Theta[idx] = self.thT * N.exp(
                            (mc.g/(mc.cp*self.tT))*(z-self.zT))
                RH[idx] = 0.25

        for idx,z in enumerate(self.z):
            if idx > 0:
                dZ = self.z[idx] - self.z[idx-1]
                # dZZ = self.z[idx]

                # dP = self.compute_P_at_z(P[idx-1],DryBulb[idx-1]-(6.5*dZ*10**-3),dZ)
                # dP = self.compute_P_at_z(P[idx-1],T[idx-1],dZ)
                # dP = self.compute_P_at_z(P[idx-1],T[idx-1]-(7*dZ*10**-3),dZ)
                mT = T[idx-1] - (7*0.5*dZ*10**-3)
                # mT = T[idx-1]
                P[idx] = self.compute_P_at_z(P[idx-1],mT,RH[idx],dZ)
                # P[idx] = P[idx-1] + (1.0*dP)
                T[idx] = self.compute_drybulb(Theta[idx],P[idx])
                pass

        # pdb.set_trace()
        T = self.compute_drybulb(Theta,P)
        T_D = self.compute_Td(T,RH)
        # RV = self.compute_rv(P,T,RH)
        RV = self.compute_rv(P,Td=T_D)
        RV = RV*0.85
        # Set constant mixing ratio at surface
        # PBLidx = N.argmax(P<85000)
        PBLidx = N.argmax(RV<self.qv0)#-2
        # RV_const = RV[PBLidx+1]
        RV_const = self.qv0
        RV[0:PBLidx] = RV_const

        RH2 = self.RH_invert_RV(P,T,RV)

        # Fix constant theta-e in lowest layer
        # hLCL = ( 20 + ((T[0]-273.15)/5)) * (100-(100*RH2[0]))
        # pdb.set_trace()
        # LCLidx = N.argmax(self.z>hLCL)#-2
        # thE = self.compute_thetaep(P[LCLidx],RV[LCLidx],T[LCLidx],RH[LCLidx])
        # Theta[0:LCLidx] = self.compute_theta(thE,RV[0:LCLidx],T[LCLidx])
        # pdb.set_trace()

        test_plot = False
        if test_plot:
            # Recalculate RH in PBL
            # ES = self.compute_es(T)
            # RVs = 0.622*(ES/(P-ES))
            # RH2 = RV/RVs
            # T_D = self.compute_Td(T,RH2)
            # T = self.compute_drybulb(Theta,P)
            T_D = self.compute_Td(T,RH2)
            sT.plot_profile(P/100,T-273.15,T_D-273.15,'/home/johnlawson')
        # pdb.set_trace()
        return Theta, RV

    def MWwind(self):
        # U = N.zeros_like(self.z,dtype=N.dtype('f8'))
        U = self.Us * (N.tanh(self.z/self.zs))
        u,v = utils.decompose_wind(U-self.offset_spd,self.offset_dir)
        return u,v

    def create_textfile(self,):
        fpath = os.path.join(self.fdir,self.fname)
        U = self.profile_u
        V = self.profile_v
        TH = self.profile_th
        RV = self.profile_rv
        Z = self.z
        blanks = N.zeros_like(U)
        blanks[:] = 0
        space = '    '

        data = N.zeros([len(Z),5])
        # pdb.set_trace()
        data[:,0] = Z
        data[0,0] = 1000.0
        data[:,1] = TH
        data[:,2] = RV*1000.0
        data[:,3] = U
        # data[0,4] = ''
        data[:,4] = V
        # data[0,5] = ''
        fmts = ('%8.4f','%8.4f','%8.4f','%8.4f','%8.4f')

        # for u,v,th,rv,z in zip(U,V,TH,RV,Z):
            # lgen = '{0}{1:.4f}{0}{2:.4f}{0}{3:.4f}{0}{4:.4f}{0}{5:.4f}'.format(
                        # space,z,th,r,u,v)
        N.savetxt(fpath,data,fmt='%8.4f',delimiter=space,)
        print("Saved profile to {0}.".format(fpath))
        return

    def get_stdT(self,z):
        stdT = N.zeros_like(z,dtype=N.dtype('f8'))
        stdT[0] = 288.15
        for idx in range(1,len(stdT)):
            dZ = z[idx] - z[idx-1]
            stdT[idx] = stdT[idx-1] - (dZ * 0.0065)
        return stdT

    def get_stdP(self,z):
        stdT = self.get_stdT(z)
        stdP = N.zeros_like(z,dtype=N.dtype('f8'))
        stdP[0] = 101325
        for idx in range(1,len(stdP)):
            dZ = z[idx] 
            mT = N.mean(stdT[:idx])
            mP = N.mean(stdT[:idx])
            stdP[idx] = stdP[0] - ((dZ*mP*mc.g)/(mc.R*mT))
        pdb.set_trace()
        return stdP

    def PPP(self,Z):
        P = 1013.25 * ((1-(Z/(0.3048*145366.45)))**(1/0.19))
        return P*100
