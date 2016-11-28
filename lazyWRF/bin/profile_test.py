import os
import numpy as N
import itertools
import pdb
import math

from WEM.lazyWRF.lazyWRF.create_sounding import Profile

z = N.hstack((N.array([0,10,35]),N.arange(100,18000,100)))
zL = 454
# zL = 500
k2z = 250
E = 2000
m = 7.0
hodotype = 'curved'
n = 2
V = 12
# fdir = os.path.abspath('./')
fdir = '/home/johnlawson/idealised/WK82_profiles_3'
method='WK82'
# qv0 = 16E-3
# True uses non-boosted buoyancy profile
# testsw=0
# Us = 25
offset_spd = 5
offset_dir = 270
zT = 12000

qv0_range = N.arange(10.0,16.1,0.1)#*10**-3
U_range = N.arange(0.0,51.0,1.0)
# U_range = [0,]

def gen_fname(q,u):
    qst = q*10
    ust = u*10
    fname = 'profile_{0:03.0f}_{1:03.0f}'.format(qst,ust)
    return fname
    # print(fname)
# P1 = Profile(fdir,z,zL,E,m,hodotype,n,V,test_old_b=testsw, k2z=k2z)

# qv0_range = [qv0,]
# U_range = [Us,]

for q,u in itertools.product(qv0_range,U_range):
    fname = gen_fname(q,u)
    P2 = Profile(z,fdir,method='WK82',qv0=q*10**-3,Us=u,fname=fname,
            zT=zT,offset_spd=offset_spd,offset_dir=offset_dir)
    # pdb.set_trace()

