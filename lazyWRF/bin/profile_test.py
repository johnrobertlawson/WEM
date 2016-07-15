import os
import numpy as N
import itertools
import pdb

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
fdir = '/home/johnlawson/idealised/WK82_profiles'
method='WK82'
qv0 = 16E-3
# True uses non-boosted buoyancy profile
testsw=0
Us = 30


qv0_range = N.arange(11.0,16.1,0.1)*10**-3
U_range = N.arange(0.0,46,1)

def gen_fname(q,u):
    qst = int(q*10**4)
    ust = int(u*10)
    fname = 'profile_{0:d}_{1:03d}'.format(qst,ust)
    return fname
    # print(fname)
# P1 = Profile(fdir,z,zL,E,m,hodotype,n,V,test_old_b=testsw, k2z=k2z)

for q,u in itertools.product(qv0_range,U_range):
    fname = gen_fname(q,u)
    P2 = Profile(z,fdir,method='WK82',qv0=q,Us=u,fname=fname)
    pdb.set_trace()

