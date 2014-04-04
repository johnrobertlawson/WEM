import numpy as N
import pdb

def decompose_wind(wspd,wdir,convert=0):
    # Split wind speed/wind direction into u,v
    if (type(wspd) == N.array) & (type(wdir) == N.array):
        uwind = N.array([-s * N.sin(N.radians(d)) if ((s>-1)&(d>-1)) else -9999
                    for s,d in zip(wspd,wdir)])
        vwind = N.array([-s * N.cos(N.radians(d)) if ((s>-1)&(d>-1)) else -9999
                    for s,d in zip(wspd,wdir)])
    else:
        uwind = -wspd * N.sin(N.radians(wdir))
        vwind = -wspd * N.cos(N.radians(wdir)) 
    if convert == 'ms_kt':
        uwind *= 1.94384449
        vwind *= 1.94384449
    elif convert == 'ms_mph':
        uwind *= 2.23694
        vwind *= 2.23694
    elif convert == 'kt_ms':
        uwind *= 0.51444444
        vwind *= 0.51444444
    else:
        pass
    return uwind, vwind

def combine_wind_components(u,v):
    wdir = N.degrees(N.arctan2(u,v)) + 180
    wspd = N.sqrt(u**2 + v**2)
    return wspd, wdir

def convert_kt2ms(wspd):
    wspd_ms = wspd*0.51444
    return wspd_ms

def dewpoint(T,RH): # Lawrence 2005 BAMS?
    #T in C
    #RH in 0-100 format
    es = 6.11 * (10**((7.5*T)/(237.7+T)))
    e = es * RH/100.0
    alog = 0.43429*N.log(e) - 0.43429*N.log(6.11)
    Td = (237.7 * alog)/(7.5-alog) 
    #pdb.set_trace()
    return Td
