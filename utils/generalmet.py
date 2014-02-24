import numpy as N
import pdb

def decompose_wind(wspd,wdir):
    # Split wind speed/wind direction into u,v
    if (type(wspd) == N.array) & (type(wdir) == N.array):
        uwind = N.array([-s * N.sin(N.radians(d)) if ((s>-1)&(d>-1)) else -9999
                    for s,d in zip(wspd,wdir)])
        vwind = N.array([-s * N.cos(N.radians(d)) if ((s>-1)&(d>-1)) else -9999
                    for s,d in zip(wspd,wdir)])
    else:
        uwind = -wspd * N.sin(N.radians(wdir))
        vwind = -wspd * N.cos(N.radians(wdir)) 
    return uwind, vwind

def combine_wind_components(u,v):
    wdir = N.degrees(N.arctan2(u,v)) + 180
    wspd = N.sqrt(u**2 + v**2)
    return wspd, wdir

def convert_kt2ms(wspd):
    wspd_ms = wspd*0.51444
    return wspd_ms
