import numpy as np
from numpy import sin, cos, tan, arctan, sqrt, pi
import matplotlib.pyplot as plt
from QKP2 import rE,rtE, phitE, x_x1, get_k, get_l, get_beta, get_w
from constants import *
from QKP2 import get_u_
import antenna_pattern as ap
from scipy.integrate import cumtrapz
from scipy.interpolate import CubicSpline
from rr_enl import solve_rr

tarr=np.linspace(0,10*yr,1000)
period = 1.5*yr;inc = 0;M=1e9;eta=1/4;
n0 = 2*np.pi/period








def waveform(tarr,n0,M,e0,eta,inc,dist,order,g0=0,tref=0):
    x10 = (tsun * M * n0)**(2./3)
    x0=x_x1(x10,e0,eta,order)
    dis=M*dsun
    sc=dist*1e9*pc/dis
    k0=get_k(x0,e0,eta)
    z0=[n0,e0,g0,n0*(tarr[0]-tref)]
    narr,earr,garr,larr=solve_rr(M,eta,z0,tarr[0],tarr[-1],tarr)
    
    uarr=np.array([get_u_(l,e0) for l in larr])
    
    phiarr=larr+garr+get_w(x0,e0,eta,uarr)
    
    #phiarr=(1+k0)*larr+get_w(x0,e0,eta,uarr)
    x1arr=(M*narr*tsun)**(2/3)
    xarr=x_x1(x1arr,earr,eta,order)
    
    hp_arr=np.zeros(len(tarr))
    hx_arr=np.zeros(len(tarr))

    for i in range(len(tarr)):

        u=uarr[i]
        xx=xarr[i]
        ee=earr[i]
        phi=phiarr[i]
        r1=rE(xx,ee,eta,u)
        z=1/r1
        rphit=r1*phitE(xx,ee,eta,u)
        rt=rtE(xx,ee,eta,u)


        hp_arr[i]=(-eta*(sin(inc)**2*(z-rphit**2-rt**2)+(1+cos(inc)**2)*((z
                    +rphit**2-rt**2)*cos(2*phi)+2*rt*rphit*sin(2*phi))))
        hx_arr[i]=(-2*eta*cos(inc)*((z+rphit**2-rt**2)*sin(2*phi)-2*rt*rphit*cos(2*phi)))
    
    return hp_arr/sc, hx_arr/sc

def add_ecc_cgw(toas,
    theta,
    phi,
    cos_gwtheta,
    gwphi,
    psi,
    cos_inc,
    log10_n,
    q,
    e0,
    log10_M,
    tref,
    pdist,
    distGW,
    res='Both',
    interp_steps=1000
):
    order = 3
    n0 = 10**log10_n # mean motion
    M = 10**log10_M
    
    eta=q/(1+q)**2
    
    ts = toas - tref
    
    # ti, tf, tzs in seconds, in source frame
    ti = min(ts)
    tf = max(ts)
    Tspan=tf-ti
    
    tz_arr = np.linspace(ti, tf, interp_steps)
    delta_t_arr = (tz_arr[1]-tz_arr[0])
    
    inc = np.arccos(cos_inc)

    gwra = gwphi
    gwdec = np.arcsin(cos_gwtheta)

    psrra = phi
    psrdec = np.pi/2 - theta


    cosmu, Fp, Fx = ap.antenna_pattern(gwra, gwdec, psrra, psrdec)
    
    tP_arr=tz_arr-pdist/c*(1-cosmu)
    
    hpE,hxE=waveform(tz_arr,n0,M,e0,eta,inc,distGW,order)
    
    
    if res=='Both':
        hpP,hxP=waveform(tP_arr,n0,M,e0,eta,inc,distGW,order)
        sp=hpP-hpE; sx=hxP-hxE
    elif res=='Earth':
        sp=-hpE;sx=-hxE
    elif res=='Psr':
        hpP,hxP=waveform(tP_arr,n0,M,e0,eta,inc,distGW,order)
        sp=hpP; sx=hxP
        
        
    
#     hpP=0;hxP=0;
    
    
    c2psi = np.cos(2*psi)
    s2psi = np.sin(2*psi)
    Rpsi = np.array([[c2psi, -s2psi],
                     [s2psi, c2psi]])
    h_arr = np.dot([Fp,Fx], np.dot(Rpsi, [sp,sx]))

    # Integrate over time in SSB frame
    s_arr = cumtrapz(h_arr, initial=0)*delta_t_arr
    
    s_spline = CubicSpline(tz_arr, s_arr)
    
    s = s_spline(ts)

    return s
    