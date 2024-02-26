import sys, os
sys.path.append("/data/home/prabu/InPTA/subhajit/packages/gwecc_subhajit/2PN/")

import numpy as np
from numpy import sin, cos, tan, arctan, sqrt, pi
import matplotlib.pyplot as plt
from QKP2 import rE,rtE, phitE, x_x1, get_k, get_l, get_beta, get_w
from constants import *
from scipy.optimize import minimize, fsolve




order=2




def get_u_(l,et):
    
    res = fsolve(lambda u: u-et*sin(u)-l, 0)[0]
    return res

def u_from_l_3PN(l,e,x,eta,order):    
    for i in range(4):
        u0 = get_u_(l,e)
        dl0=delta_L(x,e,eta,u0,order)
        l1 = l-dl0
        l=l1
    return u0




def fun(t,period,log10_M,eta,e0,phi0,t0):
    if type(t) is np.ndarray:
        return np.array([fun(t1,period,log10_M,eta,e0,phi0,t0) for t1 in t])
    t=t*yr;t0=t0*yr;period=period*yr;
    M0=10**log10_M
    n0 = 2*np.pi/period
    x10 = (tsun * M0 * n0)**(2./3)
    x0=x_x1(x10,e0,eta,order)
    l_=n0*(t-t0)
    u_=get_u_(l_,e0)
    W_=get_w(x0,e0,eta,u_)
    k0=get_k(x0,e0,eta)
    r1=rE(x0,e0,1/4,u_)
    phi_=phi0+(1+get_k(x0,e0,eta))*l_+get_w(x0,e0,eta,u_)
    return phi_/pi




days=24*3600;y_to_d=365.25




import scipy.optimize as optimization




nnarr=np.linspace(1,10,10)
fl_epoch=np.array([2445310,2445667,2447771,2448288,2450246,2451184,2454005,2454705,2456329,2457019])
#fl_err = 7+7 * np.random.rand(len(nnarr))
#np.savetxt("err_fl_err.txt",fl_err)
fl_err= np.loadtxt("err_fl_err.txt",dtype="float")







def dnn(dt,period,log10_M,eta,e0):
    dt=dt*days
    period=period*yr;
    n0 = 2*np.pi/period
    M=10**log10_M
    x10 = (tsun * M * n0)**(2./3)
    x0=x_x1(x10,e0,eta,order)
    omg=x0**(3/2)/(M*tsun)
    return omg*dt/pi




nnarr=np.linspace(1,10,10)




d_narr=dnn(fl_err,8.13,8,1/4,0.6)




fl_epoch_n=fl_epoch+fl_err




# plt.errorbar(nnarr,fl_epoch_n/y_to_d, yerr=np.abs(fl_err/y_to_d), fmt=".k", capsize=0)
# plt.xlabel("N")
# plt.ylabel("Flare Epoch [in years]")




fl_epoch_data=fl_epoch/y_to_d
fl_epoch_data
ti=fl_epoch_data[0];tf=fl_epoch_data[-1]




from nautilus import Prior

prior = Prior()

prior.add_parameter('P', dist=(0,20))
prior.add_parameter('log10_M', dist=(6,11))
prior.add_parameter('eta', dist=(0.01,0.25))
prior.add_parameter('e', dist=(0.01,0.9))
prior.add_parameter('phi0', dist=(0,pi))
prior.add_parameter('t0', dist=(ti,tf))



def log_likelihood(param_dict):
    P,log10_M, eta, e,phi0,t0= [param_dict[key] for key in prior.keys]
    model = fun(fl_epoch_data,P,log10_M,eta,e,phi0,t0)
    
    yerr=dnn(fl_err,P,log10_M,eta,e)
    #sigma2 = yerr**2 + model**2 * np.exp(2 * log_f)
    sigma2 = yerr**2 
    return -0.5 * np.sum((nnarr - model) ** 2 / sigma2 + np.log(sigma2))




import corner
import numpy as np
from nautilus import Prior, Sampler


ncpus=int(os.cpu_count()/4)
nlive=2000
sampler = Sampler(prior, log_likelihood,n_live=nlive,pool=ncpus,filepath='./datadir/data.h5')

sampler.run(verbose=True)
points, log_w, log_l = sampler.posterior()


import pickle
with open("posterior.pkl", "wb") as f:
    pickle.dump(sampler.posterior(), f)

fig=corner.corner(points,weights=np.exp(log_w),show_titles=True,labels=prior.keys,
                  bins=50,plot_datapoints=False,smooth=2,range=[0.99999] *points.shape[1],color="#1f77b4",truth_color="maroon")
fig.legend(["log_z ="+str("{:.2f}".format(sampler.log_z))])
plt.savefig('post_2PNn_all.png',dpi=80,bbox_inches='tight')





