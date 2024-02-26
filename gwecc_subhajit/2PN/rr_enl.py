import numpy as np
from numpy import cosh, sqrt, pi
from QKP2 import  x_x1, get_k, u_from_l_3PN
from scipy.integrate import solve_ivp

from constants import *

def odes(M,eta,z,t):
    n,e,g,l=z
    Mc=M*eta**(3/5)
    x1 = (tsun * M * n)**(2./3)
    x=x_x1(x1,e,eta,order=3)
    OTS=np.sqrt(1-e*e)
    dndt=1/5*(Mc*tsun*n)**(5/3)*n**2*((96+292*e**2+37*e**4)/OTS**7)
    dedt=-1/15*(Mc*tsun*n)**(5/3)*n*e*((304+121*e**2)/OTS**5)
    k=get_k(x,e,eta)
    dgdt=k*n
    dldt=n
    
    return [dndt,dedt,dgdt,dldt]

def solve_rr(M,eta,z0,Ti,Tf,Tarr):
    sol = solve_ivp(lambda t,z:odes(M,eta,z,t),[Ti,Tf],z0,t_eval=Tarr,rtol=1e-10, atol=1e-10)
    
    return sol.y
    
    
