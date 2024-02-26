import numpy as np
from numpy import sin, cos, tan, arctan, sqrt, pi as pi

def pn(term,order):
    sum1=0
    for i in range(order+1):
        sum1+=term[i]
    return sum1
    


def rE(x,et,eta,u):
	w=1-et*cos(u)
	OTS=sqrt(1-et*et)

	w1=(w*(1+(1/6)*x*(((7*eta-6)*w-9*eta+24)*et**2+(18-7*eta)*w+9*eta-24)/(OTS**2*w)+(1/72)*x**2*(((35*eta**2-231*eta+72)*et**4+(150*eta-70*eta**2+468)*et**2+648-567*eta+35*eta**2)*w+(261*eta-27*eta**2)*et**4+(288-1026*eta+54*eta**2)*et**2
	-288-27*eta**2+765*eta-36*(2*eta-5)*(w-3)*OTS**3)/(w*OTS**4))/x)

	return w1


# def rtE1(x,et,eta,u):
	
# 	w11=((et*sqrt(x)*(1 + (x*(-7*eta + et**2*(-6 + 7*eta)))/(6*(1 - et**2)) + (x**3*(-22680*(1 - et**2)**5*eta*(23 - 73*eta + 13*eta**2) + 22680*(1 - et**2)**4*eta*(-635 + 53*eta + 23*eta**2)*(1 - et*cos(u)) 
# 	+ (1 - et**2)**3*(-14515200 + (32397408 + 232470*pi**2)*eta - 4490640*eta**2 - 196560*eta**3 + et**2*(1088640*eta - 1315440*eta**2 + 151200*eta**3))*(1 - et*cos(u))**2 + (1 - et**2)**2*(9072000 + (6264432 - 464940*pi**2)*eta 
# 	- 1451520*eta**2 - 30240*eta**3 + et**2*(-2721600 + 7892640*eta - 2494800*eta**2 + 30240*eta**3))*(1 - et*cos(u))**3 + (1 - et**2)**(3/2)*(5443200 + (-23738400 + 232470*pi**2)*eta + 3084480*eta**2 + et**2*(2721600 - 997920*eta 
# 	+ 725760*eta**2))*(1 - et*cos(u))**4 + (-4717440 + (23806440 - 464940*pi**2)*eta - 3591000*eta**2 - 13720*eta**3 + et**4*(1179360 - 6191640*eta + 4190760*eta**2 - 41160*eta**3) + et**6*(-241920 + 1973160*eta - 1101240*eta**2 
# 	+ 13720*eta**3) + et**2*(-11249280 + (16034760 - 116235*pi**2)*eta - 6166440*eta**2 + 41160*eta**3) + sqrt(1 - et**2)*(1814400 + (-10029600 + 77490*pi**2)*eta + 1874880*eta**2 + et**2*(-8164800 + (14716800 - 77490*pi**2)*eta 
# 	- 2479680*eta**2) + et**4*(907200 - 2509920*eta + 604800*eta**2)))*(1 - et*cos(u))**5))/(362880*(1 - et**2)**3*(1 - et*cos(u))**5) + (x**2*(-135*eta + 9*eta**2 + et**2*(405*eta - 27*eta**2) + et**6*(135*eta - 9*eta**2) + et**4*(-405*eta 
# 	+ 27*eta**2) + (-540 + 351*eta - 9*eta**2 + et**4*(-540 + 351*eta - 9*eta**2) + et**2*(1080 - 702*eta + 18*eta**2))*(1 - et*cos(u)) + (-324 + 189*eta + 35*eta**2 + et**2*(-234 + 366*eta - 70*eta**2) + et**4*(72 - 231*eta + 35*eta**2))*(1 
# 	- et*cos(u))**3 - 36*(1 - et**2)**(3/2)*(-5 + 2*eta)*(1 - et*cos(u))**2*(4 - et*cos(u))))/(72*(1 - et**2)**2*(1 - et*cos(u))**3))*sin(u))/(1 - et*cos(u)))
	
# 	return w11

# def phitE1(x,et,eta,u):

	
# 	w11=((sqrt(1 - et**2)*x**(3/2)*(1 + (x*(-4 + eta)*(et**2 - et*cos(u)))/((1 - et**2)*(1 - et*cos(u))) + (x**2*(-6*(1 - et**2)**3*eta*(3 + 2*eta) + (1 - et**2)**2*(108 + 63*eta + 33*eta**2)*(1 - et*cos(u)) 
# 		+ (1 - et**2)*(-240 + sqrt(1 - et**2)*(180 - 72*eta) - 31*eta - 29*eta**2 + et**2*(48 - 17*eta + 17*eta**2))*(1 - et*cos(u))**2 + (42 + 22*eta + 8*eta**2 + sqrt(1 - et**2)*(-90 + 36*eta) + et**2*(-147 + 8*eta - 14*eta**2))*(1 
# 		- et*cos(u))**3))/(12*(1 - et**2)**2*(1 - et*cos(u))**3) + (x**3*(5040*(1 - et**2)**5*eta*(-3 + 8*eta + 2*eta**2) + 1120*(1 - et**2)**4*eta*(-349 - 186*eta + 6*eta**2)*(1 - et*cos(u)) + 4*(1 - et**2)**3*eta*(539788 
# 		- 4305*pi**2 + 20160*eta - 19600*eta**2 + et**2*(4200 - 5040*eta + 1120*eta**2))*(1 - et*cos(u))**2 + 140*(1 - et**2)**2*(-4032 - 15688*eta + 1020*eta**2 + 724*eta**3 + sqrt(1 - et**2)*(8640 - 5616*eta + 864*eta**2) 
# 		+ et**2*(1728 + 3304*eta - 612*eta**2 - 460*eta**3))*(1 - et*cos(u))**3 + 4*(1 - et**2)*(127680 + (19372 + 12915*pi**2)*eta - 32900*eta**2 - 11060*eta**3 + sqrt(1 - et**2)*(-235200 + (-162400 + 4305*pi**2)*eta) 
# 		+ et**4*(4620*eta + 3220*eta**2 - 4060*eta**3) + et**2*(-252000 + (-300528 + 4305*pi**2)*eta + 98560*eta**2 + 16800*eta**3 + sqrt(1 - et**2)*(134400 - 119280*eta + 40320*eta**2)))*(1 - et*cos(u))**4 + (-147840 
# 		+ (1127280 - 43050*pi**2)*eta + 8960*eta**2 + 4480*eta**3 + sqrt(1 - et**2)*(-67200 + (674240 - 8610*pi**2)*eta - 53760*eta**2) + et**4*(-221760 - 113680*eta + 94640*eta**2 + 13440*eta**3) + et**2*(-194880 + (692928 
# 		+ 12915*pi**2)*eta - 112000*eta**2 - 11200*eta**3 + sqrt(1 - et**2)*(-739200 + 544320*eta - 127680*eta**2)))*(1 - et*cos(u))**5))/(13440*(1 - et**2)**3*(1 - et*cos(u))**5)))/(1 - et*cos(u))**2)
  
# 	return w11



def rtE(x,et,eta,u):

	OTS=sqrt(1-et*et)
	w=1-et*cos(u)

	w11=((et*sqrt(x)*(1 + ((-7*eta + et**2*(-6 + 7*eta))*x)/(6*OTS**2) + ((-135*eta + 9*eta**2 + et**2*(405*eta - 27*eta**2) + et**6*(135*eta - 9*eta**2) + et**4*(-405*eta + 27*eta**2) 
		+ (-540 + 351*eta - 9*eta**2 + et**4*(-540 + 351*eta - 9*eta**2) + et**2*(1080 - 702*eta + 18*eta**2))*w + (-324 + 189*eta + 35*eta**2 + et**2*(-234 + 366*eta - 70*eta**2) + et**4*(72 - 231*eta + 35*eta**2))*w**3 
		- 36*(1 - et**2)*(-5 + 2*eta)*OTS*w**2*(3 + w))*x**2)/(72*OTS**4*w**3) + ((-22680*eta*(23 - 73*eta + 13*eta**2)*OTS**10 + 22680*eta*(-635 + 53*eta + 23*eta**2)*OTS**8*w + OTS**6*(-14515200 - 4490640*eta**2 
		- 196560*eta**3 + et**2*(1088640*eta - 1315440*eta**2 + 151200*eta**3) + eta*(32397408 + 232470*pi**2))*w**2 + OTS**4*(9072000 - 1451520*eta**2 - 30240*eta**3 + et**2*(-2721600 + 7892640*eta - 2494800*eta**2 
		+ 30240*eta**3) + eta*(6264432 - 464940*pi**2))*w**3 + OTS**3*(5443200 + 3084480*eta**2 + et**2*(2721600 - 997920*eta + 725760*eta**2) + eta*(-23738400 + 232470*pi**2))*w**4 + (-4717440 - 3591000*eta**2 
		- 13720*eta**3 + et**4*(1179360 - 6191640*eta + 4190760*eta**2 - 41160*eta**3) + et**6*(-241920 + 1973160*eta - 1101240*eta**2 + 13720*eta**3) + eta*(23806440 - 464940*pi**2) + et**2*(-11249280 - 6166440*eta**2 
		+ 41160*eta**3 + eta*(16034760 - 116235*pi**2)) + OTS*(1814400 + 1874880*eta**2 + et**4*(907200 - 2509920*eta + 604800*eta**2) + eta*(-10029600 + 77490*pi**2) + et**2*(-8164800 - 2479680*eta**2 + eta*(14716800 
		- 77490*pi**2))))*w**5)*x**3)/(362880*OTS**6*w**5))*sin(u))/w)
	return w11


def phitE(x,et,eta,u):

	OTS=sqrt(1-et*et)
	w=1-et*cos(u)

	w22=((OTS*x**(3/2)*(1 + ((-4 + eta)*(-1 + et**2 + w)*x)/(OTS**2*w) + ((-6*eta*(3 + 2*eta)*OTS**6 + (108 + 63*eta + 33*eta**2)*OTS**4*w + OTS**2*(-240 - 31*eta - 29*eta**2 + et**2*(48 - 17*eta + 17*eta**2) 
		+ (180 - 72*eta)*OTS)*w**2 + (42 + 22*eta + 8*eta**2 + et**2*(-147 + 8*eta - 14*eta**2) + (-90 + 36*eta)*OTS)*w**3)*x**2)/(12*OTS**4*w**3) + ((5040*eta*(-3 + 8*eta + 2*eta**2)*OTS**10 + 1120*eta*(-349 
		- 186*eta + 6*eta**2)*OTS**8*w + 4*eta*OTS**6*(539788 + 20160*eta - 19600*eta**2 + et**2*(4200 - 5040*eta + 1120*eta**2) - 4305*pi**2)*w**2 + 140*OTS**4*(-4032 - 15688*eta + 1020*eta**2 + 724*eta**3 
		+ et**2*(1728 + 3304*eta - 612*eta**2 - 460*eta**3) + (8640 - 5616*eta + 864*eta**2)*OTS)*w**3 + 4*OTS**2*(127680 - 32900*eta**2 - 11060*eta**3 + et**4*(4620*eta + 3220*eta**2 - 4060*eta**3) + eta*(19372 
		+ 12915*pi**2) + et**2*(-252000 + 98560*eta**2 + 16800*eta**3 + (134400 - 119280*eta + 40320*eta**2)*OTS + eta*(-300528 + 4305*pi**2)) + OTS*(-235200 + eta*(-162400 + 4305*pi**2)))*w**4 + (-147840 + 8960*eta**2 
		+ 4480*eta**3 + et**4*(-221760 - 113680*eta + 94640*eta**2 + 13440*eta**3) + eta*(1127280 - 43050*pi**2) + OTS*(-67200 - 53760*eta**2 + eta*(674240 - 8610*pi**2)) + et**2*(-194880 - 112000*eta**2 - 11200*eta**3 
		+ (-739200 + 544320*eta - 127680*eta**2)*OTS + eta*(692928 + 12915*pi**2)))*w**5)*x**3)/(13440*OTS**6*w**5)))/w**2)

	return w22




def x_x1(x1,et,eta,order):


	x_x1_0=x1
	x_x1_1=(-2*x1**2)/(-1 + et**2)
	x_x1_2=(((72 + 51*et**2 - 28*eta - 26*et**2*eta)*x1**3)/(6*(-1 + et**2)**2))
	x_x1_3=(((-1920 - 1920*et**2 + 3840*et**4 - 16000*sqrt(1 - et**2) - 26496*et**2*sqrt(1 - et**2) - 2496*et**4*sqrt(1 - et**2) 
		+ 768*eta + 768*et**2*eta - 1536*et**4*eta + 24480*sqrt(1 - et**2)*eta + 27008*et**2*sqrt(1 - et**2)*eta + 1760*et**4*sqrt(1 - et**2)*eta - 896*sqrt(1 - et**2)*eta**2 
		- 5120*et**2*sqrt(1 - et**2)*eta**2 - 1040*et**4*sqrt(1 - et**2)*eta**2 - 492*sqrt(1 - et**2)*eta*pi**2 - 123*et**2*sqrt(1 - et**2)*eta*pi**2)*x1**4)/(192*sqrt(1 - et**2)*(-1 + et**2)**3))
	
	return pn([x_x1_0,x_x1_1,x_x1_2,x_x1_3],order)




def get_k(x,et,eta):
	OTS=sqrt(1-et*et)
	w1=(3*x*(1+(1/12)*x*((51-26*eta)*et**2+54-28*eta)/OTS**2)/OTS**2)
	return w1

def get_l(x,et,eta,u):
	
	OTS=sqrt(1-et*et)
	bb=get_beta(x,et,eta)
	vmu=2*arctan(bb*sin(u)/(1-bb*cos(u)))
	w=et*cos(u)-1
	
	w1=(u-et*sin(u)+(1/8)*x**2*((60-24*eta)*w*vmu+(-eta**2+15*eta)*et*sin(u)*OTS)/(OTS*w))
	return w1

def get_beta(x,et,eta):
	OTS=sqrt(1-et*et)
	w1=((1-OTS)*(1+(8-2*eta+(4-eta)/OTS)*x+((89/48)*eta**2-(1043/48)*eta+40+((137/96)*eta**2+43-(2003/96)*eta)/OTS+((85/16)*eta+35/2-(7/16)*eta**2)/OTS**2+(9+(1/32)*eta**2+(69/32)*eta)/OTS**3)*x**2)/et)
	return w1

def get_w(x,et,eta,u):
	
	OTS=sqrt(1-et*et)
	bb=get_beta(x,et,eta)
	vmu=2*arctan(bb*sin(u)/(1-bb*cos(u)))
	w=et*cos(u)-1
	et_snu=et*sin(u)
	w1=(et*sin(u)+vmu+3*(vmu+et*sin(u))*x/OTS**2+(1/32)*(8*(((-12*eta+30)*et**2-30+12*eta)*OTS+(51-26*eta)*et**2+54-28*eta)*w**3*vmu+(-4*OTS**5*eta*(-1+3*eta)+8*(18*eta+1)*OTS**3*w+(((3*eta**2-eta)*et**2-148*eta-8+12*eta**2)*OTS
		+(-60*eta+4*eta**2)*et**4+(120*eta-8*eta**2)*et**2-60*eta+4*eta**2)*w**2+((-208*eta+408)*et**2-224*eta+432)*w**3)*et_snu)*x**2/(w**3*OTS**4))
	return w1

from scipy.optimize import minimize, fsolve

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