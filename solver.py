"""
ecam Apr18
""" 

import numpy as np #Numerical methods
from scipy.integrate import ode #ODE solver
from param import param_dict,param_array
from data import read_data_file
from random import choice
from scipy.signal import argrelextrema
from scipy.stats import pearsonr

import scipy.optimize



T = read_data_file("time") #read time for time points
index_of_0 = np.where( T == 0.)[0][0]
GG = np.linspace(0.001,20,500)

def F(t,y,k0m,k1,k2m,k2p,k3,k4,k5m,k6m,k7,Kr0,Kr1,Kr2,Kr2p,Km5,Km6,Km7,Gt,Rt,Mt,k_Gp,Gpt,n):
    """
    Right hand side of ODE y'(t) = f(t,y,...)

    It receives parameters as f_args, as given py param_array (see param.py)
    G,M (...)
    """
    G=y[0]
    M=y[1]
    if len(y) > 2:
        Gp=y[2] # GEF perturbation (what's given in the data)
        Gpvis=y[3] # GEF perturbation (what's given in the data)
  
    else:
        Gp = 0.
        Gpvis = 0.
       
    k0=k0m*Kr0 # kmi =ki/Kri or ki/Kmi
    k2=k2m*Kr2
    k5=k5m*Km5
    k6=k6m*Km6
    
    J=Kr0/Rt
    K=Kr2/Rt
    u=k0*G+k1+k0*Gpt*Gp
    v=k2p*M+k2
    Q=v-u+v*J+u*K
    Z=Q**2-4*(v-u)*u*K
    #if np.isnan(Z) or Z == np.inf:
    #    print k3,R,
    #    print k0,G,k1,k0,Gpt,Gp
    #    print v,u,J
    A=Q+np.sqrt(Z)
    Ep=(2*u*K)/A
    R=Rt*Ep
    
    try:
        return np.array( [ k3*R*(Gt-G) - k4*M*G, k5*R*(Mt-M)**n/(Km5**n+(Mt-M)**n) - k6*M/(Km6+M) + k7*(Mt-M)/(Km7+(Mt-M)), k_Gp-k_Gp*Gp-k4*Gp*M, k_Gp-k_Gp*Gpvis] )    
    except ValueError:
        return np.array([0.,0.,0.,0.])

def Fdict(dp,y):
    return F(0,y,dp["k0m"],dp["k1"],dp["k2m"],dp["k2p"],dp["k3"],dp["k4"],dp["k5m"],dp["k6m"],dp["k7"],dp["Kr0"],dp["Kr1"],dp["Kr2"],dp["Kr2p"],dp["Km5"],dp["Km6"],dp["Km7"],dp["Gt"],dp["Rt"],dp["Mt"],dp["k_Gp"],dp["Gpt"],dp["n"])

def Fosc(t,y,k0m,k1,k2m,k2p,k3,k4,k5m,k6m,k7,Kr0,Kr1,Kr2,Kr2p,Km5,Km6,Km7,Gt,Rt,Mt,k_Gp,Gpt,n):
    """
    like F, but gives only G,M.
    """
    return F(t,y,k0m,k1,k2m,k2p,k3,k4,k5m,k6m,k7,Kr0,Kr1,Kr2,Kr2p,Km5,Km6,Km7,Gt,Rt,Mt,k_Gp,Gpt,n)[:2]

def Fosc_complete(t,y,k0m,k1,k2m,k2p,k3,k4,k5m,k6m,k7,Kr0,Kr1,Kr2,Kr2p,Km5,Km6,Km7,Gt,Rt,Mt,k_Gp,Gpt,n):
    """
    like F, but gives only G,M.
    """
    return Fcomplete(t,y,k0m,k1,k2m,k2p,k3,k4,k5m,k6m,k7,Kr0,Kr1,Kr2,Kr2p,Km5,Km6,Km7,Gt,Rt,Mt,k_Gp,Gpt,n)[:3]

def linearF_complete(p,y):
    f = lambda z,p=p: Fosc_complete(0,z,p["k0m"],p["k1"],p["k2m"],p["k2p"],p["k3"],p["k4"],p["k5m"],p["k6m"],p["k7"],p["Kr0"],p["Kr1"],p["Kr2"],p["Kr2p"],p["Km5"],p["Km6"],p["Km7"],p["Gt"],p["Rt"],p["Mt"],p["k_Gp"],p["Gpt"],p["n"])

    f0 = lambda y,f=f: f(y)[0]
    f1 = lambda y,f=f: f(y)[1]
    f2 = lambda y,f=f: f(y)[2]

    return np.array([scipy.optimize.approx_fprime(y,f0,1.e-8),scipy.optimize.approx_fprime(y,f1,1.e-8),scipy.optimize.approx_fprime(y,f2,1.e-8)])


    
def F01(y,k0m,k1,k2m,k2p,k3,k4,k5m,k6m,k7,Kr0,Kr1,Kr2,Kr2p,Km5,Km6,Km7,Gt,Rt,Mt,k_Gp,Gpt,n):
    return np.linalg.norm(F(0,y,k0m,k1,k2m,k2p,k3,k4,k5m,k6m,k7,Kr0,Kr1,Kr2,Kr2p,Km5,Km6,Km7,Gt,Rt,Mt,k_Gp,Gpt,n)[:2])

def DF01(y,k0m,k1,k2m,k2p,k3,k4,k5m,k6m,k7,Kr0,Kr1,Kr2,Kr2p,Km5,Km6,Km7,Gt,Rt,Mt,k_Gp,Gpt,n):
    p = {}
    p["k0m"] = k0m
    p["k1"] = k1
    p["k2m"] = k2m
    p["k2p"] = k2p
    p["k3"] = k3
    p["k4"] = k4
    p["k5m"] = k5m
    p["k6m"] = k6m
    p["k7"] = k7
    p["Kr0"] = Kr0
    p["Kr1"] = Kr1
    p["Kr2"] = Kr2
    p["Kr2p"] = Kr2p
    p["Km5"] = Km5
    p["Km6"] = Km6
    p["Km7"] = Km7
    p["Gt"] = Gt
    p["Rt"] = Rt
    p["Mt"] = Mt
    p["k_Gp"] = k_Gp
    p["Gpt"] = Gpt
    p["n"]  = n
    df =  linearF(p,y)
    
    f = G(y,k0m,k1,k2m,k2p,k3,k4,k5m,k6m,k7,Kr0,Kr1,Kr2,Kr2p,Km5,Km6,Km7,Gt,Rt,Mt,k_Gp,Gpt,n)

    return 2*f[0] * df[0] + 2*f[1]*df[1]

def G(y,k0m,k1,k2m,k2p,k3,k4,k5m,k6m,k7,Kr0,Kr1,Kr2,Kr2p,Km5,Km6,Km7,Gt,Rt,Mt,k_Gp,Gpt,n):
    return F(0,y,k0m,k1,k2m,k2p,k3,k4,k5m,k6m,k7,Kr0,Kr1,Kr2,Kr2p,Km5,Km6,Km7,Gt,Rt,Mt,k_Gp,Gpt,n)[:2]

def G_complete(y,k0m,k1,k2m,k2p,k3,k4,k5m,k6m,k7,Kr0,Kr1,Kr2,Kr2p,Km5,Km6,Km7,Gt,Rt,Mt,k_Gp,Gpt,n):
    return Fcomplete(0,y,k0m,k1,k2m,k2p,k3,k4,k5m,k6m,k7,Kr0,Kr1,Kr2,Kr2p,Km5,Km6,Km7,Gt,Rt,Mt,k_Gp,Gpt,n)[:3]

def root(p,y0 = np.array([1.,1.])):
    try:
        return scipy.optimize.least_squares(G,y0,jac=DG,args = (p["k0m"],p["k1"],p["k2m"],p["k2p"],p["k3"],p["k4"],p["k5m"],p["k6m"],p["k7"],p["Kr0"],p["Kr1"],p["Kr2"],p["Kr2p"],p["Km5"],p["Km6"],p["Km7"],p["Gt"],p["Rt"],p["Mt"],p["k_Gp"],p["Gpt"],p["n"]),bounds = ( (0.,0.),(np.inf,np.inf)))
    except ValueError:
        return np.array([0.,0])

def root_complete(p,y0 = np.array([1.,1.,1.])):
    try:
        return scipy.optimize.least_squares(G_complete,y0,jac=DG_complete,args = (p["k0m"],p["k1"],p["k2m"],p["k2p"],p["k3"],p["k4"],p["k5m"],p["k6m"],p["k7"],p["Kr0"],p["Kr1"],p["Kr2"],p["Kr2p"],p["Km5"],p["Km6"],p["Km7"],p["Gt"],p["Rt"],p["Mt"],p["k_Gp"],p["Gpt"],p["n"]),bounds = ( (0.,0.,0.),(np.inf,np.inf,np.inf)))
    except ValueError:
        return np.array([0.,0.,0.])


def compute_rho(r,p):
    """
    Takes a solution to the system (4 components) and computes RhoA
    We also need to calculate value of RhoA from this formula
    """

    
    k2=p["k2m"]*p["Kr2"]
    k5=p["k5m"]*p["Km5"]
    k6=p["k6m"]*p["Km6"]
    
    Qr=k2-(p["k0m"]*p["Kr0"]*r[:,0]+p["k1"]+p["k0m"]*p["Kr0"]*p["Gpt"]*r[:,2])+k2*p["Kr0"]/p["Rt"]+(p["k0m"]*p["Kr0"]*r[:,0]+p["k1"]+p["k0m"]*p["Kr0"]*p["Gpt"]*r[:,2])*p["Kr2"]/p["Rt"]
    Zr=Qr**2-4*(k2-(p["k0m"]*p["Kr0"]*r[:,0]+p["k1"]+p["k0m"]*p["Kr0"]*p["Gpt"]*r[:,2]))*(p["k0m"]*p["Kr0"]*r[:,0]+p["k1"]+p["k0m"]*p["Kr0"]*p["Gpt"]*r[:,2])*p["Kr2"]/p["Rt"]
    Ar=Qr+np.sqrt(Zr)
    RhoA=2*(p["k0m"]*p["Kr0"]*r[:,0]+p["k1"]+p["k0m"]*p["Kr0"]*p["Gpt"]*r[:,2])*p["Kr2"]/Ar

    RhoA = RhoA.reshape( (len(RhoA),1) )

    return RhoA

def initial_condition(p,y0 = np.array([1.,1.])):
    """
    Finds steady state of first two components of the solution
    """
    return root(p,y0)["x"]

def initial_condition_complete(p,y0 = np.array([1.,1.,1.])):
    """
    Finds steady state of first two components of the solution
    """
    r = root_complete(p,y0)
    if not isinstance(r,np.ndarray): 
        return root_complete(p,y0)["x"]

    return r
    #return root(p,y0)


def check(p):
    yy = initial_condition(p)

    return F01(yy,p["k0m"],p["k1"],p["k2m"],p["k2p"],p["k3"],p["k4"],p["k5m"],p["k6m"],p["k7"],p["Kr0"],p["Kr1"],p["Kr2"],p["Kr2p"],p["Km5"],p["Km6"],p["Km7"],p["Gt"],p["Rt"],p["Mt"],p["k_Gp"],p["Gpt"],p["n"])

def f11(y,p):
    
    k0=p["k0m"]*p["Kr0"]
    k2=p["k2m"]*p["Kr2"]
    k5=p["k5m"]*p["Km5"]
    k6=p["k6m"]*p["Km6"]
    Q=k2-(k0*y[0]+p["k1"])+k2*p["Kr0"]/p["Rt"]+(k0*y[0]+p["k1"])*p["Kr2"]/p["Rt"]
    Qg=-k0+k0*p["Kr2"]/p["Rt"]
    Qm=0
    Z=Q**2-4*(k2-(k0*y[0]+p["k1"]))*(k0*y[0]+p["k1"])*p["Kr2"]/p["Rt"]
    Zg=2*Q*Qg-4*(-k0*(k0*y[0]+p["k1"])+k0*(k2-(k0*y[0]+p["k1"])))*p["Kr2"]/p["Rt"]
    Zm=0
    A=Q+np.sqrt(Z)
    Ag=Qg+Zg/(2*np.sqrt(Z))
    Am=0
    R=2*(k0*y[0]+p["k1"])*p["Kr2"]/A
    Rg=2*(k0*A-Ag*(k0*y[0]+p["k1"]))*p["Kr2"]/A**2
    Rm=0

    return p["k3"]*(Rg*(p["Gt"]-y[0])-R)-p["k4"]*y[1] # corrected bracket here
    
def f12(y,p):

    
    k0=p["k0m"]*p["Kr0"]
    k2=p["k2m"]*p["Kr2"]
    k5=p["k5m"]*p["Km5"]
    k6=p["k6m"]*p["Km6"]
    Q=k2-(k0*y[0]+p["k1"])+k2*p["Kr0"]/p["Rt"]+(k0*y[0]+p["k1"])*p["Kr2"]/p["Rt"]
    Qg=-k0+k0*p["Kr2"]/p["Rt"]
    Qm=0
    Z=Q**2-4*(k2-(k0*y[0]+p["k1"]))*(k0*y[0]+p["k1"])*p["Kr2"]/p["Rt"]
    Zg=2*Q*Qg-4*(-k0*(k0*y[0]+p["k1"])+k0*(k2-(k0*y[0]+p["k1"])))*p["Kr2"]/p["Rt"]
    Zm=0
    A=Q+np.sqrt(Z)
    Ag=Qg+Zg/(2*np.sqrt(Z))
    Am=0
    R=2*(k0*y[0]+p["k1"])*p["Kr2"]/A
    Rg=2*(k0*A-Ag*(k0*y[0]+p["k1"]))*p["Kr2"]/A**2
    Rm=0

    return -p["k4"]*y[0]
    
def f21(y,p):

    k0=p["k0m"]*p["Kr0"]
    k2=p["k2m"]*p["Kr2"]
    k5=p["k5m"]*p["Km5"]
    k6=p["k6m"]*p["Km6"]
    Q=k2-(k0*y[0]+p["k1"])+k2*p["Kr0"]/p["Rt"]+(k0*y[0]+p["k1"])*p["Kr2"]/p["Rt"]
    Qg=-k0+k0*p["Kr2"]/p["Rt"]
    Qm=0
    Z=Q**2-4*(k2-(k0*y[0]+p["k1"]))*(k0*y[0]+p["k1"])*p["Kr2"]/p["Rt"]
    Zg=2*Q*Qg-4*(-k0*(k0*y[0]+p["k1"])+k0*(k2-(k0*y[0]+p["k1"])))*p["Kr2"]/p["Rt"]
    Zm=0
    A=Q+np.sqrt(Z)
    Ag=Qg+Zg/(2*np.sqrt(Z))
    Am=0
    R=2*(k0*y[0]+p["k1"])*p["Kr2"]/A
    Rg=2*(k0*A-Ag*(k0*y[0]+p["k1"]))*p["Kr2"]/A**2
    Rm=0

    return k5*Rg*(p["Mt"]-y[1])**p["n"]/(p["Km5"]**p["n"]+(p["Mt"]-y[1])**p["n"]) 
    
def f22(y,p):

    k0=p["k0m"]*p["Kr0"]
    k2=p["k2m"]*p["Kr2"]
    k5=p["k5m"]*p["Km5"]
    k6=p["k6m"]*p["Km6"]
    Q=k2-(k0*y[0]+p["k1"])+k2*p["Kr0"]/p["Rt"]+(k0*y[0]+p["k1"])*p["Kr2"]/p["Rt"]
    Qg=-k0+k0*p["Kr2"]/p["Rt"]
    Qm=0
    Z=Q**2-4*(k2-(k0*y[0]+p["k1"]))*(k0*y[0]+p["k1"])*p["Kr2"]/p["Rt"]
    Zg=2*Q*Qg-4*(-k0*(k0*y[0]+p["k1"])+k0*(k2-(k0*y[0]+p["k1"])))*p["Kr2"]/p["Rt"]
    Zm=0
    A=Q+np.sqrt(Z)
    Ag=Qg+Zg/(2*np.sqrt(Z))
    Am=0
    R=2*(k0*y[0]+p["k1"])*p["Kr2"]/A
    Rg=2*(k0*A-Ag*(k0*y[0]+p["k1"]))*p["Kr2"]/A**2
    Rm=0
    
    return -k5*R*p["n"]*((p["Mt"]-y[1])**(p["n"]-1))*p["Km5"]**p["n"]/(p["Km5"]**p["n"]+(p["Mt"]-y[1])**p["n"])**2 - k6*p["Km6"]/(p["Km6"]+y[1])**2 - p["k7"]*p["Km7"]/(p["Km7"]+p["Mt"]-y[1])**2
        

def linearF(p,y):
    return np.array([[f11(y,p), f12(y,p)], [f21(y,p),f22(y,p)]])    

def DG(y,k0m,k1,k2m,k2p,k3,k4,k5m,k6m,k7,Kr0,Kr1,Kr2,Kr2p,Km5,Km6,Km7,Gt,Rt,Mt,k_Gp,Gpt,n):
    p = {}
    p["k0m"] = k0m
    p["k1"] = k1
    p["k2m"] = k2m
    p["k2p"] = k2p
    p["k3"] = k3
    p["k4"] = k4
    p["k5m"] = k5m
    p["k6m"] = k6m
    p["k7"] = k7
    p["Kr0"] = Kr0
    p["Kr1"] = Kr1
    p["Kr2"] = Kr2
    p["Kr2p"] = Kr2p
    p["Km5"] = Km5
    p["Km6"] = Km6
    p["Km7"] = Km7
    p["Gt"] = Gt
    p["Rt"] = Rt
    p["Mt"] = Mt
    p["k_Gp"] = k_Gp
    p["Gpt"] = Gpt
    p["n"] =  n

    return linearF(p,y)

def DG_complete(y,k0m,k1,k2m,k2p,k3,k4,k5m,k6m,k7,Kr0,Kr1,Kr2,Kr2p,Km5,Km6,Km7,Gt,Rt,Mt,k_Gp,Gpt,n):
    p = {}
    p["k0m"] = k0m
    p["k1"] = k1
    p["k2m"] = k2m
    p["k2p"] = k2p
    p["k3"] = k3
    p["k4"] = k4
    p["k5m"] = k5m
    p["k6m"] = k6m
    p["k7"] = k7
    p["Kr0"] = Kr0
    p["Kr1"] = Kr1
    p["Kr2"] = Kr2
    p["Kr2p"] = Kr2p
    p["Km5"] = Km5
    p["Km6"] = Km6
    p["Km7"] = Km7
    p["Gt"] = Gt
    p["Rt"] = Rt
    p["Mt"] = Mt
    p["k_Gp"] = k_Gp
    p["Gpt"] = Gpt
    p["n"] =  n

    return linearF_complete(p,y)
 
    
def eig(p,y):
    M = linearF(p,y)
    u = np.linalg.eigvals(M)

    return u

def eig_complete(p,y):
    M = linearF_complete(p,y)
    u = np.linalg.eigvals(M)

    return u


def eig_sign(p,y):
    u = eig(p,y)

    if np.real(u[0])<0 and np.real(u[1]) < 0:
        return -1 #both negative

    if np.real(u[0])>0 and np.real(u[1]) > 0:
        return 1 #both positivie

    return 0 #One of each 

def Fc(y,Gt,p):
    return G(y,p["k0m"],p["k1"],p["k2m"],p["k2p"],p["k3"],p["k4"],p["k5m"],p["k6m"],p["k7"],p["Kr0"],p["Kr1"],p["Kr2"],p["Kr2p"],p["Km5"],p["Km6"],p["Km7"],Gt,p["Rt"],p["Mt"],p["k_Gp"],p["Gpt"],p["n"])

def Fc_complete(y,Gt,p):
    return G_complete(y,p["k0m"],p["k1"],p["k2m"],p["k2p"],p["k3"],p["k4"],p["k5m"],p["k6m"],p["k7"],p["Kr0"],p["Kr1"],p["Kr2"],p["Kr2p"],p["Km5"],p["Km6"],p["Km7"],Gt,p["Rt"],p["Mt"],p["k_Gp"],p["Gpt"],p["n"])

def DFc(y,Gt,p):
    return DG(y,p["k0m"],p["k1"],p["k2m"],p["k2p"],p["k3"],p["k4"],p["k5m"],p["k6m"],p["k7"],p["Kr0"],p["Kr1"],p["Kr2"],p["Kr2p"],p["Km5"],p["Km6"],p["Km7"],Gt,p["Rt"],p["Mt"],p["k_Gp"],p["Gpt"],p["n"])

def DFc_complete(y,Gt,p):
    return DG_complete(y,p["k0m"],p["k1"],p["k2m"],p["k2p"],p["k3"],p["k4"],p["k5m"],p["k6m"],p["k7"],p["Kr0"],p["Kr1"],p["Kr2"],p["Kr2p"],p["Km5"],p["Km6"],p["Km7"],Gt,p["Rt"],p["Mt"],p["k_Gp"],p["Gpt"],p["n"])

def FFc1(Gt,y,p):
    return Fc(y,Gt[0],p)[0]

def FFc1_complete(Gt,y,p):
    return Fc_complete(y,Gt[0],p)[0]

def FFc2(Gt,y,p):
    return Fc(y,Gt[0],p)[1]

def FFc2_complete(Gt,y,p):
    return Fc_complete(y,Gt[0],p)[1]

def FFc3_complete(Gt,y,p):
    return Fc_complete(y,Gt[0],p)[2]

def D_GtFFc(Gt,y,p):
    return np.array([scipy.optimize.approx_fprime([Gt],FFc1,1.e-8,y,p),scipy.optimize.approx_fprime([Gt],FFc2,1.e-8,y,p)])

def D_GtFFc_complete(Gt,y,p):
    return np.array([scipy.optimize.approx_fprime([Gt],FFc1_complete,1.e-8,y,p),scipy.optimize.approx_fprime([Gt],FFc2_complete,1.e-8,y,p),scipy.optimize.approx_fprime([Gt],FFc3_complete,1.e-8,y,p)])

def nFc(y,Gt,p):
    return np.linalg.norm(Fc(y,Gt,p))

def nFc_complete(y,Gt,p):
    return np.linalg.norm(Fc_complete(y,Gt,p))

def rootFc(y0,Gt,p):
    try:
        return scipy.optimize.least_squares(Fc,y0,args = (Gt,p),jac=DFc,bounds = ( (0.,0.),(np.inf,np.inf) ) )["x"]
    except ValueError:
        return np.array([0,0])

def rootFc_complete(y0,Gt,p):
    try:
        return scipy.optimize.least_squares(Fc_complete,y0,args = (Gt,p),jac=DFc_complete,bounds = ( (0.,0.,0.),(np.inf,np.inf,np.inf) ) )["x"]
    except ValueError:
        return np.array([0,0,0])



def predictor(y,Gt,p,delta_Gt=0.06):
    A = DG(y,p["k0m"],p["k1"],p["k2m"],p["k2p"],p["k3"],p["k4"],p["k5m"],p["k6m"],p["k7"],p["Kr0"],p["Kr1"],p["Kr2"],p["Kr2p"],p["Km5"],p["Km6"],p["Km7"],Gt,p["Rt"],p["Mt"],p["k_Gp"],p["Gpt"],p["n"])
    b = -D_GtFFc(Gt,y,p)*delta_Gt

    return y+np.linalg.solve(A,b).ravel()
     

def corrector(y_pred,Gt,p,eps=1.e-8):
    return rootFc(y_pred,Gt,p)

def predictor_complete(y,Gt,p,delta_Gt=0.06):
    A = DG_complete(y,p["k0m"],p["k1"],p["k2m"],p["k2p"],p["k3"],p["k4"],p["k5m"],p["k6m"],p["k7"],p["Kr0"],p["Kr1"],p["Kr2"],p["Kr2p"],p["Km5"],p["Km6"],p["Km7"],Gt,p["Rt"],p["Mt"],p["k_Gp"],p["Gpt"],p["n"])
    b = -D_GtFFc_complete(Gt,y,p)*delta_Gt

    return y+np.linalg.solve(A,b).ravel()
     

def corrector_complete(y_pred,Gt,p,eps=1.e-8):
    return rootFc_complete(y_pred,Gt,p)

def fixed_points(dp,GG,eps=1.e-16):
    dp["Gt"] = GG[0]
    y0 = initial_condition(dp)

    Y = [y0]

    g = GG[0]

    for gg in GG[1:]:
        y0 = predictor(y0,g,dp,gg-g)
        y0 = corrector(y0,gg,dp,eps)
        g = gg
        Y.append(y0)
    
    return Y 
    
def fixed_points_complete(dp,GG,eps=1.e-16):
    dp["Gt"] = GG[0]
    y0 = initial_condition_complete(dp)

    Y = [y0]

    g = GG[0]

    for gg in GG[1:]:
        y0 = predictor_complete(y0,g,dp,gg-g)
        y0 = corrector_complete(y0,gg,dp,eps)
        g = gg
        Y.append(y0)
    
    return Y 
    
def to_percent(r):
    """
    Changes sol. to percentages wrt to initial value
    Only for components 0,1, and 3. Comp. 2 is 0. at time 0.
    """
    for i in [0,1,4]:
        r[:,i] = 100.*r[:,i]/(r[0,i]+1.e-9) - 100.

    r[:,3] = r[:,3] * 310.84426380000002

    return r

def solver_tmax(p,dt=1.,Tmax=500.):
    """
    Solve the pendulum ODE up to time Tmax
    Returns array of values for ang. displacement at time intervals dt.
    p is a corrected param dict (with k_Gp instead of k_Gp_rho, etc
    """

    GG = np.arange(0.0142,p["Gt"]+0.1,0.1) 
    YY = fixed_points(p,GG)
    p["Gt"] = p["Gt_osc"]

    y0 = np.concatenate( [ YY[-1],[0.,0.] ] )

    s = ode(F) 
    s.set_integrator("lsoda",nsteps=1.e6,max_step=0) #See Scipy doc for other options
    s.set_initial_value(y0,T[index_of_0]) #Initial condition 
    s.set_f_params(p["k0m"],p["k1"],p["k2m"],p["k2p"],p["k3"],p["k4"],p["k5m"],p["k6m"],p["k7"],p["Kr0"],p["Kr1"],p["Kr2"],p["Kr2p"],p["Km5"],p["Km6"],p["Km7"],p["Gt"],p["Rt"],p["Mt"],p["k_Gp"],p["Gpt"],p["n"]) #Parameters for the right hand side

    r = [y0] #Return list. Initial value.

    #While solve is OK and we are not at Tmax
    while s.t < Tmax:
        if not s.successful():
            #raise BaseException("Solver not successful")        
            return np.zeros((401,5))
        r.append(s.integrate(s.t+dt)) #Append first component (ang. disp) of result
    
    r = np.array(r) #Return numpy array for convenience.
    r = np.hstack( [ r, compute_rho(r,p) ])
    
    return r



def solver_dict(p,dt=.1,Tmax=500.):
    """
    Solve the pendulum ODE up to time Tmax
    Returns array of values for ang. displacement at time intervals dt.
    p is a corrected param dict (with k_Gp instead of k_Gp_rho, etc
    """

    y0 = np.concatenate( [ initial_condition(p),[0.,0.] ] )

    s = ode(F) #Instance of ODE integrator
    s.set_integrator("lsoda",nsteps=1.e6,max_step=0) #See Scipy doc for other options
    s.set_initial_value(y0,T[index_of_0]) #Initial condition 
    s.set_f_params(p["k0m"],p["k1"],p["k2m"],p["k2p"],p["k3"],p["k4"],p["k5m"],p["k6m"],p["k7"],p["Kr0"],p["Kr1"],p["Kr2"],p["Kr2p"],p["Km5"],p["Km6"],p["Km7"],p["Gt"],p["Rt"],p["Mt"],p["k_Gp"],p["Gpt"],p["n"]) #Parameters for the right hand side

    r = [y0] #Return list. Initial value.

    #While solve is OK and we are not at Tmax
    for t in T[index_of_0+1:]:
        if not s.successful():
            #raise BaseException("Solver not successful")        
            return np.zeros((401,5))
        r.append(s.integrate(t)) #Append first component (ang. disp) of result
    
    r = np.array(r) #Return numpy array for convenience.
    r = np.hstack( [ r, compute_rho(r,p) ])
    r = to_percent(r)
    
    return r

def solver_dict_nonorm(p,dt=.1,Tmax=500.):
    """
    Solve the pendulum ODE up to time Tmax
    Returns array of values for ang. displacement at time intervals dt.
    p is a corrected param dict (with k_Gp instead of k_Gp_rho, etc
    """

    y0 = np.concatenate( [ initial_condition(p),[0.,0.] ] )

    s = ode(F) #Instance of ODE integrator
    s.set_integrator("lsoda",nsteps=1.e6,max_step=0) #See Scipy doc for other options
    s.set_initial_value(y0,T[index_of_0]) #Initial condition 
    s.set_f_params(p["k0m"],p["k1"],p["k2m"],p["k2p"],p["k3"],p["k4"],p["k5m"],p["k6m"],p["k7"],p["Kr0"],p["Kr1"],p["Kr2"],p["Kr2p"],p["Km5"],p["Km6"],p["Km7"],p["Gt"],p["Rt"],p["Mt"],p["k_Gp"],p["Gpt"],p["n"]) #Parameters for the right hand side

    r = [y0] #Return list. Initial value.

    #While solve is OK and we are not at Tmax
    for t in T[index_of_0+1:]:
        if not s.successful():
            #raise BaseException("Solver not successful")        
            return np.zeros((401,5))
        r.append(s.integrate(t)) #Append first component (ang. disp) of result
    
    r = np.array(r) #Return numpy array for convenience.
    #return r,compute_rho(r,p)
    r = np.hstack( [ r, compute_rho(r,p) ])
    #print "Initial value (GK):",r[0][ [0,-1,1] ],"(G,R,M)"
    #G=r[:,1]
    #r = to_percent(r)
    
    return r


def compute_rho_osc(r,p):
    # rho_oscillation    
    k2=p["k2m"]*p["Kr2"]
    k5=p["k5m"]*p["Km5"]
    k6=p["k6m"]*p["Km6"]
    
    Qr=k2-(p["k0m"]*p["Kr0"]*r[:,0]+p["k1"])+k2*p["Kr0"]/p["Rt"]+(p["k0m"]*p["Kr0"]*r[:,0]+p["k1"])*p["Kr2"]/p["Rt"]
    Zr=Qr**2-4*(k2-(p["k0m"]*p["Kr0"]*r[:,0]+p["k1"]))*(p["k0m"]*p["Kr0"]*r[:,0]+p["k1"])*p["Kr2"]/p["Rt"]
    Ar=Qr+np.sqrt(Zr)
    RhoA=2*(p["k0m"]*p["Kr0"]*r[:,0]+p["k1"])*p["Kr2"]/Ar

    RhoA = RhoA.reshape( (len(RhoA),) )

    return RhoA

def to_percent_gen(v,base):
    return 100.*(v/(base+1.e-9) -1.)

def solver_osc2(p,dt=1.,Tmax=20000,step=1000):
    GG = np.arange(0.01,p["Gt_osc"]+0.1,0.01)
    #print "Computing initial condition"
    YY = fixed_points(p,GG)
    #print "Fixed points completed."
    p["Gt"] = p["Gt_osc"]
    y0 = YY[-1]
    s = ode(Fosc) #Instance of ODE integrator
    s.set_integrator("lsoda",nsteps=1.e6,max_step=0) #See Scipy doc for other options
    s.set_initial_value(y0,T[index_of_0]) #Initial condition 
    s.set_f_params(p["k0m"],p["k1"],p["k2m"],p["k2p"],p["k3"],p["k4"],p["k5m"],p["k6m"],p["k7"],p["Kr0"],p["Kr1"],p["Kr2"],p["Kr2p"],p["Km5"],p["Km6"],p["Km7"],p["Gt"],p["Rt"],p["Mt"],p["k_Gp"],p["Gpt"],p["n"]) #Parameters for the right hand side

    stable = False

    #print "Initial (fixed point): ",y0

    for k in range(300):
        s.integrate(s.t+dt)
        if not s.successful():
            return False,np.zeros(800),np.zeros(800),np.zeros(800)


    while not stable and s.t < Tmax:
        r = []
        #Solve 500 steps.
        for k in range(step):
            r.append(s.integrate(s.t+dt))
            if not s.successful():
                return False,np.zeros(800),np.zeros(800),np.zeros(800)

        r = np.array(r)
        #Constant solution
        if np.linalg.norm(r[:,1] - np.max(r[:,1])) < 1.e-8:
            return False,np.zeros(800),np.zeros(800),np.zeros(800)
        
        #Local max
        lmax = argrelextrema(r[:,1],np.greater)[0]
        lmin = argrelextrema(r[:,1],np.less)[0]

        if len(lmax) == 0:
            continue

        dlmax = np.diff(lmax)
        
        if max([ np.abs( r[i,1] - r[i-1,1] ) for i in lmax]) < 1.e-3 or np.max(dlmax)-np.min(dlmax) < 5:
            #print max([ np.abs( r[i,1] - r[i-1,1] ) for i in lmax])
            stable = True
    
    if not stable:
        return False,0,0,0

    #We are stable. 
    dt = 1.
    r = []
    #L = int(800/dt)
    #L = 2500
    L = 800

    while len(r) < L:
        r.append(s.integrate(s.t+dt))
        if not s.successful():
            return False,np.zeros(800),np.zeros(800),np.zeros(800)
        
    r = np.array(r[::int(1./dt)])
    
    R = compute_rho_osc(r,p)
    return stable,r[:,0],r[:,1],R 

def solver_osc2_complete(p,dt=1.,Tmax=20000,step=1000):
    GG = np.arange(0.01,p["Gt_osc"]+0.1,0.01)
    #print "Computing initial condition"
    YY = fixed_points_complete(p,GG)
    #print "Fixed points completed."
    p["Gt"] = p["Gt_osc"]
    y0 = YY[-1]
    s = ode(Fosc_complete) #Instance of ODE integrator
    s.set_integrator("lsoda",nsteps=1.e6,max_step=0) #See Scipy doc for other options
    s.set_initial_value(y0,T[index_of_0]) #Initial condition 
    s.set_f_params(p["k0m"],p["k1"],p["k2m"],p["k2p"],p["k3"],p["k4"],p["k5m"],p["k6m"],p["k7"],p["Kr0"],p["Kr1"],p["Kr2"],p["Kr2p"],p["Km5"],p["Km6"],p["Km7"],p["Gt"],p["Rt"],p["Mt"],p["k_Gp"],p["Gpt"],p["n"]) #Parameters for the right hand side

    stable = False

    for k in range(300):
        s.integrate(s.t+dt)
        if not s.successful():
            return False,np.zeros(800),np.zeros(800),np.zeros(800)


    while not stable and s.t < Tmax:
        r = []
        #Solve 500 steps.
        for k in range(step):
            r.append(s.integrate(s.t+dt))
            if not s.successful():
                return False,np.zeros(800),np.zeros(800),np.zeros(800)

        r = np.array(r)
        #Constant solution
        if np.linalg.norm(r[:,1] - np.max(r[:,1])) < 1.e-8:
            return False,np.zeros(800),np.zeros(800),np.zeros(800)
        
        #Local max
        lmax = argrelextrema(r[:,1],np.greater)[0]
        lmin = argrelextrema(r[:,1],np.less)[0]

        if len(lmax) == 0:
            continue

        dlmax = np.diff(lmax)
        
        if max([ np.abs( r[i,1] - r[i-1,1] ) for i in lmax]) < 1.e-3 or np.max(dlmax)-np.min(dlmax) < 5:
            #print max([ np.abs( r[i,1] - r[i-1,1] ) for i in lmax])
            stable = True
    
    if not stable:
        return False,0,0,0

    #We are stable. 
    dt = 1.
    r = []
    #L = int(800/dt)
    #L = 2500
    L = 800

    while len(r) < L:
        r.append(s.integrate(s.t+dt))
        if not s.successful():
            return False,np.zeros(800),np.zeros(800),np.zeros(800)
        
    r = np.array(r[::int(1./dt)])
    
    #R = compute_rho_osc(r,p)
    return stable,r[:,0],r[:,1],r[:,2]

def solver_osc(p,dt=1.,Tmax=1000.):
    """
    Solve the pendulum ODE up to time Tmax
    Returns array of values for ang. displacement at time intervals dt.
    p is a corrected param dict (with k_Gp instead of k_Gp_rho, etc
    """

    GG = np.arange(0.1,p["Gt"]+0.1,0.1) #Note: we will fall short of p["Gt"], so we will "close" to the steady state!
    YY = fixed_points(p,GG)
    p["Gt"] = p["Gt_osc"]
    y0 = YY[-1]
    s = ode(Fosc) #Instance of ODE integrator
    s.set_integrator("lsoda",nsteps=1.e6,max_step=0) #See Scipy doc for other options
    s.set_initial_value(y0,T[index_of_0]) #Initial condition 
    s.set_f_params(p["k0m"],p["k1"],p["k2m"],p["k2p"],p["k3"],p["k4"],p["k5m"],p["k6m"],p["k7"],p["Kr0"],p["Kr1"],p["Kr2"],p["Kr2p"],p["Km5"],p["Km6"],p["Km7"],p["Gt"],p["Rt"],p["Mt"],p["k_Gp"],p["Gpt"],p["n"]) #Parameters for the right hand side

    r = []
    y = [0,0,0]
    prev_max = 0.

    #Solve 3 steps
    for k in range(3):
        y[k] = 100.* (s.integrate(s.t+dt)[1]/(y0[1] + 1.e-9) - 1.)
        if not s.successful():
            return np.zeros(401),np.zeros(401),np.zeros(401)

    rr = [y[0],y[1],y[2]]

    #Solve until we find a max.
    while y[1] < y[0] or y[1] < y[2]:
        if s.t > 6000:
            return np.zeros(401),np.zeros(401),np.zeros(401)
        y[0] = y[1]
        y[1] = y[2]
        y[2] = 100. * (s.integrate(s.t+dt)[1]/(y0[1]+1.e-9) - 1.)
        rr.append(y[2])
        if not s.successful():
            return np.zeros(401),np.zeros(401),np.zeros(401)

    prev_max = y[1]

    #One more step
    y[0] = y[1]
    y[1] = y[2]
    y[2] = 100.*(s.integrate(s.t+dt)[1]/(y0[1] + 1.e-9) - 1.)

    #Find next max
    while y[1] < y[0] or y[1] < y[2]:
        if s.t > 6000:
            return np.zeros(401),np.zeros(401),np.zeros(401)
        y[0] = y[1]
        y[1] = y[2]
        y[2] = 100.*(s.integrate(s.t+dt)[1]/(y0[1] + 1.e-9) - 1.)
        rr.append(y[2])
        if not s.successful():
            return np.zeros(401),np.zeros(401),np.zeros(401)

    new_max = y[1]
    
    #Keep solving until the amplitude is stable
    while np.abs(new_max-prev_max)>1.e-5:
        if s.t > 6000:
            return np.zeros(401),np.zeros(401),np.zeros(401)
        y[0] = y[1]
        y[1] = y[2]
        y[2] = 100.*(s.integrate(s.t+dt)[1]/(y0[1] + 1.e-9) - 1.)
        rr.append(y[2])
        if not s.successful():
            return np.zeros(401),np.zeros(401),np.zeros(401)

        if y[1] > y[0] and y[1] > y[2]:
            prev_max = new_max
            new_max = y[1]

    dt = 3.
    #dt = .1
    L = 35
   
    while len(r) < L: 
        if s.t > 6000:
            return np.zeros(401),np.zeros(401),np.zeros(401)
        r.append(s.integrate(s.t+dt)) #Append first component (ang. disp) of result
        if not s.successful():
            #raise BaseException("Solver not successful")        
            return np.zeros(401),np.zeros(401),np.zeros(401)

    r = np.array(r) #Return numpy array for convenience.

    
    R = compute_rho_osc(r,p)
    
    return to_percent_gen(r[:,0],y0[0]),to_percent_gen(r[:,1],y0[1]),to_percent_gen(R,R[0]) #Gef,Myo,Rho


def xcorr_aux2(A,B):
    return np.array([pearsonr(A,np.roll(B,i)) for i in range(-len(A),len(A)-1)])[:,0]
    
def xcorr_aux3(A,B):
    mid = len(A)/2
    extent=950 #19000 in Leif's time step.
    rg = 150 #3000 int Leif's time step
    lags = np.arange(-rg,rg)

    return np.array([np.corrcoef(A[mid-extent:mid+extent],B[mid-extent+i:mid+extent+i])[0,1] for i in lags])


def get_xcorr(G,M,R):
    """
    Cross correlation, given gef,myo,rho
    GM,GR,MR
    """
    L = 150

    gm,rm,rg = xcorr_aux3(G,M)[L-34*3:L+35*3:3],xcorr_aux3(R,M)[L-34*3:L+35*3:3],xcorr_aux3(R,G)[L-34*3:L+35*3:3]
    return gm,rm,rg

def get_period(a):
    lmax = argrelextrema(a,np.greater)[0]
    dlmax = np.diff(lmax)

    if len(dlmax) == 0:
        return -1
    period = dlmax[0] #Units are seconds, in principle...
    
    return period 

def get_lag(a,b):
    """
    lag a after b
    """
    lmax_a = argrelextrema(a,np.greater)[0]
    lmax_b = argrelextrema(b,np.greater)[0]

    if len(lmax_a)==0 or len(lmax_b)==0:
        return -10000
    
    #Take first max of b
    t = lmax_b[0]

    if len( lmax_a[ lmax_a >= t] ) == 0:
        return -10000
    #Find first max of a after t, take diff.
    lag =  lmax_a[ lmax_a >= t][0] - t

    #Normalise to [-period/2,period/2]
    if lag > get_period(a)/2.:
        lag = lag - get_period(a) 

    if lag < -get_period(a)/2.:
        lag = lag + get_period(a)

    return lag

def period_lag(G,M,R):
    period = get_period(R) 
    #Lags
    if period == -1:
        return -10000,-10000,-10000

    rag = get_lag(R,G)
    if rag == -10000:
        return -10000,-10000,-10000

    mar = get_lag(M,R)

    if mar == -10000:
        return -10000,-10000,-10000
    
    return period,rag,mar
     

def solver_period_lag(dp):
    """
    Returns cross correlation
    """
    dp["Gt"] = dp["Gt_osc"]
    dp["k_Gp"] = 0.
    dp["Gpt"] = 0.

    state,G,M,R = solver_osc2(dp)

    if state == False:
        return -10000,-10000,-10000

    return period_lag(G,M,R)

def solver_period_lag_complete(dp):
    """
    Returns cross correlation
    """
    dp["Gt"] = dp["Gt_osc"]
    dp["k_Gp"] = 0.
    dp["Gpt"] = 0.

    state,G,R,M = solver_osc2_complete(dp)

    if state == False:
        return -10000,-10000,-10000

    return period_lag(G,M,R)


def solver_xcorr(dp):
    """
    Returns cross correlation
    """
    dp["Gt"] = dp["Gt_osc"]
    dp["k_Gp"] = 2.
    dp["Gpt"] = 0.

    state,G,M,R = solver_osc2(dp)

    if state == False:
        return np.zeros(69),np.zeros(69),np.zeros(69)

    return get_xcorr(G,M,R)    

def solver_rho(dp):
    """
    Returns rho.
    dp is full param dict (with "k_Gp_rho" and "kGp_myo"
    """
    dp["Gt"] = dp["Gt_res"]
    dp["k_Gp"] = dp["k_Gp_rho"]
    dp["Gpt"] = dp["Gpt_rho"]

    r = solver_dict(dp)
      
    return r[:,4] #Rho. 
    
def solver_myo(dp):
    """
    Returns rho.
    dp is full param dict (with "k_Gp_rho" and "kGp_myo"
    """
    dp["Gt"] = dp["Gt_res"]
    dp["k_Gp"] = dp["k_Gp_myo"]
    dp["Gpt"] = dp["Gpt_myo"]

    r = solver_dict(dp)
      
    return r[:,1] #Myosin

def solver_rho_nonorm(dp):
    """
    Returns rho.
    dp is full param dict (with "k_Gp_rho" and "kGp_myo"
    """
    dp["Gt"] = dp["Gt_res"]
    dp["k_Gp"] = dp["k_Gp_rho"]
    dp["Gpt"] = dp["Gpt_rho"]

    r = solver_dict_nonorm(dp)
      
    return r[:,4] #Rho. 
    
def solver_myo_nonorm(dp):
    """
    Returns rho.
    dp is full param dict (with "k_Gp_rho" and "kGp_myo"
    """
    dp["Gt"] = dp["Gt_res"]
    dp["k_Gp"] = dp["k_Gp_myo"]
    dp["Gpt"] = dp["Gpt_myo"]

    r = solver_dict_nonorm(dp)
      
    return r[:,1] #Myosin


def sample_Gt(dp):
    YY = np.array(fixed_points(dp,GG))
    U1 = []
    U2 = []

    for yy,gg in zip(YY,GG):
        dp["Gt"] = gg
        u = eig(dp,yy)
        U1.append(u[0])
        U2.append(u[1])

    res = []

    for i in range(len(U1)):
        u1 = U1[i]
        u2 = U2[i]
        if np.iscomplex(u1) and np.real(u1) > 0:
            res.append(GG[i])

    if len(res) > 0:
        return choice(res)

    return -1

def normalised(a):
    """
    We already have normalised to starting at 0 (% change)
    Now we normalise to a max of 100.
    """
    return 100.*a / np.max(a)
    #return a
    
    

def solver(a):
    dp = param_dict(a)

    #print "Solving rho..."
    r = solver_rho(dp)
    #print "Solving myo..."
    m = solver_myo(dp)
    #print "Solving osc..."
    p,r_after_g,m_after_r = solver_period_lag(dp) 
    
    return normalised(r),normalised(m),p,r_after_g,m_after_r

def solver_nonorm(a):
    dp = param_dict(a)

    #print "Solving rho..."
    r = solver_rho(dp)
    #print "Solving myo..."
    m = solver_myo(dp)
    #print "Solving osc..."
    p,r_after_g,m_after_r = solver_period_lag(dp) 
    
    return r,m,p,r_after_g,m_after_r



def solver_rm(a):
    dp = param_dict(a)

    r = solver_rho(dp)
    m = solver_myo(dp)

    return normalised(r),normalised(m)

    
def Fcomplete(t,y,k0m,k1,k2m,k2p,k3,k4,k5m,k6m,k7,Kr0,Kr1,Kr2,Kr2p,Km5,Km6,Km7,Gt,Rt,Mt,k_Gp,Gpt,n):
    """
    Right hand side of ODE y'(t) = f(t,y,...)

    It receives parameters as f_args, as given py param_array (see param.py)
    3 components: G, R, M
    """
    k0=k0m*Kr0 # kmi =ki/Kri or ki/Kmi
    k2=k2m*Kr2
    k5=k5m*Km5
    k6=k6m*Km6

    G=y[0]
    R=y[1]
    M=y[2]
    if len(y) > 3:
        Gp=y[3] # GEF perturbation (what's given in the data)
        Gpvis=y[4] # GEF perturbation (what's given in the data)
    else:
        Gp = 0.
        Gpvis = 0

    
    return np.array( [ k3*R*(Gt-G) - k4*M*G, (k0*G+Gpt*Gp)*(Rt-R)/(Kr0+(Rt-R)) + k1*(Rt-R)/(Kr1+(Rt-R)) - k2*R/(Kr2+R), k5*R*(Mt-M)**n/(Km5**n+(Mt-M)**n) - k6*M/(Km6+M) + k7*(Mt-M)/(Km7+(Mt-M)),k_Gp-k_Gp*Gp-k4*Gp*M, k_Gp-k_Gp*Gpvis] )

def solver_complete(a):
    dp = param_dict(a)

    #print "Solving rho..."
    r = solver_rho_complete(dp)
    #print "Solving myo..."
    m = solver_myo_complete(dp)
    #print "Solving osc..."
    p,r_after_g,m_after_r = solver_period_lag_complete(dp) 
    
    return normalised(r),normalised(m),p,r_after_g,m_after_r

def solver_complete_nonorm(a):
    dp = param_dict(a)

    #print "Solving rho..."
    r = solver_rho_complete_nonorm(dp)
    #print "Solving myo..."
    m = solver_myo_complete_nonorm(dp)
    #print "Solving osc..."
    p,r_after_g,m_after_r = solver_period_lag_complete(dp) 
    
    return r,m,p,r_after_g,m_after_r
    #return r,m


def solver_rho_complete(dp):
    """
    Returns rho.
    dp is full param dict (with "k_Gp_rho" and "kGp_myo"
    """
    dp["Gt"] = dp["Gt_res"]
    dp["k_Gp"] = dp["k_Gp_rho"]
    dp["Gpt"] = dp["Gpt_rho"]

    r = solver_dict_complete(dp)
      
    return r[:,1] #Rho. 
    
def solver_myo_complete(dp):
    """
    Returns rho.
    dp is full param dict (with "k_Gp_rho" and "kGp_myo"
    """
    dp["Gt"] = dp["Gt_res"]
    dp["k_Gp"] = dp["k_Gp_myo"]
    dp["Gpt"] = dp["Gpt_myo"]

    r = solver_dict_complete(dp)
      
    return r[:,2] #Myosin

def solver_rho_complete_nonorm(dp):
    """
    Returns rho.
    dp is full param dict (with "k_Gp_rho" and "kGp_myo"
    """
    dp["Gt"] = dp["Gt_res"]
    dp["k_Gp"] = dp["k_Gp_rho"]
    dp["Gpt"] = dp["Gpt_rho"]

    r = solver_dict_complete_nonorm(dp)
      
    return r[:,1] #Rho. 
    
def solver_myo_complete_nonorm(dp):
    """
    Returns rho.
    dp is full param dict (with "k_Gp_rho" and "kGp_myo"
    """
    dp["Gt"] = dp["Gt_res"]
    dp["k_Gp"] = dp["k_Gp_myo"]
    dp["Gpt"] = dp["Gpt_myo"]

    r = solver_dict_complete_nonorm(dp)
      
    return r[:,2] #Myosin


def solver_dict_complete(p,dt=.1,Tmax=500.):
    """
    Solve the pendulum ODE up to time Tmax
    Returns array of values for ang. displacement at time intervals dt.
    p is a corrected param dict (with k_Gp instead of k_Gp_rho, etc
    """


    aux = initial_condition(p)
    y0 = np.array( [0.0058249, 0.00116643,0.00549311,0.,0.] )

    s = ode(Fcomplete) #Instance of ODE integrator
    s.set_integrator("lsoda",nsteps=1.e6,max_step=0) #See Scipy doc for other options
    s.set_initial_value(y0,T[index_of_0]) #Initial condition 
    s.set_f_params(p["k0m"],p["k1"],p["k2m"],p["k2p"],p["k3"],p["k4"],p["k5m"],p["k6m"],p["k7"],p["Kr0"],p["Kr1"],p["Kr2"],p["Kr2p"],p["Km5"],p["Km6"],p["Km7"],p["Gt"],p["Rt"],p["Mt"],p["k_Gp"],p["Gpt"],p["n"]) #Parameters for the right hand side

    r = [y0] #Return list. Initial value.

    #While solve is OK and we are not at Tmax
    for t in T[index_of_0+1:]:
        if not s.successful():
            #raise BaseException("Solver not successful")        
            return np.zeros((401,3))
        r.append(s.integrate(t)) #Append first component (ang. disp) of result
    
    r = np.array(r) #Return numpy array for convenience.
    r = to_percent2(r)
    
    return r

def to_percent2(r):
    """
    Changes sol. to percentages wrt to initial value
    Only for components 0,1, and 3. Comp. 2 is 0. at time 0.
    """
    for i in [1,2]:
        r[:,i] = 100.*r[:,i]/(r[0,i]+1.e-9) - 100.

    return r

def solver_dict_complete_nonorm(p,dt=.1,Tmax=500.):
    """
    Solve the pendulum ODE up to time Tmax
    Returns array of values for ang. displacement at time intervals dt.
    p is a corrected param dict (with k_Gp instead of k_Gp_rho, etc
    """

    aux = initial_condition(p)
    y0 = np.array( [0.0058249, 0.00116643,0.00549311,0.,0.] )

    s = ode(Fcomplete) #Instance of ODE integrator
    s.set_integrator("lsoda",nsteps=1.e6,max_step=0) #See Scipy doc for other options
    s.set_initial_value(y0,T[index_of_0]) #Initial condition 
    s.set_f_params(p["k0m"],p["k1"],p["k2m"],p["k2p"],p["k3"],p["k4"],p["k5m"],p["k6m"],p["k7"],p["Kr0"],p["Kr1"],p["Kr2"],p["Kr2p"],p["Km5"],p["Km6"],p["Km7"],p["Gt"],p["Rt"],p["Mt"],p["k_Gp"],p["Gpt"],p["n"]) #Parameters for the right hand side

    r = [y0] #Return list. Initial value.

    #While solve is OK and we are not at Tmax
    for t in T[index_of_0+1:]:
        if not s.successful():
            #raise BaseException("Solver not successful")        
            return np.zeros((401,3))
        r.append(s.integrate(t)) #Append first component (ang. disp) of result
    
    r = np.array(r) #Return numpy array for convenience.
    r = to_percent2(r)
    
    return r


