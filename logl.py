"""
ecam Apr18

Log-likelihood for the pendulum system
"""

import numpy as np
#Meas. data, inverse covariance matrix.
from data import get_mean_rho_gef, get_cov_rho_gef, read_data_file
from solver import solver,solver_xcorr,fixed_points,eig,solver_rm, solver_complete
from param import param_dict,param_array,default_p
from random import choice

from prior import validate

T = read_data_file("time")
index_of_0 = np.where(T==0.)[0][0]

rho_data = read_data_file("norm_mean_rho")
cov_rho = read_data_file("norm_std_rho")
rho_data = rho_data[index_of_0:]
cov_rho = cov_rho[index_of_0:]

myo_data = read_data_file("norm_mean_myo")
cov_myo = read_data_file("norm_std_myo")
myo_data = myo_data[index_of_0:]
cov_myo = cov_myo[index_of_0:]

mean_pragmar = read_data_file("mean_pragmar") #0period, 1 rag, 2 mar
std_pragmar = read_data_file("std_pragmar") #0period, 1 rag, 2 mar

GG = np.concatenate([np.arange(0.001,1.,0.01),np.arange(1.,3.,.001)])

def check_xcorr(gm,rm,rg):
    for v in [gm,rm,rg]:
        if np.linalg.norm(v) == 0:
            return False

    for v in [gm,rm,rg]:
        if (v > 1.).any():
            return False
        if (v < -1.).any():
            return False

    return True

def check_pragmar(p,rag,mar):
    for x in [p,rag,mar]:
        if x <= -10000:
            return False

    return True


def logl(a):
    """
    Computes negative log-likelihood from param. array.

    In general, covariance should be estimated first, from data.

    @param a parameter array (see param.py)
    """

    if not validate(a):
        return 100000.

    dp = param_dict(a)
    select_Gt(dp)

    if dp["Gt_osc"] < 0:
        return 100000.

    p = param_array(dp)

    R,M,P,RAG,MAR = solver(p)

    if not check_pragmar(P,RAG,MAR):
        return 100000.
    
    dR = R - rho_data
    dM = M - myo_data
    dP = P - mean_pragmar[0]
    dRAG = RAG - mean_pragmar[1]
    dMAR = MAR - mean_pragmar[2]
    dt = T[1] - T[0] #Fixed time step.

    #Note: std is 0 at origin because data is norm. that way
    l = dt*(sum(dR[1:]**2/cov_rho[1:]**2)+sum(dM[1:]**2/cov_myo[1:]**2))+(dP**2/std_pragmar[0]**2)+(dRAG**2/std_pragmar[1]**2)+(dMAR**2/std_pragmar[2]**2)

    if not isinstance(l,np.float64) or np.isnan(l):
        return 10000.

    return l

def logl_complete(a):
    """
    Computes negative log-likelihood from param. array.

    In general, covariance should be estimated first, from data.

    @param a parameter array (see param.py)
    """

    dp = param_dict(a)
    p = param_array(dp)

    R,M,P,RAG,MAR = solver_complete(p)

    if not check_pragmar(P,RAG,MAR):
        return 100000.
    
    dR = R - rho_data
    dM = M - myo_data
    dP = P - mean_pragmar[0]
    dRAG = RAG - mean_pragmar[1]
    dMAR = MAR - mean_pragmar[2]
    dt = T[1] - T[0] #Fixed time step.

    #Note: std is 0 at origin because data is norm. that way
    l = dt*(sum(dR[1:]**2/cov_rho[1:]**2)+sum(dM[1:]**2/cov_myo[1:]**2))+(dP**2/std_pragmar[0]**2)+(dRAG**2/std_pragmar[1]**2)+(dMAR**2/std_pragmar[2]**2)

    if not isinstance(l,np.float64) or np.isnan(l):
        return 10000.

    return l


def logl_rm(a):
    """
    Computes negative log-likelihood from param. array.

    In general, covariance should be estimated first, from data.

    @param a parameter array (see param.py)
    """
    R,M,p,rag,mar = solver(a)

    dR = R - rho_data
    dM = M - myo_data
    dt = T[1] - T[0] #Fixed time step.

    #Note: std is 0 at origin because data is norm. that way
    l = dt*(sum(dR[1:]**2/cov_rho[1:]**2)+sum(dM[1:]**2/cov_myo[1:]**2))

    if not isinstance(l,np.float64) or np.isnan(l):
        return 10000.

    return l

def logl_rm_complete(a):
    """
    Computes negative log-likelihood from param. array.

    In general, covariance should be estimated first, from data.

    @param a parameter array (see param.py)
    """
    R,M,p,rag,mar = solver_complete(a)

    dR = R - rho_data
    dM = M - myo_data
    dt = T[1] - T[0] #Fixed time step.

    #Note: std is 0 at origin because data is norm. that way
    l = dt*(sum(dR[1:]**2/cov_rho[1:]**2)+sum(dM[1:]**2/cov_myo[1:]**2))

    if not isinstance(l,np.float64) or np.isnan(l):
        return 10000.

    return l



def logl_xcorr(p):
    dp = param_dict(p)

    GM,RM,RG = solver_xcorr(dp) 

    dgm = GM - gm_mean
    drm = RM - rm_mean
    drg = RG - rg_mean


    l = sum(dgm**2/gm_cov**2)+sum(drm**2/rm_cov**2)+sum(drg**2/rg_cov**2)

    if not isinstance(l,np.float64) or np.isnan(l):
        return 10000.

    return l

def select_Gt(dp):
    YY = np.array(fixed_points(dp,GG))
    U1 = []
    U2 = []

    for yy,gg in zip(YY,GG):
        dp["Gt"] = gg
        u = eig(dp,yy)
        U1.append(u[0])
        U2.append(u[1])

    res = []

    if np.real(U1[0]) > 0 or np.real(U2[0]) > 0:
        dp["Gt_osc"] = -1
        return

    for i in range(len(U1)):
        u1 = U1[i]
        u2 = U2[i]
        if np.iscomplex(u1) and np.real(u1) > 0 and np.iscomplex(u2) and np.real(u2) > 0:
            res.append(GG[i])

    #if len(res) == 0:
    if len(res) < 3:
        dp["Gt_osc"] = -1
        return

    if dp["Gt_osc"] > np.min(res) and dp["Gt_osc"] < np.max(res):
        return

    dp["Gt_osc"] = choice(res[1:-1]) #Skipping boundaries due to lack of precision
    return

def select_best_Gt(dp):
    YY = np.array(fixed_points(dp,GG))
    U1 = []
    U2 = []

    for yy,gg in zip(YY,GG):
        dp["Gt"] = gg
        u = eig(dp,yy)
        U1.append(u[0])
        U2.append(u[1])

    res = []

    if np.real(U1[0]) > 0 or np.real(U2[0]) > 0:
        dp["Gt_osc"] = -1
        return

    for i in range(len(U1)):
        u1 = U1[i]
        u2 = U2[i]
        if np.iscomplex(u1) and np.real(u1) > 0 and np.iscomplex(u2) and np.real(u2) > 0:
            res.append(GG[i])

    #if len(res) == 0:
    if len(res) < 3:
        dp["Gt_osc"] = -1
        return

    if dp["Gt_osc"] > np.min(res) and dp["Gt_osc"] < np.max(res):
        return

    return res[1:-1] #Skipping boundaries due to lack of precision
    return

def logl_3(a):
    b = np.array([ 2.00298211,  0.        ,  2.05374428,  0.        ,  1.33505046, 1.93364509,  0.43171212,  0.00484814,  0.        ,  0.37480475, 0.37480475,  0.09313281,  0.09313281,  0.02655731,  0.47961928, 1.        ,  0.443     ,  1.24      ,  0.07890789,  0.03970397, 0.14138117,  0.09066641,  0.01226458,  0.7400701 ,  1.08184121]) 

    dp = param_dict(b)
    dp["Gt_res"] = a[0]
    dp["Gpt_myo"] = a[1]
    dp["Gpt_rho"] = a[2]

    R,M = solver_rm(param_array(dp))

    dR = R - rho_data
    dM = M - myo_data
    dt = T[1] - T[0] #Fixed time step.

    #Note: std is 0 at origin because data is norm. that way
    l = dt*(sum(dR[1:]**2/cov_rho[1:]**2)+sum(dM[1:]**2/cov_myo[1:]**2))
    #l = dt*(sum(dR[1:-1]**2/cov_rho[1:-1]**2)+sum(dM[1:-1]**2/cov_myo[1:-1]**2))+(dP**2/std_pragmar[0]**2)+(dRAG**2/std_pragmar[1]**2)+(dMAR**2/std_pragmar[2]**2)

    if not isinstance(l,np.float64) or np.isnan(l):
        return 10000.

    return l

def logl_rho(r):

    dp = param_dict(b)
    #dp["Gt_res"] = a[0]
    #dp["Gpt_myo"] = a[1]
    #dp["Gpt_rho"] = a[2]
    dp["Gpt_rho"] = r

    R,M = solver_rm(param_array(dp))

    dR = R - rho_data
    dt = T[1] - T[0] #Fixed time step.

    #Note: std is 0 at origin because data is norm. that way
    l = np.linalg.norm(dR)

    if not isinstance(l,np.float64) or np.isnan(l):
        return 10000.

    return l

def logl_myo(m):

    dp = param_dict(b)
    #dp["Gt_res"] = a[0]
    #dp["Gpt_myo"] = a[1]
    #dp["Gpt_rho"] = a[2]
    dp["Gpt_myo"] = m

    R,M = solver_rm(param_array(dp))

    dM = M - myo_data
    dt = T[1] - T[0] #Fixed time step.

    l = dt*(sum(dM[1:]**2/cov_myo[1:]**2))

    if not isinstance(l,np.float64) or np.isnan(l):
        return 10000.

    return l

def logl_gt(g):

    dp = param_dict(b)
    dp["Gt_res"] = g
    #dp["Gpt_myo"] = a[1]
    #dp["Gpt_rho"] = a[2]

    R,M = solver_rm(param_array(dp))

    dR = R - rho_data
    dM = M - myo_data
    dt = T[1] - T[0] #Fixed time step.

    #Note: std is 0 at origin because data is norm. that way
    l = dt*(sum(dR[1:]**2/cov_rho[1:]**2)+sum(dM[1:]**2/cov_myo[1:]**2))

    if not isinstance(l,np.float64) or np.isnan(l):
        return 10000.

    return l







def interval_Gt(dp):
    GGl = np.linspace(0.1,20.,100000)
    YY = np.array(fixed_points(dp,GGl))
    U1 = []
    U2 = []

    for yy,gg in zip(YY,GGl):
        dp["Gt"] = gg
        u = eig(dp,yy)
        U1.append(u[0])
        U2.append(u[1])

    res = []

    if np.real(U1[0]) > 0 or np.real(U2[0]) > 0:
        return 0,0

    I = []

    for i in range(len(U1)):
        u1 = U1[i]
        u2 = U2[i]
        if np.iscomplex(u1) and np.real(u1) > 0 and np.iscomplex(u2) and np.real(u2) > 0 and np.real(u1) == np.real(u2):
            res.append(GGl[i])
            I.append(i)

    return np.min(res),np.max(res)
    
