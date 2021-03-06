import numpy as np
import math
from time import time

def sigmoid(x, x0, k, b = 0., L = 1.):
    y = L / (1 + np.exp(-k*(x-x0)))+b
    return (y)

def SEIR_exams (t, s0, e0, i0, r0, c0, c0m, beta, sigma, gamma, a_date, k, 
                a = 0., Imax = np.inf):
    num_steps = len(t) - 1
    
    S = np.zeros(num_steps + 1)
    E = np.zeros(num_steps + 1)
    I = np.zeros(num_steps + 1)
    R = np.zeros(num_steps + 1)
    C = np.zeros(num_steps + 1)
    C_m = np.zeros(num_steps + 1)


    S[0] = s0
    E[0] = e0
    I[0] = i0
    R[0] = r0
    C[0] = c0
    C_m[0] = c0m

    for step in range(num_steps):
        dt = t[step+1]-t[step]
        S[step+1] = S[step] + (-beta*I[step]*S[step])*dt
        E[step+1] = E[step] + (beta*I[step]*S[step] - sigma*E[step])*dt
        I[step+1] = I[step] + (sigma*E[step] - gamma*I[step])*dt
        R[step+1] = R[step] + gamma*I[step]*dt
        C[step+1] = C[step] + sigma*E[step]*dt

    dCdt = sigma*E
    alphas_C = 1 + (a-1.)/(1+np.exp(-k*(dCdt-a_date)))
    for step in range(num_steps):
        C_m[step + 1] = C_m[step] + alphas_C[step]*sigma*E[step]*dt
        #I_m[step+1] = I_m[step] + alpha * sigma * E[step]
    return S, E, I, R, C, C_m

def SEIR_exams_backward (t, s0, e0, i0, r0, c0, c0m, beta, sigma, gamma, a_date, k, 
                         a = 0., Imax = np.inf):
    num_steps = len(t) - 1
    
    S = np.zeros(num_steps + 1)
    E = np.zeros(num_steps + 1)
    I = np.zeros(num_steps + 1)
    R = np.zeros(num_steps + 1)
    C = np.zeros(num_steps + 1)
    C_m = np.zeros(num_steps + 1)

    S[0] = s0
    E[0] = e0
    I[0] = i0
    R[0] = r0
    C[0] = c0
    C_m[0] = c0m

    for step in range(num_steps):
        dt = t[step+1]-t[step]
        a_ = E[step] + S[step]
        b_ = 1 + dt * sigma
        c_ = dt*beta/(1+dt*gamma)
        d_ = 1 + c_*I[step]
        e_ = c_*dt*sigma
        p_ = (b_*d_-a_*e_)/(e_*b_)
        q_ = (S[step]-a_*d_)/(e_*b_)
        E[step + 1] = (-p_+math.sqrt(p_*p_ - 4*q_))/2.
        I[step + 1] = (I[step] + dt*sigma*E[step+1])/(1.+dt*gamma)
        S[step + 1] = S[step]/(1. + dt*beta*I[step+1]) 
        R[step + 1] = R[step] + gamma*I[step + 1]*dt
        C[step + 1] = C[step] + sigma*E[step + 1]*dt
 
    dCdt = sigma*E
    alphas_C = 1 + (a-1.)/(1+np.exp(-k*(dCdt-a_date)))
    for step in range(num_steps):
        dt = t[step+1]-t[step]
        C_m[step + 1] = C_m[step] + alphas_C[step+1]*sigma*E[step + 1]*dt
        #I_m[step+1] = I_m[step] + alpha * sigma * E[step]
        
    return S, E, I, R, C, C_m

def ValidateSEIR_exams_IR (I_real, R_real, I, R):
    return np.sqrt(((R_real - R)**2 + (I_real - I)**2).sum()/(len(I) + len(R)))

def ValidateSEIR_exams_I (I_real, I):
    return np.sqrt(((I_real - I)**2).sum()/(len(I)))

def GridSearchSEIR_exams (ts, s0, e0, i0, r0, c0, c0m, C_real, 
                          transmission_coeff = 10**np.arange(-15, -4, 1., dtype = float), # 1 / day person
                          latency_time = np.arange(5., 15, 1.), # days
                          infectious_time = np.arange(5., 22, 1.), # days
                          ks = 10**np.linspace(-3, -1, 9),
                          a_dates = np.arange (5, 41, 5),
                          a_s = np.linspace(0.25, 0.75, 9), val_R = False, backward = True):
    
    #print (len(ts), len(I_real))
    #transmission_coeff = np.power(10, np.arange(-6, -10)) # 1 / day person

    # transmission_coeff = 10**np.arange(-15, -5, 0.5, dtype = float) # 1 / day person
    # latency_time = np.arange(5., 15, 1.) # days
    # infectious_time = np.arange(5., 22, 1.) # days

    betas = transmission_coeff
    sigmas = 1./latency_time
    gammas = 1./infectious_time
    # ks = 10**np.linspace(-3, -1, 9)
    # a_dates = np.linspace (ts[0], ts[-1], 9)
    # a_s = np.linspace(0.25, 0.75, 9)

    min_ = [np.inf, betas[0], sigmas[0], gammas[0]]

    #print (betas, sigmas, gammas)
    results = {}
    for beta in betas:
        tm = time()
        for sigma in sigmas:
            for gamma in gammas:
                for k in ks:
                    for a in a_s:
                        for a_date in a_dates:
                            if backward:
                                S, E, I, R, C, Cm = SEIR_exams_backward (ts, s0, e0, i0, r0, c0, c0m, 
                                                                         beta, sigma, gamma, a_date, k, a)
                            else:
                                S, E, I, R, C, Cm = SEIR_exams (ts, s0, e0, i0, r0, c0, c0m, beta, sigma, gamma, a_date, k, a)
                            if val_R:
                                RMSE = ValidateSEIR_exams_IR (I_real, R_real, Im, Rm)
                            else:
                                RMSE = ValidateSEIR_exams_I (C_real, Cm)
                            if (RMSE < min_[0]):
                                min_ = [RMSE, beta, sigma, gamma, a_date, k, a]

        print ("beta = ", beta, time() - tm)
        print ("  min: RMSE = ", min_[0], "; b, s, g = ", min_[1:4], "; a_d, k, a = ", min_[4:], "; (", 1./min_[2], ", ", 1./min_[3], ")")
    return min_

