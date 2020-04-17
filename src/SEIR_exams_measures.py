import numpy as np
import math
from time import time
from sklearn.model_selection import ParameterGrid

def sigmoid(x, x0, k, b = 0., L = 1.):
    y = L / (1 + np.exp(-k*(x-x0)))+b
    return (y)

def SEIR_exams_measures (t, s0, e0, i0, r0, c0, c0m, i_dates_betas, betas, sigma, gamma, a_date, k, 
                         a = 0., Imax = np.inf):
    num_steps = len(t) - 1
    if (len(i_dates_betas) != len(betas) - 1):
        raise Exception ("The size of i_dates_beta should be equal to the size of betas - 1. Got:", 
                         len(i_dates_beta), len(betas))
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

    step_ini = 0
    for i_betas in range(len(i_dates_betas) + 1):
        if i_betas == len(i_dates_betas):
            step_end = num_steps - 1
        else:
            step_end = i_dates_betas[i_betas]
        beta = betas[i_betas]
        for step in range(step_ini, step_end + 1):
            dt = t[step+1]-t[step]
            S[step+1] = S[step] + (-beta*I[step]*S[step])*dt
            E[step+1] = E[step] + (beta*I[step]*S[step] - sigma*E[step])*dt
            I[step+1] = I[step] + (sigma*E[step] - gamma*I[step])*dt
            R[step+1] = R[step] + gamma*I[step]*dt
            C[step+1] = C[step] + sigma*E[step]*dt

        dCdt = sigma*E
        alphas_C = 1 + (a-1.)/(1+np.exp(-k*(dCdt-a_date)))
        for step in range(step_ini, step_end + 1):
            C_m[step + 1] = C_m[step] + alphas_C[step]*sigma*E[step]*dt

        step_ini = step_end + 1

    return S, E, I, R, C, C_m

def SEIR_exams_measures_backward (t, s0, e0, i0, r0, c0, c0m,
                                  i_dates_betas, betas, sigma, gamma, a_date, k, 
                                  a = 0., Imax = np.inf):
    num_steps = len(t) - 1
    if (len(i_dates_betas) != len(betas) - 1):
        raise Exception ("The size of i_dates_beta should be equal to the size of betas - 1. Got:", 
                         len(i_dates_beta), len(betas))

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

    step_ini = 0
    for i_betas in range(len(i_dates_betas) + 1):
        if i_betas == len(i_dates_betas):
            step_end = num_steps - 1
        else:
            step_end = i_dates_betas[i_betas]
        beta = betas[i_betas]
        for step in range(step_ini, step_end + 1):
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
        for step in range(step_ini, step_end + 1):
            dt = t[step+1]-t[step]
            C_m[step + 1] = C_m[step] + alphas_C[step+1]*sigma*E[step + 1]*dt

        step_ini = step_end + 1

    return S, E, I, R, C, C_m

def ValidateSEIR_exams_IR (I_real, R_real, I, R):
    return np.sqrt(((R_real - R)**2 + (I_real - I)**2).sum()/(len(I) + len(R)))

def ValidateSEIR_exams_I (I_real, I):
    return np.sqrt(((I_real - I)**2).sum()/(len(I)))

def GridSearchSEIR_exams_measures (ts, s0, e0, i0, r0, c0, c0m, C_real, i_dates_betas = [],
                                   transmission_coeffs = np.array([10**np.arange(-15, -4, 1., dtype = float)]), # 1 / day person
                                   latency_time = np.arange(5., 15, 1.), # days
                                   infectious_time = np.arange(5., 22, 1.), # days
                                   ks = 10**np.linspace(-3, -1, 9),
                                   a_dates = np.arange (5, 41, 5),
                                   a_s = np.linspace(0.25, 0.75, 9), val_R = False, backward = False):
    
    tcs = transmission_coeffs

    d = {"b" + str(i): tcs[i] for i in range(len(tcs))}
    grid = ParameterGrid(d)

    #betas = transmission_coeff
    sigmas = 1./latency_time
    gammas = 1./infectious_time
    # ks = 10**np.linspace(-3, -1, 9)
    # a_dates = np.linspace (ts[0], ts[-1], 9)
    # a_s = np.linspace(0.25, 0.75, 9)

    min_ = [np.inf, tcs[0], sigmas[0], gammas[0]]
    print (d)

    #print (betas, sigmas, gammas)
    results = {}
    for g in grid:
        tm = time()
        betas = np.zeros(len(tcs))
        for k in g.keys():
            i_g = int(k[1:])
            betas[i_g] = g[k]
        for sigma in sigmas:
            for gamma in gammas:
                for k in ks:
                    for a in a_s:
                        for a_date in a_dates:
                            if backward:
                                S, E, I, R, C, Cm = SEIR_exams_measures_backward (ts, s0, e0, i0, r0, c0, c0m,
                                                                         i_dates_betas,
                                                                         betas, sigma, gamma, a_date, k, a)
                            else:
                                S, E, I, R, C, Cm = SEIR_exams_measures (ts, s0, e0, i0, r0, c0, c0m,
                                                                         i_dates_betas,
                                                                         betas, sigma, gamma, a_date, k, a)
                            if val_R:
                                RMSE = ValidateSEIR_exams_IR (I_real, R_real, Im, Rm)
                            else:
                                RMSE = ValidateSEIR_exams_I (C_real, Cm)
                            if (RMSE < min_[0]):
                                min_ = [RMSE, betas, sigma, gamma, a_date, k, a]

        print ("betas = ", betas, time() - tm)
        print ("  min: RMSE = ", min_[0], "; b, s, g = ", min_[1:4], "; a_d, k, a = ", min_[4:], "; (", 1./min_[2], ", ", 1./min_[3], ")")
    return min_

