import numpy as np
import math

def SEIR_forward (t, s0, e0, i0, r0, beta, sigma, gamma):
    num_steps = len(t) - 1
    
    s = np.zeros(num_steps + 1)
    e = np.zeros(num_steps + 1)
    i = np.zeros(num_steps + 1)
    r = np.zeros(num_steps + 1)

    s[0] = s0
    e[0] = e0
    i[0] = i0
    r[0] = r0

    for step in range(num_steps):
        dt = t[step+1]-t[step]
        s[step+1] = s[step] + (-beta*i[step]*s[step])*dt
        e[step+1] = e[step] + (beta*i[step]*s[step] - sigma*e[step])*dt
        i[step+1] = i[step] + (sigma*e[step] - gamma*i[step])*dt
        r[step+1] = r[step] + gamma*i[step]*dt
        
        
    return s, e, i, r

def SEIR_backward (t, s0, e0, i0, r0, beta, sigma, gamma):
    num_steps = len(t) - 1
    
    s = np.zeros(num_steps + 1)
    e = np.zeros(num_steps + 1)
    i = np.zeros(num_steps + 1)
    r = np.zeros(num_steps + 1)

    s[0] = s0
    e[0] = e0
    i[0] = i0
    r[0] = r0

    for step in range(num_steps):
        ###Your code here.
        dt = t[step+1]-t[step]
        a_ = e[step] + s[step]
        b_ = 1 + dt * sigma
        c_ = dt*beta/(1+dt*gamma)
        d_ = 1 + c_*i[step]
        e_ = c_*dt*sigma
        p_ = (b_*d_-a_*e_)/(e_*b_)
        q_ = (s[step]-a_*d_)/(e_*b_)
        e[step + 1] = (-p_+math.sqrt(p_*p_ - 4*q_))/2.
        i[step + 1] = (i[step] + dt*sigma*e[step+1])/(1.+dt*gamma)
        s[step + 1] = s[step]/(1. + dt*beta*i[step+1]) 
        r[step + 1] = r[step] + gamma*i[step + 1]*dt
        

        
    return s, e, i, r

def ValidateSEIR (i_real, i):
    return np.sqrt(((i_real - i)**2).sum()/len(i))

def GridSearchSEIR (ts, i_real, s0, e0, i0, r0, backward = False):
    print (len(ts), len(i_real))
    #transmission_coeff = np.power(10, np.arange(-6, -10)) # 1 / day person
    transmission_coeff = 10**np.arange(-15, 1, 1, dtype = float) # 1 / day person
    latency_time = np.arange(0.5, 15, 0.5) # days
    infectious_time = np.arange(0.5, 20, 0.5) # days

    betas = transmission_coeff
    sigmas = 1./latency_time
    gammas = 1./infectious_time

    min_ = [np.inf, betas[0], sigmas[0], gammas[0]]

    #print (betas, sigmas, gammas)
    results = {}
    for beta in betas:
        for sigma in sigmas:
            for gamma in gammas:
                if backward:
                    s, e, i, r = SEIR_backward (ts, s0, e0, i0, r0, beta, sigma, gamma)
                else:
                    s, e, i, r = SEIR_forward (ts, s0, e0, i0, r0, beta, sigma, gamma)
                RMSE = ValidateSEIR (i, i_real)
                if (RMSE < min_[0]):
                    min_ = [RMSE, beta, sigma, gamma]
    return min_

def TransportSEIR_forward (t, s0s, e0s, i0s, r0s, beta, sigma, gamma, transp):
    # Asumimos transp[origen, destino]
    num_steps = len(t) - 1
    
    s_s = np.zeros((transp.shape[0], num_steps + 1))
    e_s = np.zeros((transp.shape[0], num_steps + 1))
    i_s = np.zeros((transp.shape[0], num_steps + 1))
    r_s = np.zeros((transp.shape[0], num_steps + 1))

    s_s[:, 0] = s0s
    e_s[:, 0] = e0s
    i_s[:, 0] = i0s
    r_s[:, 0] = r0s

    for step in range(num_steps):
        dt = t[step+1]-t[step]
        for i in range(transp.shape[0]):
            Nj = transp[:, i].sum()
            g_i = (transp[i]/Nj * i_s[:, step]).sum()
            s_s[i, step+1] = s_s[i, step] + (-beta*g_i*s_s[i, step])*dt
            e_s[i, step+1] = e_s[i, step] + (beta*g_i*s_s[i, step] - sigma*e_s[i, step])*dt
            i_s[i, step+1] = i_s[i, step] + (sigma*e_s[i, step] - gamma*i_s[i, step])*dt
            r_s[i, step+1] = r_s[i, step] + gamma*i_s[i, step]*dt
            
        
    return s_s, e_s, i_s, r_s

def ValidateSEIR_transport (i, i_real):
    return np.sqrt(((i_real - i)**2).sum()/len(i)) 

def GridSearchSEIR_transport (ts, i_reals, s0s, e0s, i0s, r0s, transp):
    #transmission_coeff = np.power(10, np.arange(-6, -10)) # 1 / day person
    transmission_coeff = 10**np.arange(-15, 1, 1, dtype = float) # 1 / day person
    latency_time = np.arange(0.5, 20, 0.5) # days
    infectious_time = np.arange(0.5, 20, 0.5) # days

    betas = transmission_coeff
    sigmas = 1./latency_time
    gammas = 1./infectious_time

    min_ = [np.inf, betas[0], sigmas[0], gammas[0]]

    #print (betas, sigmas, gammas)
    results = {}
    for beta in betas:
        for sigma in sigmas:
            for gamma in gammas:
                s, e, i, r = TransportSEIR_forward (ts, s0s, e0s, i0s, r0s, beta, sigma, gamma, transp)
                RMSE = ValidateSEIR_transport (i, i_reals)

                if (RMSE < min_[0]):
                    min_ = [RMSE, beta, sigma, gamma]
    return min_
