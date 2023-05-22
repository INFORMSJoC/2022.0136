import numpy as np
import matplotlib.pyplot as plt
import torch
import os

from sourcing_models.lib.dual_sourcing import single_index_zr_Delta, \
					       dual_index_ze_Delta, DualSourcingModel, \
                          		       capped_dual_index_parameters, \
                          		       tailored_base_surge_Q_S

from sourcing_models.lib.single_sourcing import SingleSourcingModel

def sample_trajectories_tailored_base_surge(n_trajectories,
                                            ce = 20,
                                            cr = 0,
                                            le = 0,
                                            lr = 2,
                                            h = 5,
                                            b = 495,
                                            T = 100,
                                            I0 = 0):
    
    Q_arr = np.arange(10)
    s_arr = np.arange(10)

    optimal_Q, optimal_s = tailored_base_surge_Q_S(Q_arr,
                                               	   s_arr,
                                               	   ce, 
                                               	   cr, 
                                               	   le, 
                                               	   lr,
                                               	   h, 
                                               	   b, 
                                               	   T)

    # each trajectory consists of T timesteps
    if le == 0:
        qe_trajectories = torch.zeros([n_trajectories, T+1])
    else:
        qe_trajectories = torch.zeros([n_trajectories, T+le])
    
    if lr == 0:
        qr_trajectories = torch.zeros([n_trajectories, T+1])
    else:
        qr_trajectories = torch.zeros([n_trajectories, T+lr])
        
    state_trajectories = torch.zeros([n_trajectories, T+1, 3])
    
    for i in range(n_trajectories):
        S = DualSourcingModel(ce=ce, 
                              cr=cr, 
                              le=le, 
                              lr=lr, 
                              h=h, 
                              b=b,
                              T=T, 
                              I0=I0,
                              Q=optimal_Q,
                              s=optimal_s,
                              tailored_base_surge=True)

        S.simulate()

        I = torch.tensor(S.inventory)
        D = torch.tensor(S.demand)
        qe = torch.tensor(S.qe)
        qr = torch.tensor(S.qr)
        c = torch.tensor(S.cost)
        state_trajectories[i, :, 0] = I
        state_trajectories[i, :, 1] = D
        qe_trajectories[i, :] = qe
        qr_trajectories[i, :] = qr
        state_trajectories[i, :, 2] = c
    
    return state_trajectories, qr_trajectories, qe_trajectories
    
def sample_trajectories_single_index(n_trajectories,
                                     optimization_samples = 100,
                                     seed = 1,
                                     ce = 20,
                                     cr = 0,
                                     le = 0,
                                     lr = 2,
                                     h = 5,
                                     b = 495,
                                     T = 100,
                                     zr = 100,
                                     I0 = 0):
    
    Delta_arr = np.arange(0,5)

    optimal_zr, optimal_Delta = single_index_zr_Delta(optimization_samples,
                                                      Delta_arr,
                                                      ce, 
                                                      cr, 
                                                      le, 
                                                      lr,
                                                      h, 
                                                      b, 
                                                      2000,
                                                      zr)

    # each trajectory consists of T timesteps
    if le == 0:
        qe_trajectories = torch.zeros([n_trajectories, T+1])
    else:
        qe_trajectories = torch.zeros([n_trajectories, T+le])
    
    if lr == 0:
        qr_trajectories = torch.zeros([n_trajectories, T+1])
    else:
        qr_trajectories = torch.zeros([n_trajectories, T+lr])
        
    state_trajectories = torch.zeros([n_trajectories, T+1, 3])
    
    for i in range(n_trajectories):
        S = DualSourcingModel(ce=ce, 
                              cr=cr, 
                              le=le, 
                              lr=lr, 
                              h=h, 
                              b=b,
                              T=T, 
                              I0=optimal_zr,
                              zr=optimal_zr,
                              Delta=optimal_Delta,
                              single_index=True)

        S.simulate()

        I = torch.tensor(S.inventory)
        D = torch.tensor(S.demand)
        qe = torch.tensor(S.qe)
        qr = torch.tensor(S.qr)
        c = torch.tensor(S.cost)
        state_trajectories[i, :, 0] = I
        state_trajectories[i, :, 1] = D
        qe_trajectories[i, :] = qe
        qr_trajectories[i, :] = qr
        state_trajectories[i, :, 2] = c
    
    return state_trajectories, qr_trajectories, qe_trajectories

def sample_trajectories_dual_index(n_trajectories,
                                   optimization_samples = 100,
                                   seed = 1,
                                   ce = 20,
                                   cr = 0,
                                   le = 0,
                                   lr = 2,
                                   ze = 100,
                                   h = 5,
                                   b = 495,
                                   T = 100,
                                   zr = 100):
                        
    np.random.seed(seed)
    Delta_arr = np.arange(0,7)
    optimal_ze, optimal_Delta = dual_index_ze_Delta(optimization_samples,
	                                            Delta_arr,
	                                            ce,
	                                            cr,
	                                            le,
	                                            lr,
	                                            h,
	                                            b,
	                                            2000,
	                                            ze)
    # each trajectory consists of T timesteps
    if le == 0:
        qe_trajectories = torch.zeros([n_trajectories, T+1])
    else:
        qe_trajectories = torch.zeros([n_trajectories, T+le])
    
    if lr == 0:
        qr_trajectories = torch.zeros([n_trajectories, T+1])
    else:
        qr_trajectories = torch.zeros([n_trajectories, T+lr])
        
    state_trajectories  = torch.zeros([n_trajectories, T+1, 3])
    
    for i in range(n_trajectories):
        S = DualSourcingModel(ce=ce,
                              cr=cr,
                              le=le,
                              lr=lr,
                              h=h,
                              b=b,
                              T=T,
                              I0=optimal_ze,
                              ze=optimal_ze,
                              Delta=optimal_Delta,
                              dual_index=True)

        S.simulate()
        
        I  = torch.tensor(S.inventory)
        D  = torch.tensor(S.demand)
        qe = torch.tensor(S.qe)
        qr = torch.tensor(S.qr)
        c  = torch.tensor(S.cost)
        state_trajectories[i, :, 0] = I
        state_trajectories[i, :, 1] = D
        qe_trajectories[i, :] = qe
        qr_trajectories[i, :] = qr
        state_trajectories[i, :, 2] = c
        
    return state_trajectories, qr_trajectories, qe_trajectories

def sample_trajectories_capped_dual_index(n_trajectories,
                                          seed = 1,
                                          ce = 20,
                                          cr = 0,
                                          le = 0,
                                          lr = 2,
                                          h = 5,
                                          b = 495,
                                          T = 100,
                                          I0 = 0,
                                          demand_distribution = [-1],
                                          optimal_u1 = -1,
                                          optimal_u2 = -1,
                                          optimal_u3 = -1):
                        
    np.random.seed(seed)
    u1_arr = np.arange(10)
    u2_arr = np.arange(10,23)#np.arange(8,27)
    u3_arr = np.arange(8)
    
    if optimal_u1 == -1:
        optimal_u1, optimal_u2, optimal_u3 = capped_dual_index_parameters(u1_arr,
                                                                          u2_arr,
                                                                          u3_arr,
                                                                          ce, 
                                                                          cr, 
                                                                          le, 
                                                                          lr,
                                                                          h, 
                                                                          b, 
                                                                          10000)        
        
    # each trajectory consists of T timesteps
    if le == 0:
        qe_trajectories = torch.zeros([n_trajectories, T+1], dtype=torch.int32)
    else:
        qe_trajectories = torch.zeros([n_trajectories, T+le], dtype=torch.int32)
    
    if lr == 0:
        qr_trajectories = torch.zeros([n_trajectories, T+1], dtype=torch.int32)
    else:
        qr_trajectories = torch.zeros([n_trajectories, T+lr], dtype=torch.int32)
        
    state_trajectories  = torch.zeros([n_trajectories, T+1, 3], dtype=torch.int32)
    
    for i in range(n_trajectories):
        S = DualSourcingModel(ce=ce, 
                              cr=cr, 
                              le=le, 
                              lr=lr, 
                              h=h, 
                              b=b,
                              T=T, 
                              I0=I0,
                              u1=optimal_u1,
                              u2=optimal_u2,
                              u3=optimal_u3,
                              capped_dual_index=True,
                              demand_distribution=demand_distribution)

        S.simulate()
        
        I  = torch.tensor(S.inventory)
        D  = torch.tensor(S.demand)
        qe = torch.tensor(S.qe)
        qr = torch.tensor(S.qr)
        c  = torch.tensor(S.cost)
        state_trajectories[i, :, 0] = I
        state_trajectories[i, :, 1] = D
        qe_trajectories[i, :] = qe
        qr_trajectories[i, :] = qr
        state_trajectories[i, :, 2] = c
        
    return state_trajectories, qr_trajectories, qe_trajectories

def optimal_u1_func(h,
                    b,
                    D_low,
                    D_high):

    return (h*D_low+b*D_high)/(h+b)

def optimal_u2_func(h,
                    b,
                    D_low,
                    D_high,
                    L):
    
    return (h*D_low*L+b*D_high*L)/(h+b)

def sample_trajectories_capped_dual_index_temporal(n_trajectories,
                                                   mean_demand_arr,
                                                   std_demand_arr,
                                                   seed = 1,
                                                   ce = 20,
                                                   cr = 0,
                                                   le = 0,
                                                   lr = 2,
                                                   h = 5,
                                                   b = 495,
                                                   T = 100,
                                                   I0 = 0,
                                                   demand_distribution = [-1],
                                                   standard_baseline = True):
    
    if standard_baseline:
    
        # determine optimal CDI parameters                             
        optimal_u1 = [int(optimal_u1_func(h, b, \
                 max(0,mean_demand_arr[i]-2.58*std_demand_arr[i]), \
                 mean_demand_arr[i]+2.58*std_demand_arr[i])) \
                 for i in range(len(mean_demand_arr))]
        optimal_u2 = [int(optimal_u2_func(h, b, \
                 max(0,mean_demand_arr[i]-2.58*std_demand_arr[i]), \
                 mean_demand_arr[i]+2.58*std_demand_arr[i], lr-le)) \
                 for i in range(len(mean_demand_arr))]
        optimal_u3 = optimal_u1
    
    else:
    
        l = lr-le
        optimal_u1 = [(h*max(0,mean_demand_arr[i]-2.58*std_demand_arr[i])+b*( \
             mean_demand_arr[i]+2.58*std_demand_arr[i]))/(h+b) \
             for i in range(len(mean_demand_arr))]

        optimal_u2 = [(h*sum([max(0,mean_demand_arr[i+j]-2.58*std_demand_arr[i+j]) for j in range(l)]) + b*(\
             sum([mean_demand_arr[i+j]+2.58*std_demand_arr[i+j] for j in range(l)])))/(h+b) if i < len(mean_demand_arr)-l else \
             (h*(sum([max(0,mean_demand_arr[i+j]-2.58*std_demand_arr[i+j]) for j in range(len(mean_demand_arr)-i)])+\
             (l-len(mean_demand_arr)+i)*max(0,mean_demand_arr[-1]-2.58*std_demand_arr[-1])) + b*(\
             sum([mean_demand_arr[i+j]+2.58*std_demand_arr[i+j] for j in range(len(mean_demand_arr)-i)])+\
             (l-len(mean_demand_arr)+i)*max(0,mean_demand_arr[-1]+2.58*std_demand_arr[-1])))/(h+b)
             for i in range(len(mean_demand_arr))]

        optimal_u3 =  [(h*max(0,mean_demand_arr[i+l]-2.58*std_demand_arr[i+l])+b*( \
             mean_demand_arr[i+l]+2.58*std_demand_arr[i+l]))/(h+b) if i < len(mean_demand_arr)-l else 
             (h*max(0,mean_demand_arr[-1]-2.58*std_demand_arr[-1])+b*( \
             mean_demand_arr[-1]+2.58*std_demand_arr[-1]))/(h+b) \
             for i in range(len(mean_demand_arr))]
    
    print(optimal_u1)
    # each trajectory consists of T timesteps
    if le == 0:
        qe_trajectories = torch.zeros([n_trajectories, T+1], dtype=torch.int32)
    else:
        qe_trajectories = torch.zeros([n_trajectories, T+le], dtype=torch.int32)
    
    if lr == 0:
        qr_trajectories = torch.zeros([n_trajectories, T+1], dtype=torch.int32)
    else:
        qr_trajectories = torch.zeros([n_trajectories, T+lr], dtype=torch.int32)
        
    state_trajectories  = torch.zeros([n_trajectories, T+1, 3], dtype=torch.int32)
    
    for i in range(n_trajectories):
        S = DualSourcingModel(ce=ce, 
                              cr=cr, 
                              le=le, 
                              lr=lr, 
                              h=h, 
                              b=b,
                              T=T, 
                              I0=I0,
                              u1=optimal_u1,
                              u2=optimal_u2,
                              u3=optimal_u3,
                              capped_dual_index=True,
                              demand_distribution=demand_distribution)

        S.simulate()
        
        I  = torch.tensor(S.inventory)
        D  = torch.tensor(S.demand)
        qe = torch.tensor(S.qe)
        qr = torch.tensor(S.qr)
        c  = torch.tensor(S.cost)
        state_trajectories[i, :, 0] = I
        state_trajectories[i, :, 1] = D
        qe_trajectories[i, :] = qe
        qr_trajectories[i, :] = qr
        state_trajectories[i, :, 2] = c
        
    return state_trajectories, qr_trajectories, qe_trajectories

def sample_trajectories_single_sourcing(n_trajectories,
                                        seed = 1,
                                        l = 2,
                                        h = 5,
                                        b = 495,
                                        T = 100,
                                        r = 10):
                        
    np.random.seed(seed)
    
    # each trajectory consists of T timesteps
    if l == 0:
        q_trajectories = torch.zeros([n_trajectories, T+1])
    else:
        q_trajectories = torch.zeros([n_trajectories, T+l])
        
    state_trajectories  = torch.zeros([n_trajectories, T+1, 3])
    
    for i in range(n_trajectories):
        S = SingleSourcingModel(l=l, 
                                h=h, 
                                b=b,
                                T=T, 
                                r=r,
                                I0=0,
                                optimal_base_stock=False)

        S.simulate()
        
        I  = torch.tensor(S.inventory)
        D  = torch.tensor(S.demand)
        q = torch.tensor(S.q)
        c  = torch.tensor(S.cost)
        state_trajectories[i, :, 0] = I
        state_trajectories[i, :, 1] = D
        q_trajectories[i, :] = q
        state_trajectories[i, :, 2] = c
        
    return state_trajectories, q_trajectories
