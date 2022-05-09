'''
SSA for comparison with snSSA
values for constants taken from snSSA paper SI (DOI: 10.1063/1.5021242)
'''

# %% import packages
import numpy as np
import time

# %% helper functions
def propfunc(x_, const):
    G_1a, G_1i, G_2a, G_2i, R_1, R_2, P_1, P_2 = x_
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16 = const
    return [c1, c2*R_1, c3*R_1, c4*P_1, c5*G_2a*P_1, c6*G_1a, c7*G_1i, c8*G_1a,
            c9,c10*R_2,c11*R_2,c12*P_2,c13*G_1a*P_2,c14*G_2a,c15*G_2i,c16*G_2a]

# %% system definition
stoich_reac = np.array([[0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0], # G_1* (active)
                     [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0], # G_1 (inactive)
                     [0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1], # G_2*
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0], # G_2
                     [0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0], # R_1
                     [0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0], # R_2
                     [0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0], # P_1
                     [0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0]]) # P_2

stoich_prod = np.array([[0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0], # G_1*
                     [0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0], # G_1
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0], # G_2*
                     [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1], # G_2
                     [1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0], # R_1
                     [0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0], # R_2
                     [0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0], # P_1
                     [0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0]]) # P_2
S = (stoich_prod-stoich_reac).T # net stoichiometry

#species = ['G_1a', 'G_1i', 'G_2a', 'G_2i', 'R_1', 'R_2', 'P_1', 'P_2']

# %% Monte Carlo specs
tf = 3000                   # final time
nsim =  10000                 # number of MC simulations
timerange = np.linspace(0,tf,tf)    # time points
reactions = np.zeros(nsim)          # to count simulated reactions
times = np.zeros(nsim)              # for timing

# %% intial conditions
x0 = np.array([0,1,0,1,1,1,1,1]) # G1a, G1i, G2a, G2i, R1, R2, P1, P2
constants = [0.01,0.05,5.0,0.04,10**-6,5.0,0.003,10**-6,0.01,0.05,5.0,0.04,10**-6,5.0,0.003,10**-6] # values taken from snSSA SI

# %% define function to iterate over
def SSAsim(timerange,x0,constants,tf,n):
    tSSA = 0.0
    x = np.zeros((len(timerange)+1,len(x0)))
    x[0] = x0
    h = 0
    reac_count = 0
    timer = time.perf_counter()  
    while tSSA < tf:
        # Gillespie direct method
        a = propfunc(x[h], constants)
        r1, r2 = np.random.uniform(0,1,size = 2)
        idx = np.where((np.cumsum(a)>r1*np.sum(a)) == True)[0][0]
        tSSA += ((1/np.sum(a)) * np.log(1/r2)) # update time
        x_current = x[h]
        
        if h < len(timerange):
            while timerange[h] < tSSA: # keep current state until reaction time
                x[h] = x_current
                h+=1
                if h >= len(timerange)-1:
                    break
        x[h] = x_current + S[idx,:] # update state x with stoichiometries
        reac_count += 1
        
    return x, time.perf_counter() - timer, reac_count

# %% run simulations
from functools import partial # allows for multiple arguments in pooled function
from multiprocessing import Pool

times = []
reactions = []
x_tot = []

if __name__ == '__main__':    
    with Pool(4) as pool:
        results = pool.map(partial(SSAsim,timerange,x0,constants,tf), [k for k in range(nsim)])
        for r in results:
            x, timing, reac_count = r
            times.append(timing)
            reactions.append(reac_count)
            x_tot.append(x)
    file = 'bistable_network'
    np.save(file + '_times', times)
    np.save(file + '_reactions', reactions)
    np.save(file + '_SSA_X', x_tot)