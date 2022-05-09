'''
implementation of the selected-node stochastic simulation algorithm with
automatically generated conditional moment equations
(snSSA DOI: 10.1063/1.5021242)
'''
import numpy as np
from random import choices
from scipy.integrate import odeint
from sympy import symbols, Matrix, Function, sympify, init_printing, Array, \
    Pow, Mul, lambdify
from sympy.core.function import AppliedUndef
import string
from operator import mul
from functools import reduce
from itertools import combinations_with_replacement

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import conditionalMomentGenerator as cmg   
       
import time
import dill
dill.settings['recurse'] = True

# %% helper functions    
def systemToVars(my_system):    
    X_select = [my_system['species'][i] for i in my_system['idx_select']]
    idx_select = my_system['idx_select']
    rates = my_system['rates']
    RC = my_system['RC']
    RL = my_system['RL']
    RH = my_system['RH']    
    moment_names = [m for _,m in my_system['mapping']]
    dM = my_system['dM']
    prop = my_system['propensities']
    mapping = my_system['mapping']
    stoich_net = np.asarray(my_system['stoich_net'])
    stoch_parts = my_system['stoch_parts']
    det_parts = my_system['det_parts']
    return X_select, idx_select, rates, RC, RL, RH, moment_names, dM, prop, mapping, stoich_net, stoch_parts, det_parts

def marginalPropensities(h, dM, mapping):
    '''
    input: propensity functions and list of computed moments
    output: marginal propensities to draw the reaction waiting time and index from
    '''   
    M = symbols('M', cls=Function)
    for m in dM:
        h = [prop.subs(m, M(m)) for prop in h]
    h = [expr.subs(mapping) for expr in h]
    return h

def writeStatsToTxt(file, sys_dict, marg_prop, times, scheme, order, nsim, reactions, constants):
    with open(file + '_snSSA_stats.txt', 'w') as f:
        f.write('species: ' + str(sys_dict['species']) + '\n\n')
        f.write('selected species: ' + str([sys_dict['species'][i] for i in sys_dict['idx_select']]) + '\n\n')
        f.write('reaction rates: ' + str(sys_dict['rates']) + '\n\n')
        f.write('reaction rate constants:\n' + str(constants) + '\n\n')
        f.write('reactant stoichiometry:\n' + str(np.array(sys_dict['stoich_reac'])) + '\n\n')
        f.write('product stoichiometry:\n' + str(np.array(sys_dict['stoich_prod'])) + '\n\n')
        f.write('net stoichiometry:\n' + str(np.array(sys_dict['stoich_net'])) + '\n\n')
        f.write('\nsimulated conditional moment equations:\n')
        f.write('(equations were closed with ' + scheme + ' closure scheme up to order ' + str(order) + '.)\n')
        for m in sys_dict['dM']:
            f.write('dM(' + str(m) + ') = ' + str(sys_dict['closed_moments'][m]) + '\n')
        f.write('\nmarginal propensities: ' + str(marg_prop) + '\n\n')
        f.write('number of simulations: ' + str(nsim) + '\n\n')
        f.write('timerange: from ' + str(timerange[0]) + ' to ' + str(timerange[-1]) + ' in ' + str(len(timerange)) + ' steps \n\n')
        f.write('mean elapsed time per simulation: ' + str(round(np.mean(times),2)) + ' seconds \n\n')
        f.write('standard deviation of elapsed simulation time: ' + str(round(np.std(times),2)) + ' seconds \n\n')
        f.write('mean number of simulated reactions: ' + str(round(np.mean(reactions))) + '\n\n')
        f.write('standard deviation of number of simulated reactions: ' + str(round(np.std(reactions))) + '\n\n')
    return

# %% define reaction network & moment generation specs
stoich_reac = Matrix(((0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0), # G_1* (active)
                     (0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0), # G_1 (inactive)
                     (0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1), # G_2*
                     (0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0), # G_2
                     (0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0), # R_1
                     (0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0), # R_2
                     (0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0), # P_1
                     (0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0))) # P_2
#                     1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6
stoich_prod = Matrix(((0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0), # G_1*
                     (0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0), # G_1
                     (0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0), # G_2*
                     (0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1), # G_2
                     (1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0), # R_1
                     (0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0), # R_2
                     (0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0), # P_1
                     (0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0))) # P_2

species = ['G_1a', 'G_1i', 'G_2a', 'G_2i', 'R_1', 'R_2', 'P_1', 'P_2']
select = ['G_1a', 'G_1i', 'G_2a', 'G_2i']

# %% specify order, closure scheme, file name to save data to
order = 2
scheme = 'normal' # possible schemes: normal, gamma, lognormal
file = 'bistable_network'

# %% generate and save equations
my_system = cmg.generateEquations(stoich_reac, 
                              stoich_prod,
                              X = species,
                              #rates = constants,
                              select = select, 
                              order = order, 
                              scheme = scheme, 
                              file = file)
# %% generate easily accessible variable names from my_system
X_select, idx_select, rates, RC, RL, RH, moment_names, dM, prop, mapping, stoich_net, stoch_parts, det_parts = systemToVars(my_system)

# generate lambdified functions for propensities, deterministic and stochastic parts
marg_prop = marginalPropensities(prop, dM, mapping)
margPropFunc = lambdify([X_select, rates, moment_names], marg_prop)
det_func = lambdify([symbols('y'), symbols('t'), X_select, rates, moment_names], det_parts)
stoch_func = lambdify([X_select, rates, moment_names], stoch_parts)

# %% Monte Carlo specs
tf = 3000                   # final time
nsim =  10000                 # number of MC simulations
timerange = np.linspace(0,tf,tf)    # time points
reactions = np.zeros(nsim)          # to count simulated reactions
times = np.zeros(nsim)              # for timing
alpha = 1.1

# %% initial conditions etc.
x0 = np.array([0,1,0,1]) # G1a, G1i, G2a, G2i
M0 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1]) # R1, R2, P1, P2, R1², R1R2, P1R1, P2R1, R2², P1R2, P2R2, P1², P1P2, P1²
constants = [0.01,0.05,5.0,0.04,10**-6,5.0,0.003,10**-6,0.01,0.05,5.0,0.04,10**-6,5.0,0.003,10**-6] # values taken from snSSA SI

x_tot = np.zeros((nsim,len(timerange)+1,len(x0)))
M_tot = np.zeros((nsim,len(timerange)+1,len(M0)))
Sx = np.asarray(stoich_net[idx_select]).T
# %% run simulations

for i in range(nsim):
    # initialization
    t = 0.0
    x = np.zeros((len(timerange)+1,4))
    M = np.zeros((len(timerange)+1,14))
    x[0] = x0
    M[0] = M0
    hM = 0
    hx = 0
    reac_count = 0
    start = time.perf_counter()
    while t < tf:
        # initialize current values
        x_current = x[hx]
        M_current = M[hM]

        H0 = sum([margPropFunc(x_current, constants, M_current)[i] for i in RH])
        H = H0 + sum([margPropFunc(x_current, constants, M_current)[i] for i in RL])

        u = 1 # accept/reject probability
        L = 10**20 # proposal propensity function to draw waiting time from   
        while u > H/L:
            L = alpha * H
            t_wait = np.random.exponential(1/L)

            # integrate moment equations until waiting time
            t += t_wait
            if hM < len(timerange)-1:
                while timerange[hM] < t: # keep current moments until reaction time
                    timevec = [timerange[hM],timerange[hM+1]]
                    M[hM] = odeint(det_func, M_current,timevec, 
                       args=(x_current,constants,M_current))[-1]
                    M_current = M[hM]
                    hM+=1
                    if hM >= len(timerange)-1:
                        break
            M[hM] = M_current
            # update propensity function            
            H = H0 + sum([margPropFunc(x_current, constants, M_current)[i] for i in RL])
            u = np.random.uniform(0,1) # draw new accept/reject probability
            
        idx = np.random.choice(RH + RL,
              p = [i/H for i in [margPropFunc(x_current, constants, M_current)[j] for j in RH + RL]])
                
        if idx in RC: # add stochastic jump terms if it is a coupling reaction
            M[hM] = M_current + stoch_func(x_current,constants,M_current)[idx]
        
        if hx < len(timerange):
            while timerange[hx] < t: # keep current state until reaction time
                x[hx] = x_current
                hx+=1
                if hx >= len(timerange)-1:
                    break
        x[hx] = x_current + Sx[idx,:] # update state x with stoichiometries
        reac_count += 1

    reactions[i] = reac_count
    times[i] = time.perf_counter() - start
    
    x_tot[i] = x
    M_tot[i] = M
    
# %% save simulation data  
writeStatsToTxt(file, my_system, marg_prop, times, scheme, order, nsim, reactions, constants)
np.save(file + '_snSSA_X', x_tot)
np.save(file + '_snSSA_M', M_tot)