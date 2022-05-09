'''
Explanatory file for conditionalMomentGenerator.py
For more information, check the docstrings via "?<function_name>"
'''

# %% load required packages and functions of the moment generator
import numpy as np
from sympy import symbols, Matrix, Function, sympify, init_printing, Array, \
    Pow, Mul, lambdify
from sympy.core.function import AppliedUndef
import string
from operator import mul
from functools import reduce
from itertools import combinations_with_replacement
import dill
dill.settings['recurse'] = True

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import conditionalMomentGenerator as cmg

init_printing()

# %% define the reaction system for bistable gene network
# species: G_1*, G_1, G_2*, G_2, R_1, R_2, P_1, P_2
# 16 reactions and 8 species in total

stoich_reac = Matrix(((0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0), # G_1* (active)
                     (0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0), # G_1 (inactive)
                     (0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1), # G_2*
                     (0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0), # G_2
                     (0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0), # R_1
                     (0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0), # R_2
                     (0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0), # P_1
                     (0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0))) # P_2

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

# %% generate equations with a single command & save them
my_system = cmg.generateEquations(stoich_reac, 
                              stoich_prod,
                              X = species,
                              #rates = constants,
                              select = select, 
                              order = order, 
                              scheme = scheme, 
                              file = file)
'''
The system is then stored in the "human readable" <file>.txt
If no file name is specified, the system is not saved.
All the symbolic expressions can be restored from <file> via:
'''  
# %% load file
my_system_loaded, det_func, stoch_func = cmg.loadAndLambdify(file)

'''
my_system_loaded is a dict with all info & equations about the system.
det_func and stoch_func are numerically evaluable functions to use for 
numerical integration.
Check ?loadAndLambdify to see which arguments det_func and stoch_func need.
'''
# %% or generate the equations step by step
X, rates, stoich_net = cmg.reactionSystem(stoich_reac, stoich_prod, X = species)

h = cmg.propensities(X, rates, stoich_reac)

idx_select, idx_latent, RX, RL, RO, RH, RC, RZ = cmg.splitNetwork(select, 
                                                              X, stoich_net, 
                                                              stoich_reac, 
                                                              stoich_prod)

CME = cmg.chemicalMasterEquation(X, rates, stoich_reac, 
                             stoich_net, RZ, idx_latent, idx_select)

filterEqT, filterEqR = cmg.computeFilterEquation(X, rates, stoich_reac, 
                                             stoich_net, RO, RL, RZ, 
                                             idx_latent, idx_select)
filterEq = filterEqT + filterEqR

dM, conditional_moments = cmg.generateConditionalMomentEquations(X, idx_latent, 
                                                             filterEq, 
                                                             order = order)

closed_moments = cmg.closeMoments(conditional_moments, scheme = scheme, order = order)

det_part, mapping, stoch_part = cmg.sympyExprToDiffEq(dM, closed_moments, stoich_net)