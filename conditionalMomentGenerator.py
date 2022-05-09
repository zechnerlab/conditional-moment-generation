'''
conditional moment equation generator

steps:
1) input system (stoichiometry, rates)
2) compute propensities
3) decide which species is "observed" & split network into subnetworks
4) set up chemical master equation for "latent/conditional" species
5) set up filtering equation
6) compute conditional moment equations
7) apply closure scheme
8) output / save / convert symbolic equations to numeric ones
    
'''
import numpy as np
from sympy import symbols, Matrix, Function, sympify, Array, Pow, Mul, lambdify
from sympy.core.function import AppliedUndef
import string
from operator import mul
from functools import reduce
from itertools import combinations_with_replacement
import dill # saving/restoring files with symbolic expressions
dill.settings['recurse'] = True

# %% 1) input system stoichiometry

def reactionSystem(stoich_reac, stoich_prod, X=None, rates=None):
    '''
    Generates symbolic species and rate names, as well as net stoichiometry
    of the reaction system.
    If names for rates and species are not specified, default names are used.
    
    Example:
        species names for 2 occurring species: X = ['G_on', 'G_off']
        rate names for 4 reactions: rates = ['prod', 'prod_2', 'degr', 'conv']
    
    Parameters
    ----------
    stoich_reac : sympy.Matrix()
        reaction stoichiometry of the reaction system.
    stoich_prod : sympy.Matrix()
        product stoichiometry of the reaction system.
    X : list of strings, optional
        specified names of species. 
        The default is None.
    rates : list of strings, optional
        specified names of chemical reaction rates. 
        The default is None.

    Returns
    -------
    X : list of symbols
        symbolic species names.
    rates : list of symbols
        symbolic rate names.
    stoich_net : sympy.Matrix()
        net stoichiometry matrix of reaction system.

    '''
    # dimensionality check of the matrices
    if np.shape(stoich_reac) != np.shape(stoich_prod):
        raise ValueError("stoich_reac and stoich_prod must match in dimension. stoich_reac has dimension {}, while stoich_prod has dimension {}".format(np.shape(stoich_reac), np.shape(stoich_prod)))
    
    stoich_net = stoich_prod - stoich_reac
    (nx,nreactions) = np.shape(stoich_reac)

    if type(stoich_net) != type(Matrix()):
        raise TypeError('inputs should be of type sympy.Matrix()')
 
    # check if X has the correct number of entries if it is not None
    if X != None and len(X) != np.shape(stoich_reac)[0]:
        raise ValueError("X must contain {} entries for species names, but currently contains {}.".format(np.shape(stoich_reac)[0], len(X)))
    # check if X contains str only
    if X != None:
        for x in X:
            if type(x) != str:
                raise TypeError("X must contain strings only but currently contains {}.".format(type(x)))
    if X is None:
        X = list(string.ascii_uppercase)[0:nx]
    X = [symbols(x) for x in X]

    # check if rates has the correct number of entries if it is not None
    if rates is not None and len(rates) is not np.shape(stoich_reac)[1]:
        raise ValueError("rates must contain {} entries for rate names, but currently contains {}.".format(np.shape(stoich_reac)[1], len(rates)))
    # check if rates contains str only
    if rates != None:
        for r in rates:
            if type(r) != str:
                raise TypeError("rates must contain strings only but currently contains {}.".format(type(r)))
    if rates is None:
        rates = ["c{}".format(i) for i in range(1,nreactions+1)]
    rates = [symbols(r) for r in rates]
    
    return X, rates, stoich_net

# %% 2) compute propensities

def propensities(X, rates, stoich_reac):
    '''
    Computes reaction propensities of the given reaction system following
    mass action kinetics.

    Parameters
    ----------
    X : list of symbols
        symbolic species names.
    rates : list of symbols
        symbolic rate names.
    stoich_reac : sympy.Matrix()
        reactant stoichiometry of the reaction system.

    Returns
    -------
    propensities : list of symbols or symbolic multiplications or powers
        reaction propensities.

    '''
    propensities = [r*reduce(mul, [x**stoich_reac[j,i] for j,x in enumerate(X)]) for i,r in enumerate(rates)]
    return propensities

# %% 3) decide which species is/are "observed" & split network into subnetworks

def splitNetwork(selected_species, X, stoich_net, stoich_reac, stoich_prod):
    '''
    splits the reaction system into the following subnetworks:
        RX: all reactions that modify some selected species
        RL: all reactions in RX that are driven by latent species only, i.e. 
            have only latent species as reactants
        RH: all reactions in RX that are driven by selected species, i.e. have
            some selected species as reactants
        RO: all reactions in RH that have some latent species as products
        RC: all reactions in RX that involve latent species (union of RL and RO)
        RZ: latent reaction subnetwork, containing reactions that modify
            latent species only

    The reaction subnetworks are needed for computing the filter equation.

    Parameters
    ----------
    selected_species : str or list of str
        one or several species of the system that is/are known or
        observed. They are directly simulated, whereas moment
        equations conditional on these selected_species are computed for all 
        other species. Input selected_species like this: 'C' or ['A','C']
    X : list of symbols
        symbolic species names.
    stoich_reac : sympy.Matrix()
        reactant stoichiometry of the reaction system.
    stoich_prod : sympy.Matrix()
        product stoichiometry of the reaction system.

    Returns
    -------
    idx_select : list of int
        indices for selected species in X.
    idx_latent : list of int
        indices for latent species in X.
    RX : list of int
        reaction indices for RX subnetwork.
    RL : list of int
        reaction indices for RL subnetwork.
    RO : list of int
        reaction indices for RO subnetwork.  
    RH : list of int
        DESCRIPTION
    RC : list of int
        DESCRIPTION
    RZ : list of int
        reaction indices for RZ subnetwork. 
    '''
    # check if selected species is part of X
    for s in selected_species:
        if symbols(s) not in X:
            raise ValueError("selected_species '{}' is not in X. Please choose one/several of {}.".format(s, X))
    # check that at least one species is latent
    if len(selected_species) >= len(X):
        raise ValueError("Too many selected species. At least one species should be latent (i.e., not selected).")

    
    idx_select = [i for i,e in enumerate(X) if e in symbols(set(selected_species))]
    idx_latent = [i for i,e in enumerate(X) if e not in symbols(set(selected_species))]
    
    RX = []
    RL = []
    RO = []
    RH = []
    RC = []
    RZ = []
    
    (nx,nreactions) = np.shape(stoich_reac)
    
    for i in range(nreactions):
        if False in [[stoich_net[j,i] for j in idx_latent] == [0]*len(idx_latent)] and [stoich_net[j,i] for j in idx_select] == [0]*len(idx_select):
            RZ.append(i)
    RX = list(set(np.arange(nreactions)) - set(RZ))
    for i in RX:
        if True in [stoich_reac[j,i] > 0 for j in idx_latent]:
            RL.append(i)
    RH = list(set(RX) - set(RL))
    for i in RH:
        if True in [stoich_prod[j,i] > 0 for j in idx_latent]:
            RO.append(i)
    RC = RL + RO
    
    return idx_select, idx_latent, RX, RL, RO, RH, RC, RZ


# %% 4) set up chemical master equation for latent species
    
def chemicalMasterEquation(X, rates, stoich_reac, stoich_net, RZ, idx_latent, idx_select, func_name = 'p'):
    '''
    generates a symbolic expression of the chemical master equation which only
    involves fully latent reactions (reactions of the subnetwork RZ) and latent
    species.

    Parameters
    ----------
    X : list of symbols
        symbolic species names.
    rates : list of symbols
        symbolic rate names.
    stoich_reac : sympy.Matrix()
        reactant stoichiometry of the reaction system.
    stoich_net : sympy.Matrix()
        net stoichiometry matrix of reaction system.
    RZ : list of int
        reaction indices for RZ subnetwork (reactions with latent species only)
    idx_latent : list of int
        indices for latent species in X.
    idx_select : list of int
        indices for selected species in X.
    func_name : str, optional
        function name for the probability distribution that is displayed in
        the symbolic expression of the chemical master equation.
        The default is 'p'.

    Returns
    -------
    expr : sympy expression (sympy.Add)
        chemical master equation which involves latent species only.

    '''

    func = symbols('{}'.format(func_name), cls=Function)
    t = symbols('t')
 
    expr = 0

    for i, reac in enumerate(RZ): # every latent reaction
        h_mod = rates[reac] # propensity with modified stoichiometry
        p_mod = [] # argument of probability function with modified stoichiometry

        for j in range(np.shape(stoich_net)[0]): # construct modified stoichiometry
            h_mod = h_mod * (X[j] - stoich_net[j,reac])**stoich_reac[j, reac]

        for j, x in enumerate(idx_latent): # every latent species
            p_mod.append(sympify("{} - {}".format(X[x], stoich_net[x,reac])))
        
        expr += h_mod * func(Array(p_mod),t)
        expr = expr - propensities(X, rates, stoich_reac)[reac] * func(Array([X[x] for x in idx_latent]), t)

    return expr

# %% 5) set up filtering equation
  
def getPropensityPolynomials(X, rates, stoich_reac, idx_latent, idx_select):
    '''
    split symbolic expressions of the reaction propensities in independent 
    factors of selected and latent species.
    propensity hk(x,z) = ck * gk(x) * fk(z)

    Parameters
    ----------
    X : list of symbols
        symbolic species names.
    rates : list of symbols
        symbolic rate names.
    stoich_reac : sympy.Matrix()
        reactant stoichiometry of the reaction system.
    idx_latent : list of int
        indices for latent species in X.
    idx_select : list of int
        indices for selected species in X.

    Returns
    -------
    ck : list of symbols
        contains constant rate coefficients.
    fk : list of symbols, symbolic multiplications, or powers of symbols
        polynomial part of h dependent on latent species (X_latent).
    gk : list of symbols, symbolic multiplications, or powers of symbols
        polynomial part of h dependent on selected species (X_select).

    '''

    temp = []
    fk = []
    ck = []
    gk = []
    for i,h in enumerate(propensities(X, rates, stoich_reac)):
        # as_independent(*[]) takes list of values as input
        temp.append(h.as_independent(*[X[x] for x in idx_latent], as_Mul = True)[0])
        fk.append(h.as_independent(*[X[x] for x in idx_latent], as_Mul = True)[1])
        ck.append(temp[i].as_independent(*[X[x] for x in idx_select], as_Mul = True)[0])
        gk.append(temp[i].as_independent(*[X[x] for x in idx_select], as_Mul = True)[1])    
    return ck, fk, gk
    

def getStochPrefactor(X, fk, reac, stoich_net):
    '''
    helper function in computeFilterEquation(). It computes a prefactor for 
    the stochastic part of the filter equation depending on the type of 
    reaction propensity. Different prefactors are computed for the cases
    fk[i] = 0 or fk = A (constant), 
    fk[i] = A**x (power), and
    fk[i] = A*B*... (multiplication)

    Parameters
    ----------
    X : list of symbols
        symbolic species names.
    fk : list of symbols, symbolic multiplications or powers of symbols
        polynomial part of reaction propensity dependent on a latent species
        fk[i].func is the type of function, e.g. Mul or Pow
        fk[i].args contain coefficients and/or powers
    reac : int
        reaction index.
    stoich_net : sympy.Matrix
        net stoichiometry matrix of reaction system.

    Returns
    -------
    prefactor : list of symbolic expressions
        prefactors for different reactions in the stochastic part of the
        filter equation.

    '''
    prefactor = []
    if fk[reac].func == Pow:
        prefactor.append((fk[reac].args[0] - stoich_net[X.index(fk[reac].args[0]),reac])**fk[reac].args[1])
    elif fk[reac].func == Mul:
        for i in range(len(fk[reac].args)):
            arg = fk[reac].args[i]
            if arg.func == Pow:
                prefactor.append((arg.args[0] - stoich_net[X.index(arg.args[0]),reac])**arg.args[1])
            else:
                prefactor.append(arg - stoich_net[X.index(arg),reac])
    else:
        prefactor.append(fk[reac] - stoich_net[X.index(fk[reac]),reac])
    return prefactor


def computeFilterEquation(X, rates, stoich_reac, stoich_net, RO, RL, RZ, idx_latent, idx_select):
    '''
    computes the filter equation of a reaction system given which species
    are observed / known and which species are latent / described conditioned 
    on the known species.

    Parameters
    ----------
    X : list of symbols
        symbolic species names.
    rates : list of symbols
        symbolic rate names.
    stoich_reac : sympy.Matrix()
        reactant stoichiometry of the reaction system.
    stoich_net : sympy.Matrix()
        net stoichiometry matrix of reaction system.
    RO : list of int
        reaction indices for RO subnetwork.
    RL : list of int
        reaction indices for RL subnetwork.
    RZ : list of int
        reaction indices for RZ subnetwork.
    idx_latent : list of int
        indices for latent species in X.
    idx_select : list of int
        indices for selected species in X.

    Returns
    -------
    texpr : sympy expression (sympy.Add)
        time-dependent (ODE) part of the filter equation.
    Rexpr : sympy expression (sympy.Add)
        stochastic part of the filter equation which contains reaction-
        dependent jump contributions.

    '''
    (nx,nreactions) = np.shape(stoich_reac)
    X_lat = Array([X[x] for x in idx_latent])

    func = symbols('pi', cls=Function)
    t = symbols('t')
    dt = symbols('dt')
    M = symbols('M', cls=Function)
    # define reaction counters
    R = ["dR{}".format(_) for _ in range(1,nreactions+1)]
    R = [symbols(r) for r in R]


    ### construct ODE part of filter equation
    ck, fk, gk = getPropensityPolynomials(X, rates, stoich_reac, idx_latent, idx_select)
    h = propensities(X, rates, stoich_reac)
    # first summation part
    list1 = [h[i] - ck[i]*gk[i]*M(fk[i]) for i in RL]
    list2 = [func(Array([X[x] for x in idx_latent]),t)]*len(RL)
    
    # put time-dependent parts together
    texpr = chemicalMasterEquation(X, rates, stoich_reac, stoich_net, RZ, idx_latent, idx_select, func_name = 'pi')
    if RL:
        texpr = dt*(texpr - sum([a * b for a, b in zip(list1, list2)]).subs(M(1), 0))
    else:
        texpr = dt*texpr.subs(M(1), 0)

    ### construct stochastic part of filter equation      
    # first stochastic part (all reactions in RL)
    _ = []
    for i,reac in enumerate(RL):
        prefactor = getStochPrefactor(X, fk, reac, stoich_net)
        _.append(R[reac]*(reduce(mul,prefactor)/M(fk[reac])*func(Array([X[x] - stoich_net[x,reac] for x in idx_latent]),t) - func(X_lat,t)))
    stoch1 = sum(_)
    
    # second stochastic part (all reactions in RO)
    _ = []
    for i,reac in enumerate(RO):
        _.append(R[reac]*(func(Array([X[x] - stoich_net[x,reac] for x in idx_latent]),t) - func(X_lat,t)))
    stoch2 = sum(_)
    Rexpr = stoch1 + stoch2

    return texpr, Rexpr

# %% 6) compute conditional moment equations

def flatten(t):
    '''
    flattens a nested list of list
    
    Parameters
    ----------
    t : list or nested list of list
        list to be flattened
        
    Returns
    -------
    flattened list    
    '''
    return [item for sublist in t for item in sublist]

def wantedMoments(X, idx_latent, order = 2):
    '''
    returns all possible multiplication combinations of latent species 
    symbols up to the specified order (default: 2nd order).
    Conditional moment equations will then be computed for all the symbols 
    generated here.
    
    Example: A and B are latent species and order = 2. Then wantedMoments()
    will generate the list [A, B, A**2, A*B, B**2]

    Parameters
    ----------
    X : list of symbols
        symbolic species names.
    idx_latent : list of int
        indices for latent species in X.
    order : int, optional
        order up to which conditional moment equations should be generated.
        The default is 2.

    Returns
    -------
    wanted : list of symbols or multiplications and powers of symbols
        list of moments that conditional moment equations will be generated for.

    '''
    X_lat = Array([X[x] for x in idx_latent])
    wanted = [list(combinations_with_replacement(X_lat,i)) for i in range(1,order+1)]
    wanted = flatten(wanted)
    wanted = [reduce(mul, i) for i in wanted]
    return wanted

def doIndexShifts(filterEq, m):
    '''
    Helper function in generateConditionalMomentEquations(). 
    Shifts sums that occur in the arguments of the filter distribution 
    function outside, according to some substitution rules (see examples).
    
    examples:
        A * B * pi(A+1,B,t)     -> (A-1) * B * pi(A,B,t)
        A * pi(A,B-1,t)         -> A * pi(A,B,t)
        A * (A-1) * pi(A+1,B,t) -> (A-1) * (A-2) * pi(A,B,t)

    Parameters
    ----------
    filterEq : sympy expression (sympy.Add)
        expression of the unsanitized filter equation.
    m : sympy.Symbol
        symbol of a conditional moment (an entry of the dM list of moments).

    Returns
    -------
    index_shifted_expr : sympy expression (sympy.Add)
        expression which does not contain any sums inside the filter 
        distribution function anymore.

    '''
    
    mom_terms = (filterEq * m).expand().as_ordered_terms()

    for j,t in enumerate(mom_terms):
        _ = flatten([i.args[0] for i in t.atoms(Function) if i.func.name == "pi"])
        
        # save current M() term before index is shifted
        M_old = [i for i in t.atoms(Function) if i.func.name == 'M']
        
        for k in range(len(_)):
            if _[k].is_Add:
                # if it is a sum, args[1] contains the species
                # and args[0] the integer (stoichiometric coefficient) which
                # has to be moved out of the filter function
                mom_terms[j] = mom_terms[j].subs(_[k].args[1], _[k].args[1] - _[k].args[0])
            # else: no substitution needed
        
        # back-substitute old M() term
        if M_old:
            M_new = [i for i in mom_terms[j].atoms(Function) if i.func.name == 'M']
            mom_terms[j] = mom_terms[j].subs(*M_new, *M_old)

    return sum(mom_terms).simplify()


def substitutePiToMoment(index_shifted_expr, X, idx_latent):
    '''
    Helper function in generateConditionalMomentEquations(). 
    Substitutes filter distribution function with respective symbolic moment 
    function. The function doIndexShifts() has to be applied to the filter
    equation beforehand.
    
    Examples:
        A * pi(A,B,t)           -> M(A)
        A**2 * pi(A,B,t)        -> M(A**2)
        c * A * B * pi(A,B,t)   -> c * M(A*B)

    Parameters
    ----------
    index_shifted_expr : sympy expression (sympy.Add)
        expression of the filter equation which does not contain any sums 
        inside the filter distribution function pi().
    X : list of symbols
        symbolic species names.
    idx_latent : list of int
        indices for latent species in X.

    Returns
    -------
    moment_eq : sympy expression (sympy.Add)
        expression for a given conditional moment.

    '''
    # create functions and symbols again for simple access
    pi = symbols('pi', cls=Function)
    t = symbols('t')
    M = symbols('M', cls=Function)
    
    X_lat = Array([X[x] for x in idx_latent])
    
    subs_terms = [] # list that stores latent factors that should be substituted
    
    for i,term in enumerate(index_shifted_expr.expand().as_ordered_terms()):
        subs_factor = []
        for j,atom in enumerate(term.as_ordered_factors()):
            if atom.is_Pow:
                _ = atom.args[0]
            else:
                _ = atom
            if _ in X_lat:
                subs_factor.append(atom)
        
        # if no latent species is in the prefactor, pi will be replaced by 1
        if subs_factor == []: # no latent species in the prefactor
            subs_terms.append(term.subs(pi(Array(X_lat),t), 1))
        # if there are latent species in the prefactor, replace x*pi with M(x)
        else: # substitute x*pi(A,B,t) with M(x)
            _ = reduce(mul,subs_factor)
            subs_terms.append(term.subs(_*pi(Array(X_lat),t), M(_)))
            
    return sum(subs_terms).simplify()


def generateConditionalMomentEquations(X, idx_latent, filterEq, order = 2):
    '''
    generates conditional moment equations from a given filter equation for 
    specified latent species up to a certain order (default is 2).

    Parameters
    ----------
    X : list of symbols
        symbolic species names.
    idx_latent : list of int
        indices for latent species in X.
    filterEq : sympy expression (sympy.Add)
        texpr + Rexpr from computeFilterEquation() function. This is the full
        filter equation with deterministic time-dependent and stochastic 
        reaction-dependent terms.
    order : int, optional
        order up to which the conditional moment equations for the latent
        species will be generated.
        The default is 2.

    Returns
    -------
    dM : list of symbols
        list of moments that the conditional moment equations are generated for.
    moment_eqs : dict
        dictionary which contains as keys the moments and as values their 
        respective differential conditional moment equation.
        {moment : conditional moment equation}.

    '''
    dM = wantedMoments(X, idx_latent, order = order)
    moment_eqs = {}
    for i,m in enumerate(dM):
        index_shifted_expr = doIndexShifts(filterEq, m)
        moment_eqs[m] = substitutePiToMoment(index_shifted_expr, X, idx_latent)
    return dM, moment_eqs

# %% 7) apply closure scheme

def getMomentOrders(moment_eq):
    '''
    get unique moments that occur in a single moment equation and determine 
    their orders.
    Helper function for closeMomentEq().
    
    Example:
        moment_eq:  M(A) + c * M(A) + M(A*B) - M(B**2)
        -> moms:    [M(A),  M(A*B), M(B**2)]
        -> orders:  [1,     2,      2]

    Parameters
    ----------
    moment_eq : sympy expression (sympy.Add)
        a single moment equation generated in the 
        generateConditionalMomentEquations() function.

    Returns
    -------
    moms : list of symbols or symbol multiplications / powers
        list of unique moments that occur in a given moment equation.
    factors: list of symbols or symbol multiplications / powers
        corresponding list of the arguments of the unique moments.
    orders : list of int
        corresponding list of orders of the moments in moms and factors list.

    '''
    moms = [i for j in moment_eq.expand().as_ordered_terms() for i in j.as_ordered_factors() if isinstance(i, AppliedUndef)]
    moms = list(set(moms)) # unique moments
    factors = [i.args[0] for i in moms]
    orders = [0]*len(factors)
    for i,f in enumerate(factors):
        if f.is_Symbol: orders[i] = 1
        elif f.is_Pow: orders[i] = f.args[1]
        elif f.is_Mul:
            for arg in f.as_ordered_factors():
                if arg.is_Pow: orders[i] += arg.args[1]
                if arg.is_Symbol: orders[i] += 1
    return moms, factors, orders


def closeMomentEq(moment_eq, max_order, scheme):
    '''
    applies a moment approximation scheme of a certain order to a moment 
    equation.

    Parameters
    ----------
    moment_eq : sympy expression (sympy.Add)
        a single moment equation generated in the 
        generateConditionalMomentEquations() function.
    max_order : int
        maximum order to occur in a moment equation. Higher orders are 
        approximated. Valid maximum orders are 2 or 3.
    scheme : str
        moment closure scheme. "normal", "gamma", and "lognormal" scheme 
        are supported. 

    Returns
    -------
    closed_moment_eq : sympy expr (sympy.Add)
        closed conditional moment equation

    '''

    if scheme not in ['normal', 'gamma', 'lognormal']:
        raise ValueError('Moment approximation scheme not supported. Please use one of: normal, gamma, lognormal.')
    if not isinstance(max_order, int):
        raise TypeError('order should be an integer number (2 or 3).')
    
    M = symbols('M', cls=Function)

    # create a list of the moments of the moment_eq that have order > max_order
    moms, factors, orders = getMomentOrders(moment_eq)
    wanted_factors = [factors[i] for i in [i for i,o in enumerate(orders) if o > max_order]]
    
    closed_moment_eq = moment_eq
    
    if wanted_factors:
        # iterate over moments of order > max_order
        for i,f in enumerate(wanted_factors):
            _ = [] # list of substitutions
            if f.is_Mul:
                for j,arg in enumerate(f.args):
                    if arg.is_Symbol: _.append(arg)
                    elif arg.is_Pow:
                        for k in range(arg.args[1]): # repeat this for the number of times as the power
                            _.append(arg.args[0])
            elif f.is_Pow:
                for k in range(f.args[1]):
                    _.append(f.args[0])
                    
            if max_order == 2:
                if scheme == "normal":
                    closed_moment_eq = closed_moment_eq.subs(M(reduce(mul, _)), -2*M(_[0])*M(_[1])*M(_[2]) + M(_[0]*_[1])*M(_[2]) + M(_[1]*_[2])*M(_[0]) + M(_[0]*_[2])*M(_[1]))
                elif scheme == "gamma":
                    # differentiate cases (different formulas for different number of species)
                    if len(set(_)) == 1 or len(set(_)) == 2:
                        # first and last item of the list are the different elements (_[0] contains the single variable, _[2] the powered one)
                        closed_moment_eq = closed_moment_eq.subs(M(reduce(mul, _)), 2 * M(_[2]**2)*M(_[2]*_[0])/M(_[2]) - M(_[2]**2)*M(_[0]))
                    if len(set(_)) == 3:
                        closed_moment_eq = closed_moment_eq.subs(M(reduce(mul, _)), M(_[0])*M(_[1]*_[2]) + M(_[1])*M(_[0]*_[2]) + M(_[2])*M(_[1]*_[0]) - 2*M(_[0])*M(_[1])*M(_[2]))
                elif scheme == "lognormal":
                    closed_moment_eq = closed_moment_eq.subs(M(reduce(mul, _)), M(_[0]*_[1])*M(_[1]*_[2])*M(_[0]*_[2])/(M(_[0])*M(_[1])*M(_[2])))
                    
            elif max_order == 3:
                if scheme == "normal":
                    closed_moment_eq = closed_moment_eq.subs(M(reduce(mul, _)),
                                                      M(_[0])*M(_[1]*_[2]*_[3]) + M(_[1])*M(_[0]*_[2]*_[3]) + M(_[2])*M(_[0]*_[1]*_[3]) + M(_[3])*M(_[1]*_[2]*_[0]) \
                                                          + M(_[0]*_[1])*M(_[2]*_[3]) + M(_[0]*_[2])*M(_[1]*_[3]) + M(_[0]*_[3])*M(_[2]*_[1]) \
                                                              - 2 * (M(_[0]*_[1])*M(_[2])*M(_[3]) + M(_[0]*_[2])*M(_[1])*M(_[3]) + M(_[0]*_[3])*M(_[2])*M(_[1]) \
                                                                     + M(_[1]*_[2])*M(_[0])*M(_[3]) + M(_[1]*_[3])*M(_[0])*M(_[2]) + M(_[2]*_[3])*M(_[0])*M(_[1])) \
                                                                  + 6 * M(_[0])*M(_[1])*M(_[2])*M(_[3]))
                elif scheme == "gamma":
                    raise ValueError('Cannot compute gamma closure for moments of 4th order. Please use normal or lognormal closure instead.')
                elif scheme == "lognormal":
                    closed_moment_eq = closed_moment_eq.subs(M(reduce(mul, _)),
                                                 M(_[0]*_[1]*_[2])*M(_[0]*_[2]*_[3])*M(_[0]*_[1]*_[3])*M(_[3]*_[1]*_[2])*M(_[0])*M(_[1])*M(_[2])*M(_[3]) \
                                                     / (M(_[0]*_[1])*M(_[0]*_[2])*M(_[0]*_[3])*M(_[1]*_[2])*M(_[1]*_[3])*M(_[2]*_[3])))
    
            else: # max_order not in [2,3]:
                raise ValueError('order should be 2 or 3.')

    return closed_moment_eq.simplify()

def closeMoments(conditional_moments, scheme = "gamma", order = 2):
    '''
    applies moment closure schemes to given conditional moment equations
    in order to close the system of differential equations.
    
    3rd order moments can be approximated using "normal",
    "gamma", or "lognormal" closure scheme. 
    4th order moments can approximated using "normal"
    or "lognormal" closure scheme.
    Closure for higher moments and other approximation schemes are not 
    implemented yet.

    Parameters
    ----------
    conditional_moments : dict
        dictionary which contains as keys the moments and as values their 
        respective differential conditional moment equation.
        {moment : moment equation}.
    scheme : str, optional
        moment closure approximation scheme to apply. 
        Possible schemes are: "normal", "gamma", "lognormal".
        The default is "gamma".
    order : int, optional
        maximum order of moments to occur in the moment equations. 
        All higher order moments will be approximated.
        The default is 2. (Third order moments will be approximated.)


    Returns
    -------
    closed_moments : dict
        dictionary which contains as keys the moments and as values their 
        respective approximated conditional moment equation with moments of
        max_order and lower order only.
        {moment : closed conditional moment equation}

    '''
    closed_moments = conditional_moments.copy()
    for m, moment_eq in closed_moments.items():
        closed_moments[m] = closeMomentEq(moment_eq, max_order = order, scheme = scheme)   
    return closed_moments

# %% 8) output / convert symbolic to numeric equations

def sympyExprToDiffEq(dM, closed_moments, stoich_net):
    '''
    converts closed_moments from closeMoments() into expressions
    that can be used for, e.g., numerical integration with scipy package.
    More specifically, the deterministic (time-dependent) parts of the 
    conditional moment equations are separated from the stochastic (reaction-
    dependent) terms. Those terms can then be transformed to lambda functions
    using sympy.lambdify().

    Parameters
    ----------
    dM : list of symbols
        list of moments that the conditional moment equations are generated for.
    closed_moments : dict
        dictionary which contains as keys the moments and as values their 
        respective approximated conditional moment equation with moments of
        1st and 2nd order only.
        {moment : closed conditional moment equation}
    stoich_net : sympy.Matrix()
        net stoichiometry matrix of reaction system.

    Returns
    -------
    ODEs : list of sympy expressions
        every entry corresponds to the time-dependent differential part of a 
        conditional moment equation.
    mapping : list of tuples
        list of substitutions (which moment was mapped to which variable).
    SDE_exprs : list of lists of sympy expressions
        list of stochastic contributions of every reaction channel and every 
        moment.

    '''   
    # create substitution list for time-dependent parts of moment equations (mapping)
    M = symbols('M', cls=Function)
    _1 = [M(m) for m in dM] 
    _2 = [symbols('M{}'.format(i)) for i in range(len(dM))]
    mapping = [(symbols('dt'), 1)] + list(zip(_1, _2))
    
    ODEs = [sum([i for i in closed_moments[m].expand().as_ordered_terms() if i.has(symbols('dt'))]) for m in dM]
    ODEs = [expr.subs(mapping) for expr in ODEs]
    
    # split stochastic contributions into contributions
    # of each reaction for each moment equation 
    SDEs = [sum([i for i in closed_moments[m].expand().as_ordered_terms() if not i.has(symbols('dt'))]) for m in dM]    
    SDEs = [expr.subs(mapping[1:]) if not expr == 0 else 0 for expr in SDEs]
    nreactions = np.shape(stoich_net)[1]
    
    R = ["dR{}".format(_) for _ in range(1,nreactions+1)]
    R = [symbols(r) for r in R]
        
    SDE_exprs = [[] for i in range(nreactions)]
    for j,m in enumerate(dM):
        for i in range(0,nreactions):
            if SDEs[j] != 0:
                SDE_exprs[i].append(sum([k for k in SDEs[j].as_ordered_terms() if k.has(R[i])]))
    
    # substitute dR terms with 1 in every nested entry of the SDE expressions
    SDE_exprs = [[_.subs(R[reac],1) if not type(_) == int else _ for _ in expr] for reac,expr in enumerate(SDE_exprs)]

    return ODEs, mapping[1:], SDE_exprs


# %% wrap everything together

def writeToTxt(file, sys_dict, scheme, order):
    with open(file + '.txt', 'w') as f:
        f.write('species: ' + str(sys_dict['species']) + '\n\n')
        f.write('selected species: ' + str([sys_dict['species'][i] for i in sys_dict['idx_select']]) + '\n\n')
        f.write('reaction rates: ' + str(sys_dict['rates']) + '\n\n')
        f.write('reactant stoichiometry:\n' + str(np.array(sys_dict['stoich_reac'])) + '\n\n')
        f.write('product stoichiometry:\n' + str(np.array(sys_dict['stoich_prod'])) + '\n\n')
        f.write('net stoichiometry:\n' + str(np.array(sys_dict['stoich_net'])) + '\n\n')
        f.write('filter equation:\ndpi(' + str([sys_dict['species'][i] for i in sys_dict['idx_latent']]) + ', t) = ' + str(sys_dict['filterEq']) + '\n\n')
        f.write('conditional moment equations:\n')
        for m in sys_dict['dM']:
            f.write('dM(' + str(m) + ') = ' + str(sys_dict['conditional_moments'][m]) + '\n')
        f.write('\nclosed conditional moment equations:\n')
        f.write('(equations were closed with ' + scheme + ' closure scheme up to order ' + str(order) + '.)\n')
        for m in sys_dict['dM']:
            f.write('dM(' + str(m) + ') = ' + str(sys_dict['closed_moments'][m]) + '\n')
    return

def generateEquations(stoich_reac, stoich_prod, select, file=None, X=None, rates=None, order=2, scheme="gamma"):
    '''
    generates conditional moment equations and closed conditional moment 
    equations from given reaction network stoichiometry. If a filename is 
    specified, the equations will be saved to a <file>.txt and pickled to 
    restore or share them.

    Parameters
    ----------
    stoich_reac : sympy.Matrix()
        reaction stoichiometry of the reaction system.
    stoich_prod : sympy.Matrix()
        product stoichiometry of the reaction system.
    select : str or list of str
        one or several species of the system that is/are known or
        observed. They are directly simulated, whereas moment
        equations conditional on these selected_species are computed for all 
        other species. Input select like this: 'C' or ['A','C']
    file : str, optional
        filename to save the reaction system and generated equations to. If 
        not specified, the system and equations will not be saved. If 
        specified, a "human readable" txt file with the important information
        about the system and the conditional moment equations, as well as file 
        for later restoring the symbolic / numeric equations, is generated.
        The default is None.
    X : list of strings, optional
        names of species. The default is None.
    rates : list of strings, optional
        names of chemical reaction rates. The default is None.
    order : int, optional
        order up to which the conditional moment equations for the latent
        species will be generated.
        The default is 2.
    scheme : str, optional
        moment closure approximation scheme to apply. 
        Possible schemes are: "normal", "gamma", "lognormal".
        The default is "gamma".

    Returns
    -------
    sys_dict. dict
        Dictionary with the following information:
        {
        'species': X,
        'rates': rates,
        'stoich_net': stoich_net,
        'stoich_reac': stoich_reac,
        'stoich_prod': stoich_prod,
        'propensities': h,
        'idx_select': idx_select,
        'idx_latent': idx_latent,
        'RZ': RZ,
        'RL': RL,
        'RO': RO,
        'filterEq': filterEq,
        'dM': dM,
        'conditional_moments': conditional_moments,
        'closed_moments': closed_moments,
        'det_parts': det_part,
        'stoch_parts': stoch_part,
        'mapping': mapping
        }
    '''
   
    if X:
        if rates:
            (X, rates, stoich_net) = reactionSystem(stoich_reac, stoich_prod, X=X, rates=rates)
        (X, rates, stoich_net) = reactionSystem(stoich_reac, stoich_prod, X=X)
    elif rates:
        (X, rates, stoich_net) = reactionSystem(stoich_reac, stoich_prod, rates=rates)
    else:
        (X, rates, stoich_net) = reactionSystem(stoich_reac, stoich_prod)
    h = propensities(X, rates, stoich_reac)
    idx_select, idx_latent, RX, RL, RO, RH, RC, RZ = splitNetwork(select, X, stoich_net, stoich_reac, stoich_prod)
    filterEqT, filterEqR = computeFilterEquation(X, rates, stoich_reac, stoich_net, RO, RL, RZ, idx_latent, idx_select)
    filterEq = filterEqT + filterEqR
    dM, conditional_moments = generateConditionalMomentEquations(X, idx_latent, filterEq, order = order)
    closed_moments = closeMoments(conditional_moments, scheme = scheme, order = order)
    det_part, mapping, stoch_part = sympyExprToDiffEq(dM, closed_moments, stoich_net)
    
    sys_dict = {
        'species': X,
        'rates': rates,
        'stoich_net': stoich_net,
        'stoich_reac': stoich_reac,
        'stoich_prod': stoich_prod,
        'propensities': h,
        'idx_select': idx_select,
        'idx_latent': idx_latent,
        'RX': RX,
        'RL': RL,
        'RO': RO,
        'RH': RH,
        'RC': RC,
        'RZ': RZ,
        'filterEq': filterEq,
        'dM': dM,
        'conditional_moments': conditional_moments,
        'closed_moments': closed_moments,
        'det_parts': det_part,
        'stoch_parts': stoch_part,
        'mapping': mapping
        }
    
    # save to txt and dill file
    if file:
        writeToTxt(file, sys_dict, scheme, order)
        
        global M, pi
        M = symbols('M', cls=Function) # needs to be redefined in order to pickle/dill correctly
        pi = symbols('pi', cls=Function)
        
        with open(file, "wb") as f:
            dill.dump(sys_dict, f)
    
    return sys_dict


def loadAndLambdify(file):
    '''
    restores all information of the reaction system and then generated 
    conditional moment equations from a file generated with dill package.

    Parameters
    ----------
    file : dill file
        file generated with dill package (e.g. through the generateEquations() 
        function).

    Returns
    -------
    my_system_loaded : dict
        Restores the dictionary from file with the following information:
        {
        'species': X,
        'rates': rates,
        'stoich_net': stoich_net,
        'stoich_reac': stoich_reac,
        'stoich_prod': stoich_prod,
        'propensities': h,
        'idx_select': idx_select,
        'idx_latent': idx_latent,
        'RX': RX,
        'RL': RL,
        'RO': RO,
        'RH': RH,
        'RC': RC,
        'RZ': RZ,
        'filterEq': filterEq,
        'dM': dM,
        'conditional_moments': conditional_moments,
        'closed_moments': closed_moments,
        'det_parts': det_part,
        'stoch_parts': stoch_part,
        'mapping': mapping
        }
    det_func : lambdified function
        lambdified function of the deterministic parts of the conditional
        moment equations. It can be used with numerical packages like numpy
        or scipy.
        arguments: y, t, [X_select], [rates], [moments]
    stoch_func : lambdified function
        lambdified function of the stochastic parts of the conditional
        moment equations. It can be used with numerical packages like numpy
        or scipy.
        arguments: [X_select], [rates], [moments]
    '''
    # load file
    global M, pi
    M = symbols('M', cls=Function) # needs to be redefined in order to pickle/dill correctly
    pi = symbols('pi', cls=Function)
    my_system_loaded = dill.load(open(file, "rb"))
    
    # lambdify what's needed
    moment_names = [m for _,m in my_system_loaded['mapping']] # list of moment names (symbolic)
    X_select = [my_system_loaded['species'][i] for i in my_system_loaded['idx_select']]
    det_func = lambdify([symbols('y'), symbols('t'), X_select, my_system_loaded['rates'], moment_names], my_system_loaded['det_parts'])
    stoch_func = lambdify([X_select, my_system_loaded['rates'], moment_names], my_system_loaded['stoch_parts'])

    return my_system_loaded, det_func, stoch_func