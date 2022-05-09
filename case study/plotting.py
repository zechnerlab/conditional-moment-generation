# plotting for snSSA and SSA

import matplotlib.pyplot  as  plt
import numpy as np
import scipy.stats
import random

# %% load data
file = 'bistable_network_snSSA_'
X = np.load(file + "X.npy")
M = np.load(file + 'M.npy')

file = 'bistable_network_SSA_'
X_SSA = np.load(file + "X.npy")

#file = 'SSA/bistable_network_'
#reacs = np.load(file + 'reactions.npy')
#times = np.load(file + 'times.npy')

# %% data prep for first plot
#species = ['G_1a', 'G_1i', 'G_2a', 'G_2i', 'R_1', 'R_2', 'P_1', 'P_2']

# SSA data
G1 = list(X_SSA[:,-3,0])
G2 = list(X_SSA[:,-3,2])
n = len(G2)
G1a = [i for i,e in enumerate(G1) if e == 1]
G2a = [i for i,e in enumerate(G2) if e == 1]
G1i = [i for i,e in enumerate(G1) if e == 0]
G2i = [i for i,e in enumerate(G2) if e == 0]
case00 = len([w for w in G1i if w in G2i])/n
case10 = len([w for w in G1a if w in G2i])/n
case01 = len([w for w in G1i if w in G2a])/n
case11 = len([w for w in G1a if w in G2a])/n
SSAcases = [case00,case10,case01,case11]

# snSSA data
G1 = list(X[:,-3,0])
G2 = list(X[:,-3,2])
n = len(G2)
G1a = [i for i,e in enumerate(G1) if e == 1]
G2a = [i for i,e in enumerate(G2) if e == 1]
G1i = [i for i,e in enumerate(G1) if e == 0]
G2i = [i for i,e in enumerate(G2) if e == 0]
case00 = len([w for w in G1i if w in G2i])/n
case10 = len([w for w in G1a if w in G2i])/n
case01 = len([w for w in G1i if w in G2a])/n
case11 = len([w for w in G1a if w in G2a])/n
snSSAcases = [case00,case10,case01,case11]

# =============================================================================
# # %% first plot
# barWidth = 0.4
# r1 = np.arange(len(snSSAcases)) # x positions of bars
# r2 = [x + barWidth for x in r1]
# 
# plt.bar(r1, snSSAcases, width = barWidth, color = 'grey', label='snSSA')
# plt.bar(r2, SSAcases, width = barWidth, color = 'lightgrey', label='SSA')
#  
# plt.xticks([r + barWidth/2 for r in range(len(snSSAcases))], ['(0,0)', '(1,0)', '(0,1)', '(1,1)'])
# plt.ylabel('$\mathcal{P}\,(G_1^*,G_2^*,t=3000)$')
# plt.xlabel('$(\,G_1^*,G_2^*)$')
# plt.legend()
# plt.show()
# =============================================================================

# %% data prep for 2nd plot
x = np.linspace(0,16000,100) # values of P1 to evaluate estimated density at
mu = M[:,-3,2] # means as mu[i] = E[P1|Xi]
var = [np.abs(v) for v in M[:,-3,-3]-[m**2 for m in M[:,-3,2]]] # variances as var[i] = E[P²|Xi]-E[P|Xi]²
std = [np.sqrt(v) for v in var]

# estimate density of P1 as overlay of N=100 Gaussians
idx = [random.randint(0,len(mu)-1) for n in range(100)] # 100 random indices
y100 = [sum(scipy.stats.norm(mu[i],std[i]).pdf(xi) for i in idx)/(float(100)) for xi in x]

idx = [random.randint(0,len(mu)-1) for n in range(1000)] # 1000 random indices)
y1000 = [sum(scipy.stats.norm(mu[i],std[i]).pdf(xi) for i in idx)/(float(1000)) for xi in x]

# =============================================================================
# # %% 2nd plot
# plt.plot(x,y100,label = 'snSSA ($n=100$)', color = 'grey', ls='--')
# plt.plot(x,y1000,label = 'snSSA ($n=1000$)', color = 'grey')
# plt.hist(X_SSA[:,-3,-2], bins = 64, range = (0,16000), density = True, color = 'lightgrey', label = 'SSA')
# plt.xlabel('$P_1$')
# plt.ylabel('$\mathcal{P}\,(P_1,t=3000)$')
# plt.ylim(0,0.0015)
# plt.xticks(np.arange(0, 16000, step=5000))
# plt.yticks(np.arange(0,0.0016, step=0.0005))
# plt.legend(frameon=False, loc='upper right')
# =============================================================================

# %% plot together
plt.rcParams['font.size'] = '12'
fig, (ax1, ax2) = plt.subplots(1, 2)

# left plot
barWidth = 0.3
r1 = np.arange(len(snSSAcases)) # x positions of bars
r2 = [x + barWidth for x in r1]
ax1.bar(r1, snSSAcases, width = barWidth, color = 'grey', label='snSSA')
ax1.bar(r2, SSAcases, width = barWidth, color = 'lightgrey', label='SSA')
ax1.set_xticks([r + barWidth/2 for r in range(len(snSSAcases))])
ax1.set_xticklabels(['(0,0)', '(1,0)', '(0,1)', '(1,1)'])
ax1.set(xlabel='$(\,G_1^*,G_2^*)$', ylabel='$\mathcal{P}\,(G_1^*,G_2^*,t=3000)$')
ax1.legend(frameon=False, loc='upper right')

# right plot
ax2.plot(x,y100,label = 'snSSA ($n=100$)', color = 'grey', ls='--')
ax2.plot(x,y1000,label = 'snSSA ($n=1000$)', color = 'grey')
ax2.hist(X_SSA[:,-3,-2], bins = 64, range = (0,16000), density = True, color = 'lightgrey', label = 'SSA')
ax2.set(xlabel='$P_1$', ylabel='$\mathcal{P}\,(P_1,t=3000)$')
ax2.set(ylim=(0,0.0015), yticks=(np.arange(0,0.0016, step=0.0005)))
ax2.legend(frameon=False, loc='upper right')

# general layout
fig.set_size_inches(10, 3.5, forward=True) # 10,4
fig.tight_layout()
#fig.savefig('snSSA_SSA.eps', format = 'eps', bbox_inches="tight")