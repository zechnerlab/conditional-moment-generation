# plotting for snSSA and SSA

import matplotlib.pyplot  as  plt
import numpy as np
import scipy.stats
import random

# %% load data
file = 'bistable_network_snSSA_'
X = np.load(file + 'X.npy')
M = np.load(file + 'M.npy')

X_SSA = np.load(file + 'bistable_network_SSA_X.npy')

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

a = [w for w in G1i if w in G2i]
b = [w for w in G1a if w in G2i]
c = [w for w in G1i if w in G2a]
d = [w for w in G1a if w in G2a]
SSA_bs = np.zeros((10000))
for i in a:
    SSA_bs[i] = 1 # case 00
for i in b:
    SSA_bs[i] = 2 # case 10
for i in c:
    SSA_bs[i] = 3 # case 01
for i in d:
    SSA_bs[i] = 4 # case 11

# bootstrapping for SSA standard errors
n_bs = 1000
size_bs = len(SSA_bs)
SSA_bs_a = np.zeros((n_bs))
SSA_bs_b = np.zeros((n_bs))
SSA_bs_c = np.zeros((n_bs))
SSA_bs_d = np.zeros((n_bs))

for i in range(n_bs):
    boot = random.choices(SSA_bs, k = size_bs)
    SSA_bs_a[i] = boot.count(1)/size_bs
    SSA_bs_b[i] = boot.count(2)/size_bs
    SSA_bs_c[i] = boot.count(3)/size_bs
    SSA_bs_d[i] = boot.count(4)/size_bs

# compute means and SEs
SSAcases = [np.mean(SSA_bs_a),np.mean(SSA_bs_b),np.mean(SSA_bs_c),np.mean(SSA_bs_d)]
SSAerror = [np.std(SSA_bs_a),np.std(SSA_bs_b),np.std(SSA_bs_c),np.std(SSA_bs_d)]

# snSSA data
G1 = list(X[:,-3,0])
G2 = list(X[:,-3,2])
n = len(G2)
G1a = [i for i,e in enumerate(G1) if e == 1]
G2a = [i for i,e in enumerate(G2) if e == 1]
G1i = [i for i,e in enumerate(G1) if e == 0]
G2i = [i for i,e in enumerate(G2) if e == 0]
a = [w for w in G1i if w in G2i]
b = [w for w in G1a if w in G2i]
c = [w for w in G1i if w in G2a]
d = [w for w in G1a if w in G2a]
snSSA_bs = np.zeros((10000))
for i in a:
    snSSA_bs[i] = 1 # case 00
for i in b:
    snSSA_bs[i] = 2 # case 10
for i in c:
    snSSA_bs[i] = 3 # case 01
for i in d:
    snSSA_bs[i] = 4 # case 11

# bootstrapping for snSSA standard errors

n_bs = 1000
size_bs = len(SSA_bs)
snSSA_bs_a = np.zeros((n_bs))
snSSA_bs_b = np.zeros((n_bs))
snSSA_bs_c = np.zeros((n_bs))
snSSA_bs_d = np.zeros((n_bs))

for i in range(n_bs):
    boot = random.choices(snSSA_bs, k = size_bs)
    snSSA_bs_a[i] = boot.count(1)/size_bs
    snSSA_bs_b[i] = boot.count(2)/size_bs
    snSSA_bs_c[i] = boot.count(3)/size_bs
    snSSA_bs_d[i] = boot.count(4)/size_bs

# compute means and SEs
snSSAcases = [np.mean(snSSA_bs_a),np.mean(snSSA_bs_b),np.mean(snSSA_bs_c),np.mean(snSSA_bs_d)]
snSSAerror = [np.std(snSSA_bs_a),np.std(snSSA_bs_b),np.std(snSSA_bs_c),np.std(snSSA_bs_d)]

# =============================================================================
# # %% first plot
# barWidth = 0.4
# r1 = np.arange(len(snSSAcases)) # x positions of bars
# r2 = [x + barWidth for x in r1]
# 
# plt.bar(r1, snSSAcases, width = barWidth, color = 'grey', label='snSSA',
#         yerr = snSSAerror, align = 'center', ecolor = 'black', capsize = 7.5)
# plt.bar(r2, SSAcases, width = barWidth, color = 'lightgrey', label='SSA',
#         yerr = SSAerror, align = 'center', ecolor = 'black', capsize = 7.5)
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
ax1.bar(r1, snSSAcases, width = barWidth, color = 'grey', label='snSSA',
        yerr = snSSAerror, align = 'center', ecolor = 'black', capsize = 5)
ax1.bar(r2, SSAcases, width = barWidth, color = 'lightgrey', label='SSA',
        yerr = SSAerror, align = 'center', ecolor = 'black', capsize = 5)
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
fig.set_size_inches(10, 3.5, forward=True)
fig.tight_layout()
#fig.savefig('snSSA_SSA.eps', format = 'eps', bbox_inches="tight")