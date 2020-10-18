import markovianSBM
from markovianSBM.SBM import SBM
import numpy as np
import scipy as sc
import os


list_n = [10*i for i in range(1,15)]
pvalue_iid = np.zeros(len(list_n))
pvalue_markov = np.zeros(len(list_n))

array = os.environ['SLURM_ARRAY_TASK_ID']
np.random.seed(int(array))
K = 4
P = np.array([[0.1, 0.3, 0.5,0.1],[0.45, 0.15, 0.2, 0.2],[0.15, 0.3, 0.1, 0.45 ],[0.25, 0.3, 0.1, 0.35 ]])
Q = np.array([[0.22, 0.48, 0.29, 0.44], [0.48, 0.61 ,0.18, 0.15],  [0.29 , 0.18 ,0.08 ,0.87],  [0.44 ,0.15 ,0.87, 0.27]])
gap = np.float('inf')
s,v = np.linalg.eig(P.T)
invmeas = np.zeros(K)
for i in range(len(s)):
	if abs(s[i]-1)<gap:
		gap = abs(s[i]-1)
		invmeas = v[:,i]

if invmeas[0]<0:
	invmeas *= -1
invmeas /= np.sum(invmeas)

Piid = np.tile(np.array(invmeas).reshape(1,-1),(K,1))

for ind,n in enumerate(list_n):
	print(ind)

	G = SBM(n, K, framework='markov', P=P, Q=Q)
	G.estimate_partition()
	G.estimate_effectifs()
	G.estimate_parameters()
	Giid = SBM(n, K, framework='markov', P=Piid, Q=Q)
	Giid.estimate_partition()
	Giid.estimate_effectifs()
	Giid.estimate_parameters()
	iidstat = -2 * np.sum(  Giid.approx_transitions * (np.log( Giid.approx_effectifs.reshape(-1,1) * Piid )   - np.log(Giid.approx_transitions)  ))

	piid = 1-sc.stats.chi2.cdf(iidstat,df=K**2-K)

	stat = -2 * np.sum(  G.approx_transitions * ( np.log( G.approx_effectifs.reshape(-1,1) * Piid )  - np.log(G.approx_transitions)  ))

	plat = 1-sc.stats.chi2.cdf(stat,df=K**2-K)
	pvalue_iid[ind] = piid
	pvalue_markov[ind] = plat
	np.save('pvalue_iid'+array+'_K4.npy',pvalue_iid)
	np.save('pvalue_markov'+array+'_K4.npy',pvalue_markov)
	np.save('approx_P'+array+'_n_'+str(n)+'_K4.npy',G.approx_P)
	np.save('approx_Q'+array+'_n_'+str(n)+'_K4.npy',G.approx_Q)
	np.save('clusters'+array+'_n_'+str(n)+'_K4.npy',G.clusters)
	np.save('clusters_approx'+array+'_n_'+str(n)+'_K4.npy',G.clusters_approx)