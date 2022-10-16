
import numpy as np
import cvxpy as cp
import os
import matplotlib.pyplot as plt
from numpy.core.multiarray import _vec_string

class BaumWelch():
	"""Class which performs the Baum-Welch algorithm and that uses it to solve the link prediction and the collaborative filtering problems."""
	def __init__(self):
		pass

	def forward(self, ini, P, O, m, n, delta, Y, K=None):
		"""
		Forward step of the BaumWelch algorithm.
		We observe the connections between nodes 0,...,m,n,...,n+delta. However, we consider that the edges involving
		nodes between m+1 and n-1 are not reliable and we do not take into account the estimated clusters for the nodes 
		between m+1 and n-1. 

		:param ini: initial distribution of the Markov chain
		:param P: Transition matrix of the Markov chain
		:param O: matrix of emission probabilities
		:param m: We observe fully the graph until time m
		:param n: We do not observe reliably the connection involving nodes between time m and n
		:param delta: We observe the connections between nodes n and n+delta
		:param Y: vector of length n+delta+1 of the estimated communities
		:param K: Number of communities

		.. note: 
		Using this function with a specified K can be typically interesting when we try to 
		estimate the number of clusters with the procedure describes in the paper.
		"""
		if K is None:
			K=self.K
		alpha = np.ones((K, n+delta+1))
		alpha[:,0] =  ini * O[:,Y[0]]
		# scaling 
		alpha[:,0] /= np.sum(alpha[:,0])
		puissP = P
		for t in range(1,n+delta+1):
			for i in range(K):
				if t <= m or t >= n:
					alpha[i,t] = np.sum(alpha[:,t-1] * P[:,i]) * O[i,Y[t]]
				else:
					alpha[i,t] = np.sum(alpha[:,m] * puissP[:,i])
					puissP = puissP @ P
			# scaling 
			# alpha[:,t] /= np.sum(alpha[:,t])
		return alpha

	def backward(self, P, O, m, n, delta, Y, K=None):
		"""
		Backward step of the BaumWelch algorithm.
		We observe the connections between nodes 0,...,m,n,...,n+delta. However, we consider that the edges involving
		nodes between m+1 and n-1 are not reliable and we do not take into account the estimated clusters for the nodes 
		between m+1 and n-1. 

		:param P: Transition matrix of the Markov chain
		:param O: matrix of emission probabilities
		:param m: We observe fully the graph until time m
		:param n: We do not observe reliably the connection involving nodes between time m and n
		:param delta: We observe the connections between nodes n and n+delta
		:param Y: vector of length n+delta+1 of the estimated communities
		:param K: Number of communities

		.. note: 
		Using this function with a specified K can be typically interesting when we try to 
		estimate the number of clusters with the procedure describes in the paper.
		"""
		if K is None:
			K=self.K
		beta = np.ones((K, n+delta+1))
		beta[:,n+delta] =  np.ones(K)
		# scaling 
		# beta[:,n+delta] /= np.sum(beta[:,n+delta])
		puissP = P
		for t in range(n+delta-1,-1,-1):
			for i in range(K):
				if t <= m-1 or t >= n-1:
					beta[i,t] = np.sum(beta[:,t+1] * P[i,:] * O[:,Y[t+1]]) 
				else:
					beta[i,t] = np.sum(beta[:,n-1] * puissP[i,:]) 
					puissP = puissP @ P
			# scaling 
			# beta[:,t] /= np.sum(beta[:,t])
		return beta

	def update(self, alpha, beta, P, O, m, n, delta, Y, K=None):
		"""
		Update step of the BaumWelch algorithm: the parameters of the HMM are updated and returned.

		:param alpha: Matrix of size $K \times n+delta$ giving the output of the "forward" method 
		:param beta: Matrix of size $K \times n+delta$ giving the output of the "backward" method
		:param P: Transition matrix of the Markov chain
		:param O: matrix of emission probabilities 
		:param m: We observe fully the graph until time m
		:param n: We do not observe reliably the connection involving nodes between time m and n
		:param delta: We observe the connections between nodes n and n+delta
		:param Y: vector of length n+delta+1 of the estimated communities
		:param K: Number of communities

		.. note: 
		Using this function with a specified K can be typically interesting when we try to 
		estimate the number of clusters with the procedure describes in the paper.
		"""
		if K is None:
			K=self.K
		xi = np.zeros((K,K, n+delta+1))
		for t in range(n+delta):
			for i in range(K):
				for j in range(K):
					if t >= m-1 and t <= n-1:
						xi[i,j,t] = alpha[i,t] * P[i,j] * beta[j,t+1]
					else:
						xi[i,j,t] = alpha[i,t] * P[i,j] * O[j,Y[t+1]] * beta[j,t+1]
			xi[:,:,t] /= np.sum(xi[:,:,t])
		P = np.zeros((K,K))
		O = np.zeros((K,K))
		gamma = alpha * beta
		gamma /= np.tile(np.sum(gamma, axis=0).reshape(1,n+delta+1), (K,1))
		ini = gamma[:,0]
		for i in range(K):
			for j in range(K):
				P[i,j] = np.sum(xi[i,j,:]) / np.sum(gamma[i,:])
		for i in range(K):
			for t in range(gamma.shape[1]):
				O[i,Y[t]] += gamma[i,t]
			O[i,:] /= np.sum(gamma[i,:])
		return (ini, gamma, P, O)

	def adjacency_BaumWelch(self, m, n, delta):
		"""
		Builds an adjacency matrix from the attribute self.X removing the nodes between m+1 and n-1 (that we consider to be not reliable).

		:param m: We observe fully the graph until time m
		:param n: We do not observe reliably how nodes between m+1 and n-1 are connected
		:param delta: We observe reliably how nodes between n and n+delta are connected
		"""
		X = np.zeros((m+1+delta+1,m+1+delta+1))
		for i in range(m+1):
			X[i,:m+1] = self.X[i,:m+1]
			X[:m+1,i] = self.X[i,:m+1]
		for i in range(delta+1):  
			X[m+i+1,m+1:] = self.X[n+i,n:n+delta+1]
			X[m+1:,m+i+1] = self.X[n:n+delta+1,n+i]
			X[m+i+1,:m+1] = self.X[n+i,:m+1]
			X[:m+1,m+i+1] = self.X[:m+1,n+i]
		return X

	def BaumWelch(self, m, n, delta, nbite, eps = 1e-2, K=None):
		"""
		BaumWelch algorithm that iteratively perform the forward, backward and update steps.

		:param m: We observe fully the graph until time m
		:param n: We do not observe reliably the connection involving nodes between time m and n
		:param delta: We observe the connections between nodes n and n+delta
		:param nbite: Number of iterations for the Baum Welch algorithm
		:param eps: Parameter use to initialize the matrix of emission probabilities O
		:param K: Number of communities

		.. note: 
		Using this function with a specified K can be typically interesting when we try to 
		estimate the number of clusters with the procedure describes in the paper.
		"""
		if K is None:
			K=self.K
		self.estimate_partition()
		Y = np.zeros(n+delta+1)
		for i in range(m+1):
			Y[i] = self.clusters_approx[i]
		count = m+1
		for i in range(n,n+delta+1):		
			Y[i] = self.clusters_approx[count]
			count += 1
		Y = np.array(Y, dtype=int)
		ini = np.ones(K) / K
		P = np.ones((K,K)) / K
		O = (1-eps)*np.eye(K) + (eps / (K-1))* (np.ones((K,K))-np.eye(K))
		for ite in range(nbite):
			alpha = self.forward(ini, P, O, m, n, delta, Y)
			beta = self.backward(P, O, m, n, delta, Y)
			ini, gamma, P, O = self.update(alpha, beta, P, O, m, n, delta, Y)
		return ini, gamma, P, O

	def collaborative_filtering_robustMAP(self, ini, alpha, beta, O, observed_links, observed_nodes, m, n, Pbaum=None):
		"""
		Solve robustly the collaborative filtering problem when we observe fully the graph at time m and we want 
		to predict the community of node n when we observe only a subset of the edges that connects (or not) n
		with the nodes 1,...,m. 

		:param ini: initial distribution of the Markov chain given by the Baum-Welch algorithm
		:param alpha: Matrix of size $K \times n+delta$ given by the Baum-Welch algorithm
		:param beta: Matrix of size $K \times n+delta$ given by the Baum-Welch algorithm
		:param observed_links: A vector with binary variables of length less than m. For any i, observed_links[i] is 1 if and only if we observe an edge between nodes n and observed_nodes[i] (and 0 otherwise).
		:param observed_nodes: A vector with the same length as the vector "observed_links". It contains the nodes for which we observe the connection (or not) with node n
		:param m: We observe fully the graph until time m
		:param n: Node that we want to learn the community

		.. note: 
		Our implementation do not respect striclty the formula of the paper. We get rid of quantities that do not depend on
		the cluster k of node n. This allows to avoid underflow issues.
		"""
		if Pbaum is None:
			Pbaum = self.approx_P
		best_pred = 0
		best_k = -1
		indices = np.argsort(observed_nodes)[::-1]
		observed_nodes = observed_nodes[indices]
		observed_links = observed_links[indices]
		ROB = []
		for k in range(self.K): # c_{n}
			res =  beta[:,observed_nodes[0]] / np.sum(beta[:,observed_nodes[0]])
			for c in range(self.K):  # c_i
				if observed_links[0]:
					res[c] *= self.approx_Q[c,k]
				else:
					res[c] *= 1-self.approx_Q[c,k]
			for i,node in enumerate(observed_nodes[:int(len(observed_links)-1)]):
				chi = np.ones((self.K, self.K))
				nodei = node
				nodeim = observed_nodes[i+1]
				for c in range(self.K): 
					for cb in range(self.K):
						chi[c,cb] = Pbaum[c,cb] * O[cb,self.clusters_approx[nodeim+1]]
				if (nodeim+1) != nodei:
					for step in range(nodeim+2,nodei+1):
						chitemp = np.zeros((self.K, self.K))
						for c in range(self.K): 
							for cb in range(self.K):
								for cd in range(self.K):
									chitemp[c,cb] += chi[c,cd] * Pbaum[cd,cb] * O[cb,self.clusters_approx[step]]
						chi = np.copy(chitemp)
				restemp = np.zeros(self.K)
				for c in range(self.K):
					restemp[c] = np.sum(chi[c,:] * res)
				if observed_links[i+1]:
					res = self.approx_Q[:,k] * restemp
				else:
					res = (1-self.approx_Q[:,k]) * restemp 
			final_pred = np.sum(alpha[:,observed_nodes[-1]] * res)
			vec = [np.sum(ini * (np.linalg.matrix_power(Pbaum,n)[:,r])) for r in range(self.K)]
			vec /= np.sum(vec)
			final_pred *= vec[k]
			ROB.append(final_pred)
		ROB /= np.sum(ROB)
		return np.argmax(ROB)

	def collaborative_filtering_pluginMAP(self, alpha, beta, observed_links, observed_nodes, m, n):
		"""
		Solve the collaborative filtering problem using the plugin approach when we observe fully the graph at
		time m and we want to predict the community of node n when we observe only a subset of the edges that 
		connects (or not) n with the nodes 1,...,m. 

		:param alpha: Matrix of size $K \times n+delta$ given by the Baum-Welch algorithm
		:param beta: Matrix of size $K \times n+delta$ given by the Baum-Welch algorithm
		:param observed_links: A vector with binary variables of length less than m. For any i, observed_links[i] is 1 if and only if we observe an edge between nodes n and observed_nodes[i] (and 0 otherwise).
		:param observed_nodes: A vector with the same length as the vector "observed_links". It contains the nodes for which we observe the connection (or not) with node n
		:param m: We observe fully the graph until time m
		:param n: Node that we want to learn the community
		"""
		best_pred = 0
		best_k = -1
		for k in range(self.K):
			temp = np.linalg.matrix_power(self.approx_P,n-m)[self.clusters_approx[m],k]
			for i,node in enumerate(observed_nodes):
				if observed_links[i]:
					temp *= self.approx_Q[self.clusters_approx[node],k]
				else:
					temp *= 1-self.approx_Q[self.clusters_approx[node],k]
			if temp > best_pred:
				best_pred = temp
				best_k = k
		return best_k

	def collaborative_filtering_optimalMAP(self, alpha, beta, observed_links, observed_nodes, m, n):
		"""Solve robustly the collaborative filtering problem using the optimal approach when we observe fully 
		the graph at time m and we want to predict the community of node n when we observe only a subset of the 
		edges that connects (or not) n with the nodes 1,...,m. 

		:param alpha: Matrix of size $K \times n+delta$ given by the Baum-Welch algorithm
		:param beta: Matrix of size $K \times n+delta$ given by the Baum-Welch algorithm
		:param observed_links: A vector with binary variables of length less than m. For any i, observed_links[i] is 1 if and only if we observe an edge between nodes n and observed_nodes[i] (and 0 otherwise).
		:param observed_nodes: A vector with the same length as the vector "observed_links". It contains the nodes for which we observe the connection (or not) with node n
		:param m: We observe fully the graph until time m
		:param n: Node that we want to learn the community
		"""
		best_pred = 0
		best_k = -1
		for k in range(self.K):
			temp = np.linalg.matrix_power(self.P,n-m)[self.clusters[m],k]
			for i,node in enumerate(observed_nodes):
				if observed_links[i]:
					temp *= self.Q[self.clusters[node],k]
				else:
					temp *= 1-self.Q[self.clusters[node],k]
			if temp > best_pred:
				best_pred = temp
				best_k = k
		return best_k


	def RES(self, ini, alpha, beta, O ,observed_links, observed_nodes, m, n):
		"""
		Solve robustly the collaborative filtering problem when we observe fully the graph at time m and we want 
		to predict the community of node n when we observe only a subset of the edges that connects (or not) n
		with the nodes 1,...,m. 

		:param ini: initial distribution of the Markov chain given by the Baum-Welch algorithm
		:param alpha: Matrix of size $K \times n+delta$ given by the Baum-Welch algorithm
		:param beta: Matrix of size $K \times n+delta$ given by the Baum-Welch algorithm
		:param observed_links: A vector with binary variables of length less than m. For any i, observed_links[i] is 1 if and only if we observe an edge between nodes n and observed_nodes[i] (and 0 otherwise).
		:param observed_nodes: A vector with the same length as the vector "observed_links". It contains the nodes for which we observe the connection (or not) with node n
		:param m: We observe fully the graph until time m
		:param n: Node that we want to learn the community

		.. note: 
		Our implementation do not respect striclty the formula of the paper. We get rid of quantities that do not depend on
		the cluster k of node n. This allows to avoid underflow issues.
		"""
		best_pred = 0
		best_k = -1
		indices = np.argsort(observed_nodes)[::-1]
		observed_nodes = observed_nodes[indices]
		observed_links = observed_links[indices]
		ROB = []
		for k in range(self.K):
			res =  np.ones(self.K)  # beta[:,observed_nodes[0]]
			for c in range(self.K):
				if observed_links[0]:
					res[c] *= self.approx_Q[c,k]
				else:
					res[c] *= 1-self.approx_Q[c,k]
			for i,node in enumerate(observed_nodes[:int(len(observed_links)-1)]):
				chi = np.zeros((self.K, self.K))
				nodei = node
				nodeim = observed_nodes[i+1]
				for c in range(self.K): 
					for cb in range(self.K):
						chi[c,cb] = self.approx_P[c,cb] * O[cb,self.clusters_approx[nodei]]
				for step in range(nodei-1,nodeim,-1):
					chitemp = np.zeros((self.K, self.K))
					for c in range(self.K): 
						for cb in range(self.K):
							for cd in range(self.K):
								chitemp[c,cb] += self.approx_P[c,cd] * ( O[cd,self.clusters_approx[step]] + chi[cd,cb])
					chi = np.copy(chitemp)
				restemp = np.zeros(self.K)
				for c in range(self.K):
					restemp[c] = np.sum(chi[c,:] * res)
				if observed_links[i+1]:
					res = self.approx_Q[:,k] * restemp
				else:
					res = (1-self.approx_Q[:,k]) * restemp 
			final_pred = np.sum(alpha[:,observed_nodes[-1]] * res)
			final_pred *= np.sum(ini * (np.linalg.matrix_power(self.approx_P,n))[:,k])
			ROB.append(final_pred)
		MAP = []
		for k in range(self.K):
			temp = np.linalg.matrix_power(self.approx_P,n-m)[self.clusters_approx[m],k]
			for i,node in enumerate(observed_nodes):
				if observed_links[i]:
					temp *= self.approx_Q[self.clusters_approx[node],k]
				else:
					temp *= 1-self.approx_Q[self.clusters_approx[node],k]
			MAP.append(temp)

		OPT = []
		for k in range(self.K):
			temp = np.linalg.matrix_power(self.P,n-m)[self.clusters[m],k]
			for i,node in enumerate(observed_nodes):
				if observed_links[i]:
					temp *= self.Q[self.clusters[node],k]
				else:
					temp *= 1-self.Q[self.clusters[node],k]
			OPT.append(temp)
		return ROB/np.sum(ROB), OPT/np.sum(OPT), MAP/np.sum(MAP)