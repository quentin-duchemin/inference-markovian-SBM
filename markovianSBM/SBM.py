import numpy as np
import cvxpy as cp
import os
import matplotlib.pyplot as plt
from .Clustering import Clustering
from .Estimation import Estimation
from .RelaxedKmeans import RelaxedKmeans


class SBM(RelaxedKmeans, Clustering, Estimation, BaumWelch):
	"""Main class building the graph and running the algorithm to recover communities."""
	def __init__(self, n, K, ini_distribution='uniform', framework='iid', X=None, Q=None, P=None, save_B_matrices = False):
		RelaxedKmeans.__init__(self)
		Clustering.__init__(self, n, K) 
		Estimation.__init__(self)
		BaumWelch.__init__(self)
		self.fw = framework
		self.permutation = None
		self.save_B_matrices = save_B_matrices
		# First state
		self.ini_distribution = ini_distribution
		# Connection matrix : Q
		self.edges_matrix(Q)
		# Transition matrix | Clusters of each node : P, clusters
		self.generate_clusters(P)
		# Effectif of each cluster | B matrix : effectifs, B
		self.effectif_clusters()
		# Adjacency matrix : X
		self.adjacency_matrix(X)
		
	def edges_matrix(self, Q):
		"""Builds the connectivity matrix Q."""
		if Q is None:
			a = np.random.rand(self.K, self.K)
			self.Q = np.tril(a) + np.tril(a, -1).T
		else:
			self.Q = Q
			
	def initial_distribution(self):
		"""Defines the distribution of the community of the first node of the graph."""
		if self.ini_distribution == 'uniform':
			return np.random.randint(0,self.K)
			
	def generate_clusters(self, P):
		"""Samples the communities of the nodes."""
		if self.fw == 'iid':
			self.clusters = np.random.randint(0,self.K,self.n)
		if self.fw == 'markov':
			if P is None:
				self.P = np.random.rand(self.K, self.K)
				self.P /= np.tile(np.sum(self.P, axis=1).reshape(-1,1),(1,self.K))
			else:
				self.P = P
			clusters = [self.initial_distribution()]
			for node in range(1,self.n):
				clusters.append(self.next_state(clusters[-1]))
			self.clusters = clusters

	def effectif_clusters(self):
		"""Computes the sizes of each clusters"""
		self.effectifs = np.zeros(self.K)
		for node in range(self.n):
			self.effectifs[self.clusters[node]] += 1

		if self.save_B_matrices:
			self.B = np.zeros((self.n,self.n))
			for i in range(self.n):
				for j in range(self.n):
					if self.clusters[i]==self.clusters[j]:
						self.B[i,j] = 1/self.effectifs[self.clusters[i]]

	def next_state(self, i):
		"""Method used to sample the community of the next code knowing that the community of the previous node was i."""
		a = np.cumsum(self.P[i,:])
		u = np.random.rand()
		state = 0
		for ind in range(self.K):
			if u < a[ind]:
				state = ind
				break
		return state
	
	def bernoulli(self, q):
		"""Bernoulli distribution."""
		u = np.random.rand()
		if u < q:
			return 1
		else:
			return 0

	def adjacency_matrix(self, X=None):
		"""Builds the adjacency matrix of the graph."""
		if X is None:
			X = np.zeros((self.n,self.n))
			for i in range(1,self.n):
				for j in range(i):
					X[i,j] = self.bernoulli(self.Q[self.clusters[i],self.clusters[j]])
					X[j,i] = X[i,j]
		self.X = X


	def estimate_partition(self):
		"""Runs the algorithm to estimate communities of the nodes."""
		B_relaxed = self.solve_relaxed_SDP()
		if self.save_B_matrices:
			self.B_relaxed = B_relaxed
		barx, bary, C =  self.solve_relaxed_LP(B_relaxed)
		self.Kmedoids(barx, bary, C)

	def proportion_error(self):
		"""Compute the misclassification error of our estimated clustering of the nodes."""
		error = 0
		for k in range(self.K):
			try:
				error += len(self.true_partition[k]-self.approx_partition[k])
			except:
				error += len(self.true_partition[k])
		return (error / self.n)