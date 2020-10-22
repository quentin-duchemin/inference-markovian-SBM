import numpy as np
import os


class Estimation():
	"""Class related to the estimation of the parameters of our model."""
	def __init__(self):
		pass 

	def estimate_transition_matrix(self):
		"""
		Estimates the transition matrix and the invariant measure of the chain.
		"""
		assert(self.fw == 'markov')
		self.approx_P = np.zeros((self.K,self.K))
		self.approx_pi = np.zeros(self.K)
		for i in range(self.n-1):
			k = self.clusters_approx[i]
			l = self.clusters_approx[i+1]
			self.approx_P[k,l] += 1
			self.approx_pi[k]  += 1
		self.approx_pi[l] += 1
		self.approx_pi /= self.n
		self.approx_P /= self.n-1
		self.approx_P /= np.tile(self.approx_pi.reshape(-1,1),(1,self.K))

	def estimate_connectivity_matrix(self):
		"""
		Estimates the connectivity matrix.
		"""
		self.approx_Q = np.zeros((self.K,self.K))
		approx_effectifs = np.zeros(self.K)
		for i in range(self.n):
			k = self.clusters_approx[i]
			approx_effectifs[k] += 1
			for j in range(self.n):
				l = self.clusters_approx[j]
				self.approx_Q[k,l] += self.X[i,j]
		for k in range(self.K):
			for l in range(self.K):
				self.approx_Q[k,l] /= approx_effectifs[k]*approx_effectifs[l]

	def estimate_parameters(self):
		"""
		Estimates the parameters of the model.
		"""
		self.estimate_transition_matrix()
		self.estimate_connectivity_matrix()

	def estimate_effectifs(self):
		"""
		Computes the sizes of each clusters
		"""
		self.approx_effectifs = np.zeros(self.K)
		self.approx_transitions = np.zeros((self.K,self.K))
		for node in range(self.n):
			self.approx_effectifs[self.clusters_approx[node]] += 1
		for i in range(self.n-1):
			k = self.clusters_approx[i]
			l = self.clusters_approx[i+1]
			self.approx_transitions[k,l] += 1
