import numpy as np
import cvxpy as cp
import os
import matplotlib.pyplot as plt

class RelaxedKmeans():
	"""Class solving the Semi Definite Program which is a relaxed version of the K means problem."""
	def __init__(self):
		pass
		
	def solve_relaxed_SDP(self):   
		"""Solve the Semi Deifnite Program and save the optimal solution."""
		density = 2*np.sum(self.X) / (self.n * (self.n - 1))
		#alpha = min(1, (self.K**3/self.n)*np.exp(2*self.n*density))
		alpha = 1 / min(list(filter(lambda x : x>0, self.effectifs)))
		B = cp.Variable((self.n,self.n))
		constraints = [B >> 0]
		constraints += [
			B == B.T
		]
		ones = np.ones(self.n)
		constraints += [
			B@ones == ones
		]
		constraints += [
			cp.trace(B)==self.K
		]
		constraints += [
			B[i//self.n,i%self.n]>=0 for i in range(self.n * self.n)
		]
		constraints += [
			alpha-B[i//self.n,i%self.n]>=0 for i in range(self.n * self.n)
		]
		prob = cp.Problem(cp.Minimize(-cp.trace(self.X@(self.X).T@B)),
						  constraints)
		prob.solve()
		self.B_relaxed = B.value

	def compute_costs(self):
		"""Show the value of the objective function of the SDP with the matrix $B^*$ 
		which is the solution of the K means problem and with $\hat{B}$ which is the optimal solution of the SDP."""
		print('True cost', np.trace(self.X@self.X.T@self.B))
		print('Approximated cost', np.trace(self.X@self.X.T@self.B_relaxed))

	def visualize_B_matrices(self):
		"""Method providing a vizualisation of the matrices $B^*$ (optimal solution of the K-means problem) and 
		$\hat{B}$ optimal solution of the SDP relaxation. This allows to easily see if a final rounding step on the rows of 
		$\hat{B}$ could allow to reach a relevant clustering of the nodes of the graph."""
		I = np.argsort(self.clusters)
		fig = plt.figure()
		ax = fig.add_subplot(121)
		temp = self.B_relaxed[I,:]
		ax.imshow(temp[:,I], cmap='Greys')
		ax.set_title('$\hat{B}$', fontsize=15)
		ax = fig.add_subplot(122)
		temp = self.B[I,:]
		sc=ax.imshow(temp[:,I], cmap='Greys')
		ax.set_title('$B^*$', fontsize=15)
		left, bottom, width, height = ax.get_position().bounds
		cax = fig.add_axes([left+width+0.01, bottom, width*0.05, height])
		plt.colorbar(sc, cax=cax)
		plt.show()