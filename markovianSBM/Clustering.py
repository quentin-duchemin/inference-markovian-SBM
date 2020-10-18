import numpy as np
import cvxpy as cp
import numpy as np
import os


def add_liste(Level, Liste, Node, UP=True):
	if Level==-1:
		Liste = [Node] + Liste
	elif Level >= len(Liste):
		Liste.append([Node])
		if UP:
			Level -= 1
		else:
			Level += 1
	else:
		Liste.append([Node])
		if UP:
			Level -= 1
		else:
			Level += 1
	return Level, Liste
	
def recu(node, level, graph, liste, node2ind, dejavu, dico_closest, nodes):
	L = []
	for ind in range(len(nodes)):
		if graph[ind,node2ind[node]] and dejavu[ind]==0:
			L.append(ind)
	for ind in L:
		if graph[ind,node2ind[node]] and dejavu[ind]==0:
			level, liste = add_liste(level, liste, nodes[ind], UP=True)
			dejavu[ind] = 1
			graph, liste, dejavu = recu(nodes[ind], level, graph, liste, node2ind, dejavu, dico_closest, nodes)
	return graph, liste, dejavu


class Clustering:
	"""Class that performs the final rounding step on the rows of the matrix $\hat{B}$ which is the 
	optimal solution of the SDP relaxation of the K-means problem."""
	def __init__(self, n, K):
		self.n = n
		self.K = K

	def solve_relaxed_LP(self, M):
		"""Prelimanary to run the K-medoid algorithm."""
		barx = cp.Variable((self.n,self.n))
		bary = cp.Variable(self.n)
		C = np.zeros((self.n,self.n))
		for i in range(1, self.n):
			for j in range(i):
				C[i,j] = np.linalg.norm(M[:,i]-M[:,j], ord=1)
				C[j,i] = C[i,j]
		
		constraints = [ cp.sum(barx[:,j]) == 1 for j in range(self.n)]
		constraints += [
		   - barx[i%self.n,i//self.n] +  bary[i%self.n] >= 0 for i in range(self.n * self.n)
		]
		constraints += [
		   self.K - cp.sum(bary) >= 0
		]
		constraints += [
		   barx[i%self.n,i//self.n] >=0 for i in range(self.n * self.n)
		]
		constraints += [
		   bary[i] >=0 for i in range(self.n)
		]
		prob = cp.Problem(cp.Minimize(cp.trace(C@barx)),
						 constraints)
		prob.solve()
		barx = barx.value
		bary = bary.value
		return barx, bary, C
		
	def Kmedoids(self, barx, bary, C):
		"""Kmedoid algorithm that performs a rounding step on the rows of $\hat{B}$."""
		# Step 1 : consolidating locations
		
		## It consists in moving revelantly demand.		
		## All locations with positive demand will be far from eachother, garanteeing:
		##	- not increase the cost of the fractional solution
		##	- each feasible integer solution for the modified instance can be converted to a feasible integer 
		##	  solution for the original instance with a small added cost
		d = np.ones(self.n)
		barx = barx * (1* (barx>0))
		barC = np.diag(C @ barx)
		ind = np.argsort(barC)
		for j in range(self.n):
			for i in range(j):
				if d[ind[i]]>0 and C[ind[i],ind[j]]<=4*barC[ind[j]]:
					d[ind[i]] += d[ind[j]]
					d[ind[j]] = 0
		# Step 2 : Consolidating centers
		
		## Construction of a 1/2-resticted solution
		## y_i=0 if d[i]=0 and y_i >=1/2 otherwise (without paying this too much). 
		px = np.copy(barx)
		py = np.copy(bary)
		# Closest[i] will be the closest node to i (except i), let's say j, satisfying d[j]>0
		closest = []
		dico_weights = {}
		sorted_lines_C = np.zeros((self.n,self.n))
		for i in range(self.n):
			ls = np.argsort(C[i,:])
			sorted_lines_C[i,:] = ls
			j = 0
			while (d[int(ls[j])]==0 or i==int(ls[j])):
				j += 1
			closest.append(ls[j])
			
			# We only move the nodes with a null demand and a partial open center (i.e y>0) and thus that can't be a center			
			if d[i]==0 and py[i]>0:
				# We assign i to the closest point j such that d[j] > 0			 
				py[ls[j]] = min(1 , py[i] + py[ls[j]])
				py[i] = 0
				for pj in range(self.n):
					px[ls[j],pj] += px[i,pj]
					px[i,pj] = 0
			
			if d[i]>0:
				dico_weights[i] = d[i]*C[closest[i],i]
				
		## Construction of a (1/2-1)-integral solution
		## y_i=0 if d[i]=0 and y_i =1/2 or 1 otherwise (without paying this too much). 
		pn = len(dico_weights)
		haty = np.zeros(self.n)
		hatx = np.zeros((self.n,self.n))
		# We sort the locations j \in N' (ie such that d[j]>0) in decreasing order of their weight
		sorted_dico_weights = {k: v for k, v in sorted(dico_weights.items(), key=lambda item: item[1], reverse=True)}
		list_keys = sorted_dico_weights.keys() #list of nodes in N' (i.e. with d[i]>0)
		for ind, j in enumerate(list_keys):
			if ind < 2*self.K-pn:
				haty[j] = 1
			else:
				haty[j] = 1/2
				hatx[closest[j],j] = 1/2
			hatx[j,j] = haty[j]

		# Step 3 : rounding a {1/2, 1}-integral solution to an integral one
		centers = [k for k in list_keys if haty[k] == 1]
		odd_level = []
		even_level = []
		dico_closest = {i:closest[i] for i in list_keys if haty[i]==1/2}
		import copy


		dic = dico_closest.copy()
		for key,value in dic.items():
				try:
						if dico_closest[value]==key:
								del dico_closest[key]
				except:
						pass

		keys = list(dico_closest.keys())
		values = list(dico_closest.values())
		nodes = list(set(keys+values))
		graph = np.zeros((len(nodes),len(nodes)))
		node2ind = {node:ind for ind,node in enumerate(nodes)}
		for ind,key in enumerate(keys):
				graph[node2ind[key],node2ind[values[ind]]] = 1

		dejavu = np.zeros(len(nodes))

		while len(dejavu)!=np.sum(dejavu):
				notfound = True
				indj = 0
				level = 0
				liste = []
				while notfound and indj<len(nodes):
						if np.sum(graph[:,indj])==0 and np.sum(graph[indj,:])>0 and dejavu[indj]==0:
								notfound = False
								j = nodes[indj]
								liste.append([j])
								s_j = dico_closest[j]
								liste.append([s_j])
								inds_j = node2ind[s_j]
								dejavu[indj] = 1
								dejavu[inds_j] = 1
								graph, liste, dejavu = recu(s_j, 1, graph, liste, node2ind, dejavu, dico_closest, nodes)
								level = 2
								j = s_j
								indj = inds_j

								leaf = False
								while not(leaf):
									if np.sum(graph[indj,:])>0:
										s_j = dico_closest[j]
										inds_j = node2ind[s_j]
										if dejavu[inds_j]==0:
												dejavu[inds_j] = 1
												graph, liste, dejavu = recu(s_j, level-1, graph, liste, node2ind, dejavu, dico_closest, nodes)
												level, liste = add_liste(level, liste, s_j, UP=False)
												j = s_j
												indj = inds_j
										else:
										  leaf = True
									else:
									  leaf = True
						indj += 1
				for ind in range(len(liste)):
						if ind % 2:
								even_level += liste[ind]
						else:
								odd_level += liste[ind]
			
				
		if len(odd_level)<len(even_level):
			centers = centers + list(set(odd_level))
		else:
			centers = centers + list(set(even_level))
		centers = list(set(centers))
		# -1 is assigned to the nodes that are not centers. Otherwise, we numerote them.
		num_center = -np.ones(self.n)
		for ind, j in enumerate(centers):
			num_center[j] = int(ind)

		A = np.zeros((self.n,self.K))
		for i in range(self.n):
			if num_center[i] == -1:
				ls = sorted_lines_C[i,:]
				count = 0
				while num_center[int(ls[count])] == -1:
					count += 1
				A[i,int(num_center[int(ls[count])])] = 1
			else:
				A[i,int(num_center[i])] = 1
		  
		self.clusters_approx = []
		for i in range(self.n):
			for j in range(self.K):
				if A[i,j]==1:
					self.clusters_approx.append(j)

		self.true_partition   = self.build_partition(self.clusters)
		approx_partition = self.build_partition(self.clusters_approx) 
		self.find_permutation(self.true_partition, approx_partition)
		inverse = [0] * len(self.permutation)
		for i, p in enumerate(self.permutation):
				inverse[p] = i
		clusts_approx = np.copy(self.clusters_approx)
		for i,group in enumerate(clusts_approx):
			self.clusters_approx[i] = inverse[group]

	def build_partition(self, clust):
		"""Given a list clust that associates to each node its community, this methods builds the associated
		partition of the noes of the graph."""
		d = {i:set() for i in range(self.K)}
		for i in range(len(clust)):
			d[clust[i]].add(i)
		return (d)

	def find_permutation(self, true_partition, approx_partition):
		"""Find the permutation between the names of the true communities and the ones estimated by our 
		algorithm."""
		import itertools
		permus = list(itertools.permutations([i for i in range(self.K)]))
		best_error = np.float('inf')
		for permu in permus:
			error = 0
			for k in range(self.K):
				try:
					error += len(true_partition[k]-approx_partition[permu[k]])
				except:
					error += len(true_partition[k])
			if error < best_error:
				best_error = error
				self.permutation = permu
