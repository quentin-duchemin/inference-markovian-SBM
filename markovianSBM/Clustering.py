import numpy as np
import cvxpy as cp
import numpy as np
import os


class Clustering:
    def __init__(self, n, K):
        self.n = n
        self.K = K

    def solve_relaxed_LP(self, M):
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
        self.barx = barx.value
        self.bary = bary.value
        self.C = C
        
    def Kmedoids(self):
        
        # Step 1 : consolidating locations
        
        ## It consists in moving revelantly demand.        
        ## All locations with positive demand will be far from eachother, garanteeing:
        ##    - not increase the cost of the fractional solution
        ##    - each feasible integer solution for the modified instance can be converted to a feasible integer 
        ##      solution for the original instance with a small added cost
        d = np.ones(self.n)
        barC = np.diag(self.C@self.barx)
        ind = np.argsort(barC)
        for j in range(self.n):
            for i in range(j):
                if d[ind[i]]>0 and self.C[ind[i],ind[j]]<=4*barC[ind[j]]:
                    d[ind[i]] += d[ind[j]]
                    d[ind[j]] = 0

        # Step 2 : Consolidating centers
        
        ## Construction of a 1/2-resticted solution
        ## y_i=0 if d[i]=0 and y_i >=1/2 otherwise (without paying this too much). 
        px = np.copy(self.barx)
        py = np.copy(self.bary)
        # Closest[i] will be the closest node to i (except i), let's say j, satisfying d[j]>0
        closest = []
        dico_weights = {}
        sorted_lines_C = np.zeros((self.n,self.n))
        for i in range(self.n):
            ls = np.argsort(self.C[i,:])
            sorted_lines_C[i,:] = ls
            j = 0
            while (d[int(ls[j])]==0 or i==int(ls[j])):
                j += 1
            closest.append(ls[j])
            
            # We only move the nodes with a null demand and a partial open center (i.e y>0) and thus that can't be a center            
            if d[i]==0 and py[i]>0:
                # We assign i to the closest point j such that d[j] > 0             
                py[j] = min(1 , py[i] + py[j])
                py[i] = 0
                for pj in range(self.n):
                    px[j,pj] += px[i,pj]
                    px[i,pj] = 0
            
            if d[i]>0:
                dico_weights[i] = d[i]*self.C[closest[i],i]
                
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
        if len(dico_closest)>0:
            j, s_j = dico_closest.popitem()
        level = 0
        while len(dico_closest)>0:
            if level%2==0:
                odd_level.append(j)
            else:
                even_level.append(j)
            level += 1
            if s_j in dico_closest:
                j = s_j
                s_j = dico_closest[s_j]
                del dico_closest[j]
            elif len(dico_closest)>0:              
                j, s_j = dico_closest.popitem()
                level = 0
        if len(odd_level)<len(even_level):
            centers = centers + odd_level
        else:
            centers = centers + even_level
        
        # -1 is assigned to the nodes that are not centers. Otherwise, we numerote them.
        num_center = -np.ones(self.n)
        for ind, j in enumerate(centers):
            num_center[j] = int(ind)

        self.A = np.zeros((self.n,self.K))
        for i in range(self.n):
            if num_center[i] == -1:
                ls = sorted_lines_C[i,:]
                count = 0
                while num_center[int(ls[count])] == -1:
                    count += 1
                self.A[i,int(num_center[int(ls[count])])] = 1
            else:
                self.A[i,int(num_center[i])] = 1
          
        self.clusters_approx = []
        for i in range(self.n):
          for j in range(self.K):
            if self.A[i,j]==1:
              self.clusters_approx.append(j)

        self.true_partition   = self.build_partition(self.clusters)
        approx_partition = self.build_partition(self.clusters_approx) 
        self.find_permutation(self.true_partition, approx_partition)
        self.approx_partition = {}
        for k in range(self.K):
          self.approx_partition[k] = approx_partition[self.permutation[k]]
        clusts_approx = np.copy(self.clusters_approx)
        for i,group in enumerate(clusts_approx):
          self.clusters_approx[i] = self.permutation[group]

    def build_partition(self, clust):
      d = {i:{} for i in range(self.K)}
      for i in range(len(clust)):
          d[clust[i]] = {i}
      return (d)

    def find_permutation(self, true_partition, approx_partition):
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

