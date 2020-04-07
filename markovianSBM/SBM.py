import numpy as np
import cvxpy as cp
import os
import matplotlib.pyplot as plt
from .Clustering import Clustering
from .Estimation import Estimation

class SBM(Clustering, Estimation):
    def __init__(self, n, K, ini_distribution='uniform', framework='iid', Q=None, P=None):
        Clustering.__init__(self, n, K) 
        Estimation.__init__(self)
        self.fw = framework
        self.permutation = None
        # First state
        self.ini_distribution = ini_distribution
        # Connection matrix : Q
        self.edges_matrix(Q)
        # Transition matrix | Clusters of each node : P, clusters
        self.generate_clusters(P)
        # Effectif of each cluster | B matrix : effectifs, B
        self.effectif_clusters()
        # Adjacency matrix : X
        self.adjacency_matrix()
        
    def edges_matrix(self, Q):
        if Q is None:
            a = np.random.rand(self.K, self.K)
            self.Q = np.tril(a) + np.tril(a, -1).T
        else:
            self.Q = Q
            
    def initial_distribution(self):
        if self.ini_distribution == 'uniform':
            return np.random.randint(0,self.K)
            
    def generate_clusters(self, P):
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
        self.effectifs = np.zeros(self.K)
        for node in range(self.n):
            self.effectifs[self.clusters[node]] += 1

        self.B = np.zeros((self.n,self.n))
        for i in range(self.n):
            for j in range(self.n):
                if self.clusters[i]==self.clusters[j]:
                    self.B[i,j] = 1/self.effectifs[self.clusters[i]]
            
    def next_state(self, i):
        a = np.cumsum(self.P[i,:])
        u = np.random.rand()
        state = 0
        for ind in range(self.K):
            if u < a[ind]:
                state = ind
                break
        return state
    
    def bernoulli(self, q):
        u = np.random.rand()
        if u < q:
            return 1
        else:
            return 0

    def adjacency_matrix(self):
        X = np.zeros((self.n,self.n))
        for i in range(1,self.n):
            for j in range(i):
                X[i,j] = self.bernoulli(self.Q[self.clusters[i],self.clusters[j]])
                X[j,i] = X[i,j]
        self.X = X
        
    def solve_relaxed_SDP(self):   
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
        print('True cost', np.trace(self.X@self.X.T@self.B))
        print('Approximated cost', np.trace(self.X@self.X.T@self.B_relaxed))

    def proportion_error(self):
        error = 0
        for k in range(self.K):
            try:
                error += len(self.true_partition[k]-self.approx_partition[k])
            except:
                error += len(self.true_partition[k])
        return (error / self.n)

    def visualize_B_matrices(self):
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