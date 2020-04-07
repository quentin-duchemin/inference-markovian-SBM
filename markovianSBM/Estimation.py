import numpy as np
import os


class Estimation():
    def __init__(self):
        pass 

    def estimate_transition_matrix(self):
        assert(self.fw == 'markov')
        self.approx_P = np.zeros((self.K,self.K))
        if self.permutation is None:
            true_partition   = self.build_partition(self.clusters)
            approx_partition = self.build_partition(self.clusters_approx)
            self.find_permutation(true_partition, approx_partition)
        permu_approx2true = np.arange(len(self.permutation))[np.argsort(self.permutation)]
        self.approx_pi = np.zeros(self.K)
        for i in range(self.n-1):
            k = permu_approx2true[self.clusters_approx[i]]
            l = permu_approx2true[self.clusters_approx[i+1]]
            self.approx_P[k,l] += 1
            self.approx_pi[k]  += 1
        self.approx_pi[l] += 1
        self.approx_pi /= self.n
        self.approx_P /= self.n-1
        self.approx_P /= np.tile(self.approx_pi.reshape(-1,1),(1,self.K))

    def estimate_connectivity_matrix(self):
        self.approx_Q = np.zeros((self.K,self.K))
        if self.permutation is None:
            true_partition   = self.build_partition(self.clusters)
            approx_partition = self.build_partition(self.clusters_approx)
            self.find_permutation(true_partition, approx_partition)
        permu_approx2true = np.arange(len(self.permutation))[np.argsort(self.permutation)]
        approx_effectifs = np.zeros(K)
        for i in range(self.n):
            k = permu_approx2true[self.clusters_approx[i]]
            approx_effectifs[k] += 1
            for j in range(self.n):
                l = permu_approx2true[self.clusters_approx[j]]
                self.approx_Q[k,l] += self.X[i,j]
        for k in range(self.K):
            for l in range(self.K):
                self.approx_Q[k,l] /= approx_effectifs[k]*approx_effectifs[l]


    def estimate_partition(self):
        self.solve_relaxed_SDP()
        self.solve_relaxed_LP(self.B_relaxed)
        self.Kmedoids()

    def estimate_parameters(self):
        self.estimate_transition_matrix()
        self.estimate_connectivity_matrix()