Project Presentation
====================

This repository provides the code related to the preprint: [https://arxiv.org/abs/2004.04402](https://arxiv.org/abs/2004.04402).

We introduce a new Stochastic Block Model where the communities of the nodes of the graph are assigned using a Markovian dynamic. 
Solving a relaxed Semi Definite Program followed by a rounding step, our algorithm aims at recovering the communities of the nodes of the graph. We show that the misclassification error decays exponentially fast with respect to an appropriate Signal to Noise Ratio. We prove also that in the relatively sparse regime, we are able to estimate with consistency the parameters of our model.

We have implemented our algorithm and the code is contained in the folder **markovianSBM**. In the notebook **experiments_markovian_SBM**, the reader could find simple examples to use our code and to reproduce the results presented in our paper.



