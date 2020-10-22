Project Presentation
====================

This repository provides the code related to the paper 806 under review for the AISTATS conference 2021.

### Description of the project

We provide a general framework to tackle the so-called link prediction and collaborative filtering problem in growth model of random graphs. We illustrate our method with the Markovian Stochastic Block Model where the communities of the nodes of the graph are assigned using a Markovian dynamic. Additionnally, we show that our approach can be used to identify the errors made by a given clustering algorithm and we provide a heuristic to estimate the number of communities in a given graph with clusters.

Solving a relaxed Semi Definite Program followed by a rounding step, our clustering algorithm aims at recovering the communities of the nodes of the graph. We show that the misclassification error decays exponentially fast with respect to an appropriate Signal to Noise Ratio. We prove also that in the relatively sparse regime, we are able to estimate with consistency the parameters of our model.

We have implemented our algorithm and the code is contained in the folder **markovianSBM**. In the notebook **experiments**, the reader could find simple examples to use our code and to reproduce the results presented in our paper.


### How to launch your own experiments ?

A user-friendly notebook has been written to have a simple description to use our package. Open a terminal and run the following commands. 

```python
git clone https://github.com/quentin-duchemin/inference-markovian-SBM.git
cd inference-markovian-SBM
sudo pip3 install --upgrade --force-reinstall virtualenv
# create and active the virtualenv
virtualenv pyenv
. pyenv/bin/activate
# install the required python packages in the virtualenv
pip install -r requirements.txt
# launch the jupyter to open the notebook experiment_markovian_SBM.ipynb and get familiar with the way to run the code
jupyter notebook
```