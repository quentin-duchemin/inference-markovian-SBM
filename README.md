# inference-markovian-SBM

This repository provides the code related to the preprint : . A documentation for this code is available [here](https://inference-markovian-sbm.readthedocs.io/).

### Description of the project


We introduce a new Stochastic Block Model where the communities of the nodes of the graph are assigned using a Markovian dynamic. 
Solving a relaxed Semi Definite Program followed by a rounding step, our algorithm aims at recovering the communities of the nodes of the graph. We show that the misclassification error decays exponentially fast with respect to an appropriate Signal to Noise Ratio. We prove also that in the relatively sparse regime, we are able to estimate with consistency the parameters of our model.

We have implemented our algorithm and the code is contained in the folder **markovianSBM**. In the notebook **experiments_markovian_SBM**, the reader could find simple examples to use our code and to reproduce the results presented in our paper.


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