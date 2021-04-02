# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 21:28:30 2021

@author: ricro
"""

# Simple examples of Bayesian statistics in Python

import numpy as np
import pymc3 as pm
import arviz as az

# First, we want to estimate the bias of a coin

p = 0.65 # bias of our coin 

# Now generate data from that coin

n_flips = 150

compare = np.random.rand(n_flips)
heads = (compare <= p).astype(int)

# Now to fit the model

with pm.Model() as coin_model: #This creates an instance of the Model class from pm
    
    # First, set up the priors
    theta = pm.Beta('theta', alpha = 1, beta = 1)
    
    # Now connect to the data
    y = pm.Bernoulli('y', p = theta, observed = heads) # The 'observed' argument is where you input the data
    # This tells PyMC3 that this variable is the likelihood of the data
    
    coin_trace = pm.sample(500, cores = 1) 
    
results = az.summary(coin_trace)
print(results)

az.plot_posterior(coin_trace)


# Now let's try a hierarchical model, where we flip a bunch of different coins,
# each with their own bias, and those biases are distributed a particular way

# First, simulate the data

total_data = np.array([]) # Initializing an array to store data

which_coin = np.array([])

biases = np.array([])

n_coins = 15

for coin in range(n_coins):
    
    # First, choose a bias
    
    # Suppose each coin's bias comes from a Beta(5,10) distribution
    
    bias = np.random.beta(5,10)
    
    biases = np.append(biases, bias)
    
    compare = np.random.rand(n_flips)
    
    heads = (compare <= bias).astype(int)
    
    # Now add data to the total_data:
    total_data = np.append(total_data, heads)
    
    # And add index of coin:
    which_coin = np.append(which_coin, coin*np.ones(n_flips))
    
which_coin = which_coin.astype(int)
    
# Now that we've generated data, let's fit a simple hierarchical model to the data

with pm.Model() as hier_coin:
    
    # First start with hyperpriors
    A = pm.Gamma('A',alpha = 5, beta = 1)
    B = pm.Gamma('B', alpha = 5, beta = 1)
    
    # Now prior on the theta's
    theta = pm.Beta('theta', alpha = A, beta = B, shape = n_coins)
    # The shape argument is how we tell PyMC3 that the number of coins' biases we 
    # want to estimate is n_coins
    
    # This next part is a bit tricky in terms of syntax
    # We need to tell Python which parameter we should use for which data point
    # We can do this by using the following syntax
    
    y = pm.Bernoulli('y', p = theta[which_coin], observed = total_data)
    
    hier_trace = pm.sample(1000, cores = 1)
    
    
hier_results = az.summary(hier_trace)

print(hier_results)
print(hier_results["mean"])
print(biases)

az.plot_posterior(hier_trace)


