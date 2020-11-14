# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 14:00:55 2020

@author: ricro
"""

import pymc3 as pm
import arviz as az
import numpy as np
import pandas as pd
import theano.tensor as T
import matplotlib.pyplot as plt

## Goal is to practice simple Bayesian fitting in pymc3 when:
    # (1) Each person goes through multiple trials
    # (2) We have multiple groups of people
    
    
# Practice using a simple IRT simulation with binary elements
# Survey with 10 questions:
# (For more background on IRT, see the short Primer on the github page
# that hosts this code.)
    
np.random.seed(19284)

Disc = np.linspace(.5,1.5,num = 10) # called 'a' in the primer
Diff = np.linspace(-1,1, num = 10) # called 'b' in the primer
# We want to ultiately compare the estimated values to the true values,
# so it's easier if we fix them like this beforehand.
# In essence, this kind of scale becomes more difficult as the items progress,
# and also become more discriminating between groups as the items progress

# However, in practice, the simulated data we have here is too few to get an 
# accurate estimate of the item parameters. You typically need several hundred
# respondents for that; this can be seen by observing the mean values of the 
# posteriors in the "traces" variable


# A simple function to get the probability of responding "1" on an item
def item(a,b,theta):
    x = 1 + np.exp(-a*(theta - b))
    return x ** (-1)

# Generate group theta's:
N = 30 # how many participants in each group?
group1_theta = np.random.normal(loc = 1, scale = 1, size = N)
group2_theta = np.random.normal(loc = -1, scale = 1, size = N)

# So the two groups differ in their mean latent trait values.

theta = np.append(group1_theta, group2_theta)
theta = np.repeat(theta, 10) # we do this to build the dataframe

ID = np.repeat(np.arange(2*N), 10)

# Also want Group IDs
Group_ID = np.append( np.repeat(np.zeros(shape = 10), N), 
                     np.repeat(np.ones(shape = 10), N))
Group_ID.astype('int') # Have to convert to integers to use in the pm.Model()

# Also indexing item number:
Item_ID = np.tile(np.arange(10), 2*N)
Item_ID.astype('int')

# Here we create a dataframe where each row is a different trial,
# so the IDs and the theta values need to each be repeated 10 times
# (10 for the number of trials we'll simulate)

prac_data = {"ID":ID, "Group ID": Group_ID, "Item ID": Item_ID,"Theta":theta}
prac_data = pd.DataFrame(prac_data)
print(prac_data)

# Now we need to generate data for each person

def genData(A,B,theta):
    n_item = len(A)
    data = np.zeros(shape = n_item)
    for j in range(n_item):
        # Generate theoretical probability of responding to item with code 1
        prob = item(A[j], B[j], theta)
        x = np.random.rand()
        # Compare theoretical prob with a uniform random variable 
        # to determine response
        data[j] = (x <= prob)
    
    # Return array of 0's and 1's indicating which responses were made
    return data 

# Now we want to generate data for each person
ids = np.arange(2*N)

response = np.array([]) # responses will be either "1" or "0"
# In IRT, this could be correct/incorrect for knowledge tests,
# or endorsed/not endorsed for opinion inventories

for prt in ids:
    # first, get the correct theta value:
    to_get = prac_data["ID"] == prt
    th = prac_data[to_get]["Theta"].unique()[0]
    
    data = genData(Disc, Diff, th)
    response = np.append(response, data)
    
# print(len(Correct))

prac_data["Response"] = response


### Now we're ready to fit the data using a basic Bayesian model
    
group = np.array([])
for prt in range(2*N):
    d = prac_data["ID"] == prt
    group = np.append(group, prac_data[d]["Group ID"].unique())
    
#print(len(group))   # To check that output was correct size
group = group.astype('int') 
data = prac_data["Response"]

with pm.Model() as irt:
    # Priors
    
    # Item parameters:
    a = pm.Gamma('a', alpha = 1, beta = 1, shape = 10) # Discrimination
    
    b = pm.Normal('b', mu = 0, sd = 1, shape = 10) # Difficulty
    
    # Now for the hyperpriors on the groups: shape = 2 as there are 2 groups
    theta_mu = pm.Normal('theta_mu', mu = 0, sd = 1, shape = 2)
    theta_sigma = pm.Uniform('theta_sigma', upper = 2, lower = 0, shape = 2)
    
   
    # Individual-level person parameters:
    # group is a 2*N array that lets the model know which
    # theta_mu to use for each theta to estimate
    theta = pm.Normal('theta', mu = theta_mu[group],
                             sd = theta_sigma[group], shape = 2*N)
    
    
    # Here, we're building an array of the probabilities we need for 
    # each trial:
    p = np.array([])
    for n in range(2*N):
        for t in range(10):
            x = -a[t]*(theta[n] - b[t])
            p = np.append(p, x)
      
    # Here, we turn p into a tensor object to put as an argument to the
    # Bernoulli random variable
    p = T.stack(p, axis = 0)
    
    y = pm.Bernoulli('y', logit_p = p, observed = data)
    
   
    # On my computer, this took about 5 minutes to run. 
    traces = pm.sample(1000, cores = 1)
        
print(az.summary(traces)) # Summary of parameter distributions

# We can check that the model found the difference between the two groups
# by checking the mean of the difference distribution between
# theta_mu[1] and theta_mu[0]:
    
# Obtaining distribution of differences:
t = traces["theta_mu"][:,1] - traces["theta_mu"][:,0]

plt.hist(t, bins = 50)
plt.title('Distribution of $\theta_\mu_2 - \theta_\mu_1$')

# Recovers the difference in the means pretty well
print(t.mean())
    