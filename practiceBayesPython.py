import pymc3 as pm
import arviz as az
import numpy as np

# First, generate some simple coin flipping data:
n, p = 1, .75
Ndata = 50

data = np.random.binomial(n=n,p=p, size=Ndata)


# Now fit the model and plot the posterior using the above generated data

with pm.Model() as prac:
    theta = pm.Beta('theta',alpha = 1., beta = 1.) # uniform prior on probability of coming up heads
    y = pm.Bernoulli('y',p = theta, observed = data)
    trace = pm.sample(1000, cores=1)

az.summary(trace)
az.plot_posterior(trace)
    
