# bayes-irt
Some practice code for fitting an item response model the Bayesian way - especially when the probability of a Bernoulli random variable changes on every trial.

Here is the main documentation to get started with PyMC3: https://docs.pymc.io/notebooks/getting_started.html

# Basic Idea
I worked with item response models before in my thesis work, but I had done the Bayesian modeling in JAGS and MATLAB. Since I've been learning Python, I wanted to try doing the same kinds of analyses using the pymc3 package. 

Item response theory consists of a set of models that are meant to describe how people respond to surveys or inventories that measure some aspect of the person. For instance, if I have an inventory that is supposed to measure the amount of anxiety a person is currently feeling, the set of questions would presumably be related to the concept of anxiety, and the model would give a score, called the latent trait, that is supposed to represent a kind of summary measure of the person's anxiety level. For more detail on the basics of item response theory, I uploaded a PDF that is a short primer on the subject, with some further references included. (The primer also includes as an appendix a proof of the "total information" formula presented in the primer, a proof technique I haven't seen used in the item response theory literature before.) 

# Data
The data are simulated in this project. I took a set of 10 equally spaced item parameters to represent my inventory/scale that I'm modeling. I also supposed that there were two groups of people I was modeling. The first group was presumed to represent a group very low on the latent trait; this group has a mean score of -1.5. The second group was presumed to be fairly high in their latent traits; they were given a score of 1.5. I sampled 30 latent traits (called "theta" in the code) for each group (for a total of 60 latent traits). I then used these variables to simulate a set of binary responses.

As I simulated 10 item parameters, the simulation represents two groups of 30 people each taking an inventory of 10 questions. The simulation overall represents a scenario where one group is low on the latent trait while the other group is fairly high. (For instance, this might represent a scenario where I'm measuring trait anxiety in two groups, where I think one group is highly anxious and the other is not.) 

# Why I did this Project
The principal reason I completed this project was to practice using the pymc3 package to fit data in python. Using pymc3, I represented the simulated inventory responses as Bernoulli random variables. Typically, in simple coin examples, it is presumed that the probability of coming up heads is the same on each trial, some parameter usually denoted "p." However, as item response models require that the probability of responding changes from item to item, I needed a way to fit the model by allowing this probaiblity of "coming up heads" (responding/endorsing) to change for each trial. This issue was especially made more difficult by the fact that each person had more than one data point. 

After struggling for a bit to find a way to do it, I managed to come up with the following solution: First, build a numpy array of the probabilities you want to use for each trial of data you have. I built the array by essentially generating the input to a logistic function trial by trial in a loop within the pm.Model(). Then, you can use theano.tensor.stack(your_array, axis = 0) to turn the array into a single tensor that can then be input into either the p argument or the logit_p argument of the Bernoulli instance in your pm.Model(). In the example posted here, I found it easier to build the "-a(theta - b)" part first, then pass that tensor to logit_p to have Python automatically transform the tensor into a tensor of probabilities. 

This example was very helpful for me to build, since in my work there are many instances in which the probability distribution of scores changes over trials, like in a learning experiment. I thought this would be helpful to other people who may be struggling with the same kind of problem and may have been stuck like I was. 

As an aside, I also got more practice working with numpy and pandas, especially in simulating and organizing the data for fitting with the model. 
