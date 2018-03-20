This foler contains 5 files.

SimLinearODE.m is the main Matlab code to simulate the parametric ODE model  
in Section 4 of the article "Bayesian Robust Inference of Mixed-effects Ordinary
Differential Equations Models Using Heavy-tailed Distributions" by Liu, Wang, Nie and Cao (2018)

This Matlab file will use the other four files:

LinearOde.m is the parametric ODEs function.

LinearLSQNONLIN_fun.m is the numeric ODE solver applying the penalized splines approach.

LinearMCMC_funNN.m is the MCMC algorithm assuming that the random-effects and errors follow Gaussian distributions.

LinearMCMC_funTT.m is the MCMC algorithm assuming that the random-effects and errors follow heavy-tailed distributions.
