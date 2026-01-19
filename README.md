# Learning-curve sample size simulation (hierarchical / long-format data)

This repository contains an simulation that illustrates sample size planning for predictive modeling using learning curves. The goal is to estimate the most efficient sample size because we want robust models but study participants aren't cheap. 
Datasets in this code are synthetic.

**What does this thing do**

The script: 

- generates a synthetic hierarchical (repeated-measures) dataset in long format with N participants, K items, a mix of predictors (participant-level and item-level predictors, some noise variables to stress-test our algorithm a little bit and to reflect real psychological experiments).
  
- constructs a realistic outcome as a blend of signal derived from a few relevant predictors, and some noise, and centers outcomes within-participants to predict within person relative behavior towards an item rather than absolute (because we didn't care about that in our case).
  
- performs a rigorous  participant-wise train-test split so that participants in the training set don't leak in the test set.
  
- tests four algorithms (elastic net regression, XGboost, random forest, single hidden layer NN) against a shuffled outcome benchmark to see if they beat chance at predicting the outcome.
  
- outputs a nice learning-curve plot showing mean test R2 (with error bars) as a function of the number of participants
