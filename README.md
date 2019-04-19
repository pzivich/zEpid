![zepid](docs/images/zepid_logo.png)
# zEpid

[![PyPI version](https://badge.fury.io/py/zepid.svg)](https://badge.fury.io/py/zepid)
[![Build Status](https://travis-ci.com/pzivich/zEpid.svg?branch=master)](https://travis-ci.com/pzivich/zEpid)
[![Join the chat at https://gitter.im/zEpid/community](https://badges.gitter.im/zEpid/community.svg)](https://gitter.im/zEpid/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

zEpid is an epidemiology analysis package, providing easy to use tools for epidemiologists coding in python3. The 
purpose of this library is to provide a toolset to make epidemiology e-z. A variety of calculations and plots can be 
generated through various functions. For a sample walkthrough of what this library is capable of, please look to the 
introduction to Python 3 for epidemiologists at https://github.com/pzivich/Python-for-Epidemiologists

A few highlights: basic epidemiology calculations, easily create functional form assessment plots, 
easily create effect measure plots, generate and conduct diagnostic tests. Implemented estimators include; inverse 
probability of treatment weights, inverse probability of censoring weights, inverse probabilitiy of missing weights, 
augmented inverse probability weights, time-fixed g-formula, Monte Carlo g-formula, Iterative conditional g-formula, 
and targeted maximum likelihood (TMLE)

If you have any requests for items to be included, please contact me and I will work on adding any requested features. 
You can contact me either through GitHub (https://github.com/pzivich), email (gmail: zepidpy), or twitter (@zepidpy).

# Installation

## Installing:
You can install zEpid using `pip install zepid`

## Dependencies:
pandas >= 0.18.0, numpy, statsmodels >= 0.7.0, matplotlib >= 2.0, scipy, tabulate

# Module Features

## Measures
Calculate measures directly from a pandas dataframe object. Implemented measures include; risk ratio, risk difference, 
odds ratio, incidence rate ratio, incidence rate difference, number needed to treat, sensitivity, specificity, 
population attributable fraction, attributable community risk, standardized mean difference

Other handy features include; splines, Table 1 generator, interaction contrast, interaction contrast ratio

For a narrative description:
http://zepid.readthedocs.io/en/latest/Measures.html

For guided tutorials with Jupyter Notebooks:
https://github.com/pzivich/Python-for-Epidemiologists/blob/master/3_Epidemiology_Analysis/a_basics/1_basic_measures.ipynb

## Calculator
Calculate measures from summary data. Functions that calculate summary measures from the pandas dataframe use these 
functions in the background. Implemented measures include; risk ratio, risk difference, odds ratio, incidence rate 
ratio, incidence rate difference, number needed to treat, sensitivity, specificity, positive predictive value, negative 
predictive value, screening cost analyzer, counternull p-values, convert odds to proportions, convert proportions to 
odds, population attributable fraction, attributable community risk, standardized mean difference

For a narrative description:
http://zepid.readthedocs.io/en/latest/Calculator.html

## Graphics
Uses matplotlib in the background to generate some useful plots. Implemented plots include; functional form assessment 
(with statsmodels output), p-value plots/functions, spaghetti plot, effect measure plot (forest plot), receiver-operator 
curve, dynamic risk plot

For a narrative description:
http://zepid.readthedocs.io/en/latest/Graphics.html

## Causal
Causal is a new branch that houses all the causal inference methods implemented. 

For a narrative description:
http://zepid.readthedocs.io/en/latest/Causal.html

#### G-Computation Algorithm
Current implementation includes; time-fixed exposure g-formula, Monte Carlo g-formula, and iterative conditional 
g-formula

#### Inverse Probability Weights 
Current implementation includes; IP Treatment W, IP Censoring W, IP Missing W. Diagnostics are also available for IPTW. 
IPMW supports monotone missing data

#### Augmented Inverse Probability Weights
Current implementation includes the estimator described by Funk et al 2011 AJE

#### Targeted Maximum Likelihood Estimator
TMLE can be estimated through standard logistic regression model, or through user-input functions. Alternatively, users 
can input machine learning algorithms to estimate probabilities. 

#### G-estimation of Structural Nested Mean Models
Single time-point g-estimation of structural nested mean models are supported.

## Sensitivity Analyses
Includes trapezoidal distribution generator, corrected Risk Ratio

For a narrative description:
http://zepid.readthedocs.io/en/latest/Sensitivity%20Analyses.html