![zepid](docs/images/zepid_logo.png)
# zEpid

zEpid is an epidemiology analysis package, providing easy to use tools for epidemiologists coding in python3. The 
purpose of this package is to provide a toolset to make epidemiology e-z. A variety of calculations and plots can be 
generated through various functions. For a sample walkthrough of what this package is capable of, please look to the 
introduction to Python 3 for epidemiologists at https://github.com/pzivich/Python-for-Epidemiologists

A few highlights: basic epidemiology calculations, easily create functional form assessment plots, 
easily create effect measure plots, generate and conduct diagnostic tests on inverse probability weight, augmented
inverse probability weight estimator, time-varying g-computation algorithm 

If you have any requests for items to be included, please contact me and I will work on adding any requested features. 
You can contact me either through github (https://github.com/pzivich), email (gmail: zepidpy), or twitter (@zepidpy).

# Installation

## Dependencies:
pandas >= 0.18.0, numpy, statsmodels >= 0.7.0, matplotlib >= 2.0, scipy, tabulate

## Installing:
You can install zEpid using `pip install zepid`

# Module Features

## Measures
Calculate measures directly from a pandas dataframe object. Implemented measures include; risk ratio, risk difference, 
odds ratio, incidence rate ratio, incidence rate difference, number needed to treat, sensitivity, specificity, 
population attributable fraction, attributable community risk, standardized mean difference

Other handy features include; splines, Table 1 generator, interaction contrast, interaction contrast ratio

http://zepid.readthedocs.io/en/latest/Measures.html

## Calculator
Calculate measures from summary data. Functions that calculate summary measures from the pandas dataframe use these 
functions in the background. Implemented measures include; risk ratio, risk difference, odds ratio, incidence rate 
ratio, incidence rate difference, number needed to treat, sensitivity, specificity, positive predictive value, negative 
predictive value, screening cost analyzer, counternull p-values, convert odds to proportions, convert proportions to 
odds, population attributable fraction, attributable community risk, standardized mean difference

http://zepid.readthedocs.io/en/latest/Calculator.html

## Graphics
Uses matplotlib in the background to generate some useful plots. Implemented plots include; functional form assessment 
(with statsmodels output), p-value plots/functions, spaghetti plot, effect measure plot (forest plot), receiver-operator 
curve, dynamic risk plot

http://zepid.readthedocs.io/en/latest/Graphics.html

## Causal
Causal is a new branch that houses all the causal inference methods implemented. 

http://zepid.readthedocs.io/en/latest/Causal.html

#### G-Computation Algorithm
Current implementation includes; time-fixed exposure g-formula and time-varying g-formula

#### Inverse Probability Weights 
Current implementation includes; IP Treatment W, IP Censoring W, IP Missing W. Diagnostics are also available for IPTW

#### Augmented Inverse Probability Weights
Current implementation includes the estimator described by Funk et al 2011 AJE

#### Targeted Maximum Likelihood Estimator
Current implementation is a simple TMLE. At the current state, it does NOT include any algorithms in the background
for variable selection or machine learning algorithms (these are to be added in the future)

## Sensitivity Analyses
Includes trapezoidal distribution generator, corrected Risk Ratio

http://zepid.readthedocs.io/en/latest/Sensitivity%20Analyses.html