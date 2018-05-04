'''
_____________________________________________________________________________________________
|                                                                                           |
|       zEPID package: making epidemiology with Python e-z                                  |
|                                                                                           |
| This package is to provide easy to use tools to aid in epidemiological analyses. A full   |
| applied guide to the entire package is available at: https://github.com/pzivich/zepid     |
| Current contents include:                                                                 |
|       -Basics tools for pandas dataframes and summary data                                |
|       -Useful graphics                                                                    |
|       -Inverse probability weighting                                                      |
|       -Sensitivity analysis tools                                                         |
|___________________________________________________________________________________________|

Contents:
  zepid 
    |
    |-RelRisk(): calculate risk ratio from pandas dataframe
    |-RiskDiff(): calculate risk difference from pandas dataframe
    |-NNT(): calcualte the number needed to treat from pandas dataframe
    |-OddsRatio(): calculate the odds ratio from pandas dataframe
    |-IncRateDiff(): calculate the incidence rate difference from pandas dataframe
    |-IncRateRatio(): calculate the incidence rate ratio from pandas dataframe
    |-IC(): calculate the interaction contrast from pandas dataframe
    |-ICR(): calculate the interaction contrast ratio from pandas dataframe
    |-ACR(): calculate attributable community risk from pandas dataframe
    |-PAF(): calculate population attributable fraction from pandas dataframe
    |-StandMeanDiff(): calculate the standardized mean difference from pandas dataframe
    |-Sensitivity(): calculate sensitivity from pandas dataframe
    |-Specificity(): calculate specificity from pandas dataframe
    |-spline(): generate spline terms for continuous variable in pandas dataframe
    |
    |___calc
    |    |
    |    |-rr(): calculate risk ratio from summary data 
    |    |-rd(): calculate risk difference from summary data 
    |    |-nnt(): calculate number needed to treat from summary data 
    |    |-oddsratio(): calculate odds ratio from summary data 
    |    |-ird(): calculate incidence rate difference from summary data 
    |    |-irr(): calculate incidence rate ratio from summary data 
    |    |-acr(): calculate attributable community risk from summary data 
    |    |-paf(): calculate population attributable fraction from summary data 
    |    |-risk_ci(): calculate risk confidence interval
    |    |-ir_ci(): calculate incidence rate confidence interval
    |    |-stand_mean_diff(): calculate standardized mean difference
    |    |-odds_to_prop(): convert odds to proportion
    |    |-prop_to_odds(): convert proportion to odds
    |    |-ppv_conv(): calculate positive predictive value
    |    |-npv_conv(): calculate negative predictive value
    |    |-screening_cost_analyzer(): calculate relative costs of screening program
    |    |-counternull_pvalue(): calculate counternull p-value
    |
    |___graphics
    |    |
    |    |-func_form_plot(): generate a functional form plot
    |    |-effectmeasure_plot(): create an effect measure plot
    |           |-labels(): change the labels, scale, reference line for plot
    |           |-colors(): change the colors and point shapes for plot
    |           |-plot(): generate the effect measure plot 
    |
    |___ipw
    |    |
    |    |-ipw(): generate probabilities/propensity scores via logistic regression
    |    |__iptw(): class for inverse probability of treament weights
    |    |     |-weight(): generate IPT weights 
    |    |     |-merge_weights(): merge weights from another IPW model
    |    |     |-fit(): fit an IPTW model via GEE or weighted Kaplan Meier
    |    |
    |    |__ipmw(): class for inverse probability of missing weights
    |    |     |-weight(): generate IPM weights
    |    |     |-merge_weights(): merge weights from another IPW model
    |    |     |-fit(): fit an IPMW model via GEE or weighted Kaplan Meier
    |    |  
    |    |__ipcw(): class for inverse probability of censoring weights
    |    |     |-longdata_converter(): convert a wide survival dataset to a long format
    |    |     |-weight(): generate IPC weights 
    |    |     |-merge_weights(): merge weights from another IPW model
    |    |     |-fit(): fit an IPCW model via weighted Kaplan-Meier
    |    |
    |    |__diagnostic
    |         |-p_boxplot():generate boxplot of probabilities by exposure
    |         |-p_hist(): generates histogram of probabilities by exposure
    |         |-positivity(): diagnostic values for positivity issues
    |         |-standardized_diff(): calculates the standardized differences of IP weights
    |    
    |___sens_analysis
         |
         |-rr_corr(): generates a corrected RR based on RR of confounder and probabilities confounder
         |-trapezoidal(): generates a trapezoidal distribution of values
         |-delta_beta(): conducts a delta-beta analysis
'''
from .base import RelRisk,RiskDiff,NNT,OddsRatio,IncRateRatio,IncRateDiff,ACR,PAF,IC,ICR,Sensitivity,Specificity,spline,StandMeanDiff

from .calc import calc
from .graphics import graphics 
from .ipw import ipw
from .sens_analysis import sens_analysis
