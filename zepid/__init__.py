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
    |-survival_lower_upper(): generate pandas dataframes for upper/lower survival analysis
    |
    |___calc
    |    |
    |    |-rr(): calculate risk ratio from summary data 
    |    |-rd(): calculate risk difference from summary data 
    |    |-nnt(): calculate number needed to treat from summary data 
    |    |-oddr(): calculate odds ratio from summary data 
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
    |    |-miss_count(): 
    |    |-weibull_calc(): 
    |    |-expected_cases_weibull():
    |    |-expected_time_weibull():
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
    |    |-ipw():
    |    |-iptw():
    |    |-ipmw():
    |    |-ipcw_data_converter():
    |    |-ipcw(): 
    |    |-ipw_fit():
    |    |-ipcw_fit(): 
    |    |__diagnostic
    |         |-p_boxplot():
    |         |-p_hist():
    |         |-positivity():
    |         |-standardized_diff():
    |         |-weighted_avg():
    |         |-weighted_std():
    |    
    |___sens_analysis
         |
         |-rr_corr():
         |-trapezoidal():
         |-delta_beta():
'''
from .base import RelRisk,RiskDiff,NNT,OddsRatio,IncRateRatio,IncRateDiff,ACR,PAF,IC,ICR,Sensitivity,Specificity,spline,datex,StandMeanDiff,survival_upper_lower

from .calc import calc
from .graphics import graphics 
from .ipw import ipw
from .sens_analysis import sens_analysis
