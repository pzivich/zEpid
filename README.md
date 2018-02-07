# zepid
zepid is an epidemiology analysis package, providing easy to use tools for epidemiologists coding in python3. The purpose of this package is to make epidemiology in Python E-Z. A variety of calculations and plots can be generated through various functions. Just a few highlights to provide; easily create functional form assessment plots, easily generate and conduct diagnostic tests on inverse probability of treatment weights, and generate a summary graphic and table
For a sample walkthrough of what this package is capable of, please look to the walkthrough.md 

Measures of association and IPW have been compared to SAS v9.4

Note: regarding confidence intervals, only Wald confidence intervals are currently implemented. Currently, there are no plans to implement other types of confidence intervals. 

# Dependencies
pandas >= 0.21.0, numpy >= 1.13.3, statsmodels >= 0.8.0, matplotlib >= 2.1.0, scipy >= 1.0.0
###### Note that earlier versions of these modules may be okay to use. When this package was built, these were the versions of the modules used

# Module Structure:
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
         |-e_value():
         |-e_value_difference():
         |-e_value_RD():


# Future Inclusions:
**Note:** Some of these, like the causal inference tools, are going to take awhile to be implemented

    Causal inference tools (https://www.hsph.harvard.edu/causal/software/) 
    Propensity score matching algorithms
    Forest plot
    Life Table 
    Network analysis tools to complement NetworkX functions
    Mathematical Modeling tools

A full guide of the package and a sample dataframe are underdevelopment. Please see my other repository ('Python-for-Epidemiologists') for more information. For examples before the guide is uploaded, please see zepid_tutorial in documents and the walkthrough.md for some general guides

If you have any requests for items to be included, please contact me and I will work on adding any requested features. 
