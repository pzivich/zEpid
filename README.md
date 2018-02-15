# zepid
zepid is an epidemiology analysis package, providing easy to use tools for epidemiologists coding in python3. The purpose of this package is to make epidemiology in Python E-Z. A variety of calculations and plots can be generated through various functions. Just a few highlights to provide; easily create functional form assessment plots, easily generate and conduct diagnostic tests on inverse probability of treatment weights, and generate a summary graphic and table
For a sample walkthrough of what this package is capable of, please look to the walkthrough.md 

# Dependencies
pandas >= 0.18.0, numpy, statsmodels >= 0.8.0, matplotlib, scipy >= 1.0.0, lifelines >= 0.14.0, tabulate

# Module Structure:
    |
    |-RelRisk(): calculate risk ratio from pandas dataframe (verified with SAS 9.4)
    |-RiskDiff(): calculate risk difference from pandas dataframe (verified with SAS 9.4)
    |-NNT(): calcualte the number needed to treat from pandas dataframe (verified with SAS 9.4)
    |-OddsRatio(): calculate the odds ratio from pandas dataframe (verified with SAS 9.4)
    |-IncRateDiff(): calculate the incidence rate difference from pandas dataframe (verified with SAS 9.4)
    |-IncRateRatio(): calculate the incidence rate ratio from pandas dataframe (verified with SAS 9.4)
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
    |    |-rr(): calculate risk ratio from summary data (verified with SAS 9.4)
    |    |-rd(): calculate risk difference from summary data (verified with SAS 9.4)
    |    |-nnt(): calculate number needed to treat from summary data (verified with SAS 9.4)
    |    |-oddr(): calculate odds ratio from summary data (verified with SAS 9.4)
    |    |-ird(): calculate incidence rate difference from summary data (verified with SAS 9.4)
    |    |-irr(): calculate incidence rate ratio from summary data (verified with SAS 9.4)
    |    |-acr(): calculate attributable community risk from summary data 
    |    |-paf(): calculate population attributable fraction from summary data 
    |    |-risk_ci(): calculate risk confidence interval (verified with SAS 9.4)
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

# Future Inclusions:
**Note:** Some of these, like the causal inference tools, are going to take awhile to be implemented

    Causal inference tools (https://www.hsph.harvard.edu/causal/software/) 
    Propensity score matching algorithms
    Forest plot
    Network analysis tools to complement NetworkX functions
    Mathematical Modeling tools

A full guide of the package and a sample dataframe are underdevelopment. Please see my other repository ('Python-for-Epidemiologists') for more information. For examples before the guide is uploaded, please see zepid_tutorial in documents and the walkthrough.md for some general guides

If you have any requests for items to be included, please contact me and I will work on adding any requested features. You can contact me either through github, email, or twitter (@zepidpy).

#To do:
-Update docs
