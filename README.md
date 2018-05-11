![zepid](docs/images/zepid_logo.png)
# zepid

If you have any requests for items to be included, please contact me and I will work on adding any requested features. You can contact me either through github (https://github.com/pzivich), email (gmail: zepidpy), or twitter (@zepidpy).

zepid is an epidemiology analysis package, providing easy to use tools for epidemiologists coding in python3. The purpose of this package is to provide a toolset to make epidemiology e-z. A variety of calculations and plots can be generated through various functions. For a sample walkthrough of what this package is capable of, please look to the introduction to Python 3 for epidemiologists at https://github.com/pzivich/Python-for-Epidemiologists

Just a few highlights to provide: basic epidemiology calculations, easily create functional form assessment plots, easily create effectmeasure plots, generate and conduct diagnostic tests on inverse probability weight.

# Dependencies
pandas >= 0.18.0, numpy, statsmodels >= 0.8.0, matplotlib, scipy >= 1.0.0, lifelines >= 0.14.0, tabulate

# Module Overall Structure / Features:
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
    |
    |___graphics
    |    |
    |    |-func_form_plot(): generate a functional form plot
    |    |-effectmeasure_plot(): create an effect measure plot
    |    |      |-labels(): change the labels, scale, reference line for plot
    |    |      |-colors(): change the colors and point shapes for plot
    |    |      |-plot(): generate the effect measure plot 
    |    |-pvalue_plot(): generate a p-value distribution plot
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

# Future Inclusions:
    Augmented IPW
    Forest plot
    Mathematical Modeling tools

# TODO items for me:
-Update docs, finish verifications
