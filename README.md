# zepid
zepid is an epidemiology analysis package, providing easy to use tools for epidemiologists coding in python3. The purpose of this package is to make epidemiology in Python E-Z. A variety of calculations and plots can be generated through various functions. Just a few highlights to provide; easily create functional form assessment plots, easily generate and conduct diagnostic tests on inverse probability of treatment weights, and generate a summary graphic and table
For a sample walkthrough of what this package is capable of, please look to the walkthrough.md 

Measures of association and IPW have been compared to SAS v9.4

Note: regarding confidence intervals, only Wald confidence intervals are currently implemented. Currently, there are no plans to implement other types of confidence intervals. 

# Dependencies
pandas >= 0.21.0, numpy >= 1.13.3, statsmodels >= 0.8.0, matplotlib >= 2.1.0, scipy >= 1.0.0
###### Note that earlier versions of these modules may be okay to use. When this package was built, these were the versions of the modules used

# Module Structure:
    zepid
        RelRisk: calculate the relative risk from a pandas dataframe
        RiskDiff: calculate the risk difference from a pandas dataframe
        NNT: calculate the number needed to treat from a pandas dataframe
        OddsRatio: calculate the odds ratio from a pandas dataframe
        IC: calculate the interaction contrast from a pandas dataframe
        ICR: calculate the interaction contrast ratio from a pandas dataframe
        IncRateDiff: calculate the incidence rate difference from a pandas dataframe
        IncRateRatio: calculate the incidence rate ratio from a pandas dataframe
        Sensitivity: calculate the sensivitiy from a pandas dataframe
        Specificity: calculate the specificity from a pandas dataframe
        StandMeanDiff: calculate the standardized mean difference from a pandas dataframe
        PAF: calculate the population attributable fraction from a pandas dataframe
        ACR: calculate the attributable community risk from a pandas dataframe
        spline: calculate spline model factors from a pandas dataframe
        datex: load a sample dataset for the zepid package
        
        calc
            rr: calculates relative risk from summary data
            rd: calculates risk difference from summary data
            nnt: calculates number needed to treat from summary data
            oddsr: calculates the odds ratio from summary data
            irr: calculates the incidence rate ratio from summary data
            ird: calculates the incidence rate difference from summary data
            acr: calculated attributable community risk from summary data
            paf: calculated the population attributable fraction from summary data
            risk_ci calculates the risk (probability) confidence interval from summary data
            ir_ci: calculates the incidence rate confidence interval from summary data
            prop_to_odds: converts proportion to odds
            odds_to_prop: converts odds to proportion
            stand_mean_diff: calculate the standard mean difference from summary data
            ppv_conv: convert sensitivity, specificity, and prevalence to Positive Predictive Value
            npv_conv: convert sensitivity, specificity, and prevalence to Negative Predictive Value
            screening_cost_analyzer: calculates the relative costs of a screening program
            counternull_pvalue: calculates the counternull and its p-value
            bayes_approx: simple Bayesian approximation from normal data        
            miss_count: recalculate two-by-two table to account for missclassification
            weibull_calc: calculate hazard, survival, cumulative incidence from Weibull
            expected_cases_weibull: calculate expected cases from Weibull
            expected_time_weibull: calculate expected person-time from Weibull
            
        graphics
            effect_measure_plotdata: convert lists to dataframe compatible with effect_measure_plot function
            effect_measure_plot: create effect measure plot, akin to forest plot
            func_form_plot: create a plot to assess functional form for regression of continuous variables
            
        ipw
            ipmw: generate inverse probability weights based on a variable with missing data
            iptw: generate inverse probability weights based on treatment
            ipw: generate propensity score (probability)
            ipw_fit: fit a inverse probability weighted model
            diagnostic:
                p_boxplot: generate boxplot for IP weights stratified by treatment
                p_hist: generate histogram for IP weights stratified by treatment
                positivity: for IPTW only, assess for possible positivity violations
                standard_diff: calculate the weighted standardized differences stratified by weight
                weighted_avg: background function to calculate weighted average for standard_diff
                weighted_std: background function to calculate weighted standard deviation for standard_diff
                
        sens_analysis
            delta_beta: conduct a delta-beta analysis either dropping a single observation each time or groups
            e_value: calculate E-value for RR,OR,HR
            e_value_difference: calculate E-value for continuous differences
            e_value_RD: calculate the E-value for the risk difference
            rr_corr: calculate the corrected RR by pre-specified distribution differences

# Future Inclusions:
**Note:** Some of these, like the causal inference tools, are going to take awhile to be implemented

    Causal inference tools (https://www.hsph.harvard.edu/causal/software/) 
    Inverse probability of censoring weights (IPCW)
    Additional IPW diagnostic tools
    Propensity score matching algorithms
    Forest plot
    Life Table 
    Simulation method to estimate expected person-time at risk
    Network analysis tools to complement NetworkX functions
    Mathematical Modeling tools

A full guide of the package and a sample dataframe are underdevelopment. Please see my other repository ('Python-for-Epidemiologists') for more information. For examples before the guide is uploaded, please see zepid_tutorial in documents and the walkthrough.md for some general guides

If you have any requests for items to be included, please contact me and I will work on adding any requested features. 
