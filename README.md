# zepid
zepid is an epidemiology analysis package, providing easy to use tools for epidemiologists coding in python3. Basic calculations are provided for both summary data and directly with raw data. Many of the more complicated package elements (ex. delta beta sensitivity analysis) are still underdevelopment. Basic measures of associations are fully implemented but still require some testing. Measures of association have been compared to SAS v9.4 to validate calculations. 

Note: regarding confidence intervals, only Wald confidence intervals are currently implemented. Currently, there are no plans to implement other types of confidence intervals. 


# Contents include:
    Association measures:
        -Measures of association, directly on dataframe (RR/RD/OR/NNT/IRR/IRD/ACR/PAF)
        -Measures of association, on summary data (RR/RD/OR/NNT/IRR/IRD/ACR/PAF)
        -Interaction Contrast & Interaction Contrast Ratio
    Inverse Probability Weighting
        -Inverse probability for missing (IPMW)
        -Inverse probability for treatment (IPTW) 
            -Standardized & unstandardized weights
        -IPW weight diagnostic tools
            -Graphical displays (histogram, boxplot)
            -Standardized differences (binary, continuous)
            -Positivity assessment
        -Fit IPW via GEE with independent covariance structure
    Screening Measures:
        -Screening measures (sensitivity/specificity/NPV/PPV)
    Sensitivity analysis tools:
        -E-value calculation (Vanderweele and Ding 2017)
        -Crude corrected RR
        -Delta Beta analysis
    Graphics:
        -Effect measure plot generator (akin to forest plot)
    Other various calculations:
        -Convert between proportions and odds
        -Risk (probability) confidence interval
        -Incidence rate confidence interval
        -Standardized mean difference
        -Weibull Model estimation of survival/incidence/expected cases/expected person-time
        -Counternull & Counternull P-value
        -Simple Bayesian Analysis

# Current items that will be added:
**Note:** Some of these, like the causal inference tools, are going to take awhile to be implemented

    Association measures:
        -Causal inference tools (https://www.hsph.harvard.edu/causal/software/) 
    IPW:
        -Inverse probability of censoring weights (IPCW)
        -Additional diagnostic tools
        -Propensity score matching algorithms
    Graphics:
        -Forest plot
        -Functional form assessment plots
    Various other items:
        -Life Table 
        -Spline
        -Simulation method to estimate expected person-time at risk

A full guide of the package and a sample dataframe are underdevelopment. Please see my other repository ('Python-for-Epidemiologists') for more information

If you have any requests for items to be included, please contact me and I will work on adding any requested features. 
