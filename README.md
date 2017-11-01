# zepid
zepid is an epidemiology analysis package, providing easy to use tools for epidemiologists coding in python3. Basic calculations are provided for both summary data and directly with raw data. Many of the more complicated package elements (ex. delta beta sensitivity analysis) are still underdevelopment. Basic measures of associations are fully implemented but still require some testing. Measures of association have been compared to SAS v9.4 to validate calculations. 

Note: regarding confidence intervals, only Wald confidence intervals are currently implemented. Currently, there are no plans to implement other types of confidence intervals. 


# Contents include:
    -Measures of association, directly on dataframe
    -Measures of association, on summary data
    -Screening measures (sensitivity, specificity, NPV, PPV, etc.)
    -Sensitivity analysis tools
    -Effect measure plot generator (alternative to a large table of results)
    -Other various calculations

# Current items to be added still:
    -ICR and IC calculations (summary & df)
    -Confidence interval calculations for risk
    -Delta beta function for sensitivity analysis (using statsmodel package)
    -Plot for functional form assessment of continuous variables (using statsmodel package)
    -g-formula
    -Marginal Structural Models

A full guide of the package and a sample dataframe are underdevelopment. If you have any requests for items to be included, please contact me and I will work on adding any requested features. 
