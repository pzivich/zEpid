#####################################################################
# Sensitivity Analysis
#####################################################################

#Basic sensitivity analyses for epidemiology

#####################################################################
#Crude RR corrected by unmeasured confounder
def rr_corr(rr_obs,rr_conf,p1,p0):
    '''Simple Sensitivity analysis calculator for Risk Ratios. Estimates the impact of 
    an unmeasured confounder on the results of a conducted study. Observed RR comes from 
    the data analysis, while the RR between the unmeasured confounder and the outcome should
    be obtained from prior literature or constitute an reasonable guess. Probability of exposure
    between the groups should also be reasonable numbers. This function can be adapted to be part
    of a sensitivity analysis with repeated sampling of rr_conf, p1, and p0 from set distributions
    to obtain a range of effects. See online example for further example of this analysis. A 
    created function for the repeated sampling is not available for this since the type of 
    distribution and values to include as reasonable should be careully considered by the researcher.
    To encourage careful thought, a function for this analysis has not been created, but examples of 
    the implementation are available to help researchers create their own. 
    
    rr_obs:
        -Observed RR from the data
    rr_conf:
        -Value of RR between unmeasured confounder and the outcome of interest
    p1:
        -Estimated proportion of those with unmeasured confounder in the exposed group
    p0:
        -Estimated porportion of those with unmeasured confounder in the unexposed group
    '''
    denom = (p1*(rr_conf-1)+1) / (p0*(rr_conf-1)+1)
    rr_adj = rr_obs / denom
    return rr_adj

