import warnings
import math 
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import family
from statsmodels.genmod.families import links
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def rr(a,b,c,d,alpha=0.05,decimal=3):
    '''Calculates the Risk Ratio from count data.
    
    WARNING: a,b,c,d must be positive numbers.
    
    a:
        -count of exposed individuals with outcome
    b:
        -count of unexposed individuals with outcome
    c:
        -count of exposed individuals without outcome
    d:
        -count of unexposed individuals without outcome
    alpha:
        -Alpha value to calculate confidence intervals. Only compatible with two-sided intervals. Default is 
         95% onfidence interval
    decimal:
        -amount of decimal points to display. Default is 3
    '''
    if ((a<=0) | (b<=0) | (c<=0) | (d<=0)):
        raise ValueError('All numbers must be positive')
    if ((a<=5) | (b<=5) | (c<=5) | (d<=5)):
        warnings.warn('At least one cell count is less than 5, therefore confidence interval approximation is invalid')
    zalpha = norm.ppf((1-alpha/2),loc=0,scale=1)
    r1=a/(a+b)
    r0=c/(c+d)
    relrisk = r1/r0
    SE=math.sqrt((1/a)-(1/(a+b))+(1/c)-(1/(c+d)))
    lnrr=math.log(relrisk)
    lcl=lnrr-(zalpha*SE)
    ucl=lnrr+(zalpha*SE)
    print('----------------------------------------------------------------------')
    print('Risk exposed:',round(r1,decimal))
    print('Risk unexposed:',round(r0,decimal))
    print('----------------------------------------------------------------------')
    print('Relative Risk:',round(relrisk,decimal))
    print(str(round(100*(1-alpha),1))+'% two-sided CI: (',round(math.exp(lcl),decimal),', ',round(math.exp(ucl),decimal),')')
    print('----------------------------------------------------------------------')
    return relrisk 


def rd(a,b,c,d,alpha=0.05,decimal=3):
    '''Calculates the Risk Difference from count data.
    
    a:
        -count of exposed individuals with outcome
    b:
        -count of unexposed individuals with outcome
    c:
        -count of exposed individuals without outcome
    d:
        -count of unexposed individuals without outcome
    alpha:
        -Alpha value to calculate confidence intervals. Only compatible with two-sided intervals. Default is 
         95% onfidence interval
    decimal:
        -amount of decimal points to display. Default is 3
    '''
    if ((a<=0) | (b<=0) | (c<=0) | (d<=0)):
        raise ValueError('All numbers must be positive')
    if ((a<=5) | (b<=5) | (c<=5) | (d<=5)):
        warnings.warn('At least one cell count is less than 5, therefore confidence interval approximation is invalid')
    zalpha = norm.ppf((1-alpha/2),loc=0,scale=1)
    r1=a/(a+b)
    r0=c/(c+d)
    riskdiff = r1-r0
    SE=math.sqrt(((a*b)/((((a+b)**2)*(a+b-1))))+((c*d)/(((c+d)**2)*(c+d-1))))
    lcl=riskdiff-(zalpha*SE)
    ucl=riskdiff+(zalpha*SE)
    print('----------------------------------------------------------------------')
    print('Risk exposed:',round(r1,decimal))
    print('Risk unexposed:',round(r0,decimal))
    print('----------------------------------------------------------------------')
    print('Risk Difference:',round(riskdiff,decimal))	
    print(str(round(100*(1-alpha),1))+'%  two-sided CI: (',round(lcl,decimal),', ',round(ucl,decimal),')')
    print('----------------------------------------------------------------------')
    return riskdiff

def nnt(a,b,c,d,alpha=0.05,decimal=3):
    '''Calculates the Number Needed to Treat from count data.
    
    a:
        -count of exposed individuals with outcome
    b:
        -count of unexposed individuals with outcome
    c:
        -count of exposed individuals without outcome
    d:
        -count of unexposed individuals without outcome
    alpha:
        -Alpha value to calculate confidence intervals. Only compatible with two-sided intervals. Default is 
         95% onfidence interval
    decimal:
        -amount of decimal points to display. Default is 3
    '''
    if ((a<=0) | (b<=0) | (c<=0) | (d<=0)):
        raise ValueError('All numbers must be positive')
    if ((a<=5) | (b<=5) | (c<=5) | (d<=5)):
        warnings.warn('At least one cell count is less than 5, therefore confidence interval approximation is invalid')
    zalpha = norm.ppf((1-alpha/2),loc=0,scale=1)
    ratiod1=a/(a+b)
    ratiod2=c/(c+d)
    riskdiff = ratiod1-ratiod2
    SE=math.sqrt(((a*b)/((((a+b)**2)*(a+b-1))))+((c*d)/(((c+d)**2)*(c+d-1))))
    lcl_rd=(riskdiff-(zalpha*SE))
    ucl_rd=(riskdiff+(zalpha*SE))
    print('----------------------------------------------------------------------')
    print('Risk Difference: ',round(riskdiff,decimal))
    print('----------------------------------------------------------------------')
    if riskdiff == 0:
        print('Number Needed to Treat = infinite')
    else:
        if riskdiff > 0:
            print('Number Needed to Harm: ',round(abs(NNT),decimal),'\n')
        if riskdiff < 0:
            print('Number Needed to Treat: ',round(abs(NNT),decimal),'\n')
    try:
        ucl = 1/lcl_rd
        lcl = 1/ucl_rd
    except:
        raise ValueError('One of the confidence limits is equal to zero, and cannot be evaluated')
    print(str(round(100*(1-alpha),1))+'% two-sided CI: ')
    if lcl_rd < 0 < ucl_rd:
        print('NNH ',round(abs(lcl),decimal),'to infinity to NNT ',round(abs(ucl),decimal))
    elif 0 < lcl_rd:
        print('NNT ',round(abs(lcl),decimal),' to ',round(abs(ucl),decimal))
    else:
        print('NNH ',round(abs(lcl),decimal),' to ',round(abs(ucl),decimal))
    print('----------------------------------------------------------------------')
    return nnt 


def oddr(a,b,c,d,alpha=0.05,decimal=3):
    '''Calculates the Odds Ratio from count data.

    a:
        -count of exposed individuals with outcome
    b:
        -count of unexposed individuals with outcome
    c:
        -count of exposed individuals without outcome
    d:
        -count of unexposed individuals without outcome
    alpha:
        -Alpha value to calculate confidence intervals. Only compatible with two-sided intervals. Default is 
         95% onfidence interval
    decimal:
        -amount of decimal points to display. Default is 3
    '''
    if ((a<=0) | (b<=0) | (c<=0) | (d<=0)):
        raise ValueError('All numbers must be positive')
    if ((a<=5) | (b<=5) | (c<=5) | (d<=5)):
        warnings.warn('At least one cell count is less than 5, therefore confidence interval approximation is invalid')
    zalpha = norm.ppf((1-alpha/2),loc=0,scale=1)
    or1=a/b
    or0=c/d
    oddsr=or1/or0
    SE=math.sqrt((1/a)+(1/b)+(1/c)+(1/d))
    lnor=math.log(oddsr)
    lcl=lnor-(zalpha*SE)
    ucl=lnor+(zalpha*SE)
    print('----------------------------------------------------------------------')
    print('Odds exposed:',round(or1,decimal))
    print('Odds unexposed:',round(or0,decimal))
    print('----------------------------------------------------------------------')
    print('Odds Ratio:',round(oddsr,decimal))
    print(str(round(100*(1-alpha),1))+'% two-sided CI: (',round(math.exp(lcl),decimal),', ',round(math.exp(ucl),decimal),')')
    print('----------------------------------------------------------------------')
    return oddsr

    
def irr(a,b,T1,T2,alpha=0.05,decimal=3):
    '''Calculates the Incidence Rate Ratio from count data.
    
    a:
        -count of exposed with outcome
    b:
        -count of unexposed with outcome
    T1:
        -person-time contributed by those who were exposed
    T2:
        -person-time contributed by those who were unexposed
    alpha:
        -Alpha value to calculate confidence intervals. Only compatible with two-sided intervals. Default is 
         95% onfidence interval
    decimal:
        -amount of decimal points to display. Default is 3
    '''
    if ((a<=0) | (b<=0) | (c<=0) | (d<=0)):
        raise ValueError('All numbers must be positive')
    zalpha = norm.ppf((1-alpha/2),loc=0,scale=1)
    irate1=a/T1
    irate2=b/T2
    irr = irate1/irate2
    SE=math.sqrt((1/a)+(1/b))
    lnirr=math.log(irr)
    lcl=lnirr-(zalpha*SE)
    ucl=lnirr+(zalpha*SE)
    print('----------------------------------------------------------------------')
    print('Incidence Rate exposed:',round(irate1,decimal))
    print('Incidence Rate unexposed:',round(irate2,decimal))
    print('----------------------------------------------------------------------')
    print('Incidence Rate Ratio:',round(irr,decimal))
    print(str(round(100*(1-alpha),1))+'% two-sided CI: (',round(math.exp(lcl),decimal),', ',round(math.exp(ucl),decimal),')')
    print('----------------------------------------------------------------------')
    return irr

def ird(a,b,T1,T2,alpha=0.05,decimal=3):
    '''Calculates the Incidence Rate Difference from count data.

    a:
        -count of exposed with outcome
    b:
        -count of unexposed with outcome
    T1:
        -person-time contributed by those who were exposed
    T2:
        -person-time contributed by those who were unexposed
    alpha:
        -Alpha value to calculate confidence intervals. Only compatible with two-sided intervals. Default is 
         95% onfidence interval
    decimal:
        -amount of decimal points to display. Default is 3
    '''
    if ((a<=0) | (b<=0) | (c<=0) | (d<=0)):
        raise ValueError('All numbers must be positive')
    zalpha = norm.ppf((1-alpha/2),loc=0,scale=1)
    rated1=a/T1
    rated2=b/T2
    ird = rated1-rated2
    SE=math.sqrt((a/((T1)**2))+(b/((T2)**2)))
    lcl=ird-(zalpha*SE)
    ucl=ird+(zalpha*SE)
    print('----------------------------------------------------------------------')
    print('Incidence Rate exposed:',round(rated1,decimal))
    print('Incidence Rate unexposed:',round(rated2,decimal))
    print('----------------------------------------------------------------------')
    print('Incidence Rate Difference:',round(ird,decimal))	
    print(str(round(100*(1-alpha),1))+'% two-sided CI: (',round(lcl,decimal),', ',round(ucl,decimal),')')
    print('----------------------------------------------------------------------')
    return ird


def acr(a,b,c,d,decimal=3):
    '''Calculates the estimated Attributable Community Risk (ACR) from count data. ACR is also 
    known as Population Attributable Risk. Since this is commonly confused with the population 
    attributable fraction, the name ACR is used to clarify differences in the formulas
    
    a:
        -count of exposed individuals with outcome
    b:
        -count of unexposed individuals with outcome
    c:
        -count of exposed individuals without outcome
    d:
        -count of unexposed individuals without outcome
    decimal:
        -amount of decimal points to display. Default is 3
    '''
    if ((a<=0) | (b<=0) | (c<=0) | (d<=0)):
        raise ValueError('All numbers must be positive')
    rt=(a+c)/(a+b+c+d)
    r0=c/(c+d)
    acr=(rt-r0)
    print('----------------------------------------------------------------------')
    print('ACR: ',round(acr,decimal))
    print('----------------------------------------------------------------------')

def paf(a,b,c,d,decimal=3):
    '''Calculates the Population Attributable Fraction from count data.
    
    a:
        -count of exposed individuals with outcome
    b:
        -count of unexposed individuals with outcome
    c:
        -count of exposed individuals without outcome
    d:
        -count of unexposed individuals without outcome
    decimal:
        -amount of decimal points to display. Default is 3
        '''
    if ((a<=0) | (b<=0) | (c<=0) | (d<=0)):
        raise ValueError('All numbers must be positive')
    rt=(a+c)/(a+b+c+d)
    r0=c/(c+d)
    paf=(rt-r0)/rt
    print('----------------------------------------------------------------------')
    print('PAF: ',round(paf,decimal))
    print('----------------------------------------------------------------------')
    

def prop_to_odds(prop):
    '''Convert proportion to odds

    prop:
        -proportion that is desired to transform into odds
    '''
    odds = prop / (1-prop)
    return odds

    
def odds_to_prop(odds):
    '''Convert odds to proportion
    
    odds:
        -odds that is desired to transform into a proportion
    '''
    prop = odds / (1+odds)
    return prop

    
def risk_ci(events,total,alpha=0.05,decimal=3):
    '''Calculate two-sided (1-alpha)% Confidence interval of Risk. Note
    relies on the Central Limit Theorem, so there must be at least 5 events 
    and 5 no events. Exact methods currently not available
    
    events:
        -Number of events/outcomes that occurred
    total:
        -total number of subjects that could have experienced the event
    alpha:
        -Alpha level. Default is 0.05
    decimal:
        -Number of decimal places to display. Default is 3 decimal places
    '''
    risk = events/total
    c = 1 - alpha/2
    zalpha = norm.ppf(c,loc=0,scale=1)
    se = math.sqrt((risk*(1-risk)) / total)
    lower = risk - zalpha*se
    upper = risk + zalpha*se
    print('Risk: ',round(risk,decimal))
    print(str(c)+'% CI: (',round(lower,decimal),', ',round(upper,decimal),')')    


def ir_ci(events,time,alpha=0.05,decimal=3):
    '''Calculate two-sided (1-alpha)% Confidence interval of Incidence Rate
        
    events:
        -number of events/outcomes that occurred
    time:
        -total person-time contributed in this group
    alpha:
        -alpha level. Default is 0.05
    decimal:
        -amount of decimal places to display. Default is 3 decimal places
    '''
    a = alpha/2
    prob_l = 2*events
    prob_u = 2*(events+1)
    a_l = 1-a
    a_u = a
    chi_l = chi2.isf(q=a_l,df=prob_l);chi_u = chi2.isf(q=a_u,df=prob_u)
    lower = ((chi_l/2)/time);upper = ((chi_u/2)/time)
    print('IR: ',round((events/time),decimal))
    print('95% CI: (',round(lower,decimal),', ',round(upper,decimal),')')


def weibull_calc(lamb,time,gamma,print_result=True,decimal=3):
    '''Uses the Weibull Model to calculaive the cumulative survival 
    and cumulative incidence functions. The Weibull model assumes that
    there are NO competing risks
    
    Returns hazard, survival, and cumulative incidence estimates
    
    lamb:
        -Lambda value (scale parameter) in cases per time
    time:
        -Number of time units. Must be the same unit as the scale parameter
    gamma:
        -Gamma value (shape parameter)
    print_result:
        -Whether to print the calculated values. Default is True
    decimal:
        -Number of decimal places to display. Default is three
    '''
    import math
    if ((lamb<0) | (gamma<0)):
        raise ValueError('Lambda/Gamma parameters cannot be less than or equal to zero')
    haza = lamb*gamma*(time**(gamma-1))
    surv = math.exp(-lamb*(time**gamma))
    inci = 1 - surv
    if print_result==True:
        print('----------------------------------------------------------------------')
        if gamma > 1:
            print('h(t) rises over time')
        elif gamma < 1:
            print('h(t) declines over time')
        else:
            print('h(t) is constant over time')
        print('----------------------------------------------------------------------')
        print('Hazard: ',round(haza,decimal))
        print('Cumulative Survival: ',round(surv,decimal))
        print('Cumulative Incidence: ',round(inci,decimal))
    return haza,surv,inci


def expected_cases_weibull(lamb,time,gamma,N):
    '''Using the Weibull Model, we calculate the expected number
    of incidence cases. Note that the Weibull Model assumption of 
    no competing risks cannot be violated. All parameters must be greater
    than zero
    
    Returns the number of expected cases
        
    lamb:
        -Lambda value (scale parameter) in cases per time
    time:
        -Number of time units. Must be the same unit as the scale parameter
    gamma:
        -Gamma value (shape parameter)
    N:
        -Total population size
    '''
    import math
    if ((lamb<=0) | (gamma<=0) | (time<=0) | (N<=0)):
        raise ValueError('All parameters must be greater than zero')
    surv = math.exp(-lamb*(time**gamma))
    A = N*(1-surv)
    return A 


def expected_time_weibull(lamb,time,gamma,N,warn=True):
    '''Using the Weibull Model, we can calculate the expected person-time
    NOTE: This formula is only valid when the hazard is constant. Additionally,
    it is assumed that follow-up is long enough for everyone to be a case. These
    two assumptions reduce the Weibull Model to the exponential formula, allowing
    the calculation of expected person-time
        
    lamb:
        -Lambda value (scale parameter) in cases per time
    time:
        -Number of time units. Must be the same unit as the scale parameter
    gamma:
        -Gamma value (shape parameter). Note that gamma MUST be one for calculation
         to proceed. This is NOT defaulted, so as to remind users about the underlying
         assumption of the Weibull Model to calculate expected person-time
    N:
        -Total population size
    warn:
        -Whether to print the warning regarding the major assumption of this method.
         Default is True
    '''
    if gamma != 1:
        raise ValueError('The expected person-time calculation is only valid when gamma is one (constant hazard)')
    if ((lamb<=0) | (time<=0) | (N<=0)):
        raise ValueError('All parameters must be greater than zero')
    if warn==True:
        print('----------------------------------------------------------------------')
        print('WARNING: This method makes the strong assumption that hazards are \nconstant. This may NOT be a valid assumption')
        print('----------------------------------------------------------------------')
    T = N/lamb
    return T 
    

def counternull_pvalue(estimate,lcl,ucl,sided='two',alpha=0.05,decimal=3):
    '''Calculates the counternull based on Rosenthal R & Rubin DB (1994). It is useful
    to prevent over-interpretation of results. For a full discussion and how to interpret
    the estimate and p-value, 
    
    Warning: Make sure that the confidence interval points put into
    the equation match the alpha level calculation
    
    estimate:
        -Point estimate for result
    lcl:
        -Lower confidence limit
    ucl:
        -Upper confidence limit
    sided:
        -Whether to compute the upper one-sided, lower one-sided, or two-sided counternull
         p-value. Default is the two-sided
            'upper'     Upper one-sided p-value
            'lower'     Lower one-sided p-value
            'two'       Two-sided p-value
    alpha:
        -Alpha level for p-value. Default is 0.05. Verify that this is the same alpha used to
         generate confidence intervals
    decimal:
        -Number of decimal places to display. Default is three
    '''
    zalpha = norm.ppf((1-alpha/2),loc=0,scale=1)
    se = (ucl - lcl) / (zalpha*2)
    cnull = 2*estimate
    up_cn = norm.cdf(x=cnull,loc=estimate,scale=se)
    lp_cn = 1 - up_cn
    lowerp = norm.cdf(x=estimate,loc=cnull,scale=se)
    upperp = 1 - lowerp
    twosip = 2 * min([up_cn,lp_cn])
    print('----------------------------------------------------------------------')
    print('Alpha = ',alpha)
    print('----------------------------------------------------------------------')
    print('Counternull estimate = ',cnull)
    if sided=='upper':
        print('Upper one-sided counternull p-value: ',round(upperp,decimal))
    elif sided=='lower':
        print('Lower one-sided counternull p-value: ',round(lowerp,decimal))
    else:
        print('Two-sided counternull p-value: ',round(twosip,decimal))
    print('----------------------------------------------------------------------')


def bayes_approx(prior_mean,prior_lcl,prior_ucl,mean,lcl,ucl,ln_transform=False,alpha=0.05,decimal=3):
    '''A simple Bayesian Analysis. Note that this analysis assumes normal distribution for the 
    continuous measure. 
    
    Warning: Make sure that the alpha used to generate the confidence intervals matches the alpha
    used in this calculation
    
    prior_mean:
        -Prior designated point estimate
    prior_lcl:
        -Prior designated lower confidence limit
    prior_ucl:
        -Prior designated upper confidence limit
    mean:
        -Point estimate result obtained from analysis
    lcl:
        -Lower confidence limit estimate obtained from analysis
    ucl:
        -Upper confidence limit estimate obtained from analysis
    ln_transform:
        -Whether to natural log transform results before conducting analysis. Should be used for 
         RR, OR, or or other Ratio measure. Default is False (use for RD and other absolute measures)
    alpha: 
        -Alpha level for confidence intervals. Default is 0.05
    decimal:
        -Number of decimal places to display. Default is three
    '''
    if ln_transform==True:
        prior_mean = math.log(prior_mean)
        prior_lcl = math.log(prior_lcl)
        prior_ucl = math.log(prior_ucl)
        mean = math.log(mean)
        lcl = math.log(lcl)
        ucl = math.log(ucl)
    zalpha = norm.ppf((1-alpha/2),loc=0,scale=1)
    prior_sd = (prior_ucl - prior_lcl) / (2*zalpha)
    prior_var = prior_sd**2
    prior_w = 1 / prior_var
    sd = (ucl - lcl) / (2*zalpha)
    var = sd**2
    w = 1 / var 
    post_mean = ((prior_mean*prior_w)+(mean*w)) / (prior_w + w)
    post_var = 1 / (prior_w + w)
    sd = math.sqrt(post_var)
    post_lcl = post_mean - zalpha*sd 
    post_ucl = post_mean + zalpha*sd
    if ln_transform==True:
        post_mean = math.exp(post_mean)
        post_lcl = math.exp(post_lcl)
        post_ucl = math.exp(post_ucl)
        prior_mean = math.exp(prior_mean)
        prior_lcl = math.exp(prior_lcl)
        prior_ucl = math.exp(prior_ucl)
        mean = math.exp(mean)
        lcl = math.exp(lcl)
        ucl = math.exp(ucl)
    print('----------------------------------------------------------------------')
    print('Prior Estimate: ',round(prior_mean,decimal))
    print(str(round((1-alpha)*100,1))+'% Prior Confidence Interval: (',round(prior_lcl,decimal),', ',round(prior_ucl,decimal),')')
    print('----------------------------------------------------------------------')
    print('Point Estimate: ',round(mean,decimal))
    print(str(round((1-alpha)*100,1))+'% Confidence Interval: (',round(lcl,decimal),', ',round(ucl,decimal),')')
    print('----------------------------------------------------------------------')
    print('Posterior Estimate: ',round(post_mean,decimal))
    print(str(round((1-alpha)*100,1))+'% Posterior Probability Interval: (',round(post_lcl,decimal),', ',round(post_ucl,decimal),')')
    print('----------------------------------------------------------------------')


def ppv_conv(sensitivity,specificity,prevalence):
    '''Generates the Positive Predictive Value from designated
    Sensitivity, Specificity, and Prevalence.  

    sensitivity:
        -sensitivity of the criteria
    specificity:
        -specificity of the criteria
    prevalence:
        -prevalence of the outcome in the population
    '''
    if ((sensitivity > 1)|(specificity > 1)|(prevalence > 1)):
        raise ValueError('sensitivity/specificity/prevalence cannot be greater than 1')
    sens_prev = sensitivity*prevalence
    nspec_nprev = (1-specificity)*(1-prevalence)
    ppv = sens_prev / (sens_prev + nspec_nprev)
    return ppv


def npv_conv(sensitivity,specificity,prevalence):
    '''Generates the Negative Predictive Value from designated Sensitivity, Specificity, and Prevalence.  

    sensitivity:
        -sensitivity of the criteria
    specificity:
        -specificity of the criteria
    prevalence:
        -prevalence of the outcome in the population
    '''
    if ((sensitivity > 1)|(specificity > 1)|(prevalence > 1)):
        raise ValueError('sensitivity/specificity/prevalence cannot be greater than 1')
    spec_nprev = specificity*(1-prevalence)
    nsens_prev = (1-sensitivity)*prevalence
    npv = spec_nprev / (spec_nprev + nsens_prev)
    return npv

    
def miss_count(countA,totalN,sensitivity,specificity):
    '''The formula is based on  Modern Epidemiology 3rd Edition (Rotham et al.) pg.XXX
    mis_count() totals the amount of those with outcome and not outcome
    removing the potential bias introduced by missclassification.  The
    formula returns A (count of Disease=1) and B (count of Disease=0)
        
    countA:
        -observed count of 
    totalN:
        -observed total amount of subjects
    sensitivity:
        -sensitivity of the criteria used for classification
    specificity:
        -specificity of the criteria used for classification
    '''
    if ((sensitivity > 1)|(specificity > 1)):
        raise ValueError('sensitivity/specificity/prevalence cannot be greater than 1')
    if ((countA<0)|(totalN<0)):
        raise ValueError('observed count of A and total sample size must be greater than 0')
    if (countA>totalN):
        raise ValueError('observed count of A cannot be greater than total sample size')
    A_top = countA - (1-specificity)*totalN
    A_bottom = sensitivity + specificity - 1
    A = A_top / A_bottom
    B = totalN - A
    return A,B


def screening_cost_analyzer(cost_miss_case,cost_false_pos,prevalence,sensitivity,specificity,population=10000,decimal=3):
    '''Compares the cost of sensivitiy/specificity of screening criteria to treating the entire population 
    as test-negative and test-positive. The lowest per capita cost is considered the ideal choice. Note that
    this function only provides relative costs. However, actual costs can be put in as well
        
    WARNING: When calculating costs, be sure to consult experts in health policy or related fields.  Costs 
    should encompass more than just monetary costs, like relative costs (regret, disappointment, stigma, 
    disutility, etc.). Careful consideration of relative costs between false positive and false negatives
    needs to be considered.
    
    cost_miss_case:
        -The relative cost of missing a case, compared to false positives. In general, set this to 1 then 
         change the value under 'cost_false_pos' to reflect the relative cost
    cost_false_pos:
        -The relative cost of a false positive case, compared to a missed case
    prevalence:
        -The prevalence of the disease in the population. Must be a float
    sensitivity:
        -The sensitivity level of the screening test. Must be a float
    specificity:
        -The specificity level of the screening test. Must be a float
    population:
        -The population size to set. Choose a larger value since this is only necessary for total calculations. Default is 10,000
    decimal:
        -amount of decimal points to display. Default value is 3
    '''
    print('----------------------------------------------------------------------')
    print('WARNING: When calculating costs, be sure to consult experts in health\npolicy or related fields.  Costs should encompass more than only monetary\ncosts, like relative costs (regret, disappointment, stigma, disutility, etc.)')
    if (sensitivity>1) | (specificity>1):
        raise ValueError('sensitivity/specificity/prevalence cannot be greater than 1')
    disease = population*prevalence
    disease_free = population - disease
    nt_cost = disease * cost_miss_case
        #per capita cost 
    pc_nt_cost = nt_cost/population
    #TEST: all
        #total cost of testing
    t_cost = disease_free * cost_false_pos
        #per capita cost 
    pc_t_cost = t_cost/population
    #TEST: criteria
    cost_b = disease - (disease*sensitivity)
    cost_c = disease_free - (disease_free*specificity)
    ct_cost = (cost_miss_case*cost_b) + (cost_false_pos*cost_c)
    pc_ct_cost = ct_cost/population
    print('----------------------------------------------------------------------')
    print('Treat everyone as Test-Negative')
    print('Total relative cost:\t\t',round(nt_cost,decimal))
    print('Per Capita relative cost:\t',round(pc_nt_cost,decimal))
    print('----------------------------------------------------------------------')
    print('Treat everyone as Test-Positive')
    print('Total relative cost:\t\t',round(t_cost,decimal))
    print('Per Capita relative cost:\t',round(pc_t_cost,decimal))
    print('----------------------------------------------------------------------')
    print('Treating by Screening Test')
    print('Total relative cost:\t\t',round(ct_cost,decimal))
    print('Per Capita relative cost:\t',round(pc_ct_cost,decimal))
    print('----------------------------------------------------------------------')
    if (pc_ct_cost>pc_nt_cost):
        print('Screening program is more costly than treating everyone as a test-negative')
    if (pc_nt_cost>pc_ct_cost>pc_t_cost):
        print('Screening program is cost efficient')
    if ((pc_t_cost<pc_ct_cost)&(pc_t_cost<pc_nt_cost)):
        print('Treating everyone as test-positive is least costly')
    print('----------------------------------------------------------------------')



def stand_mean_diff(n1,n2,mean1,mean2,sd1,sd2,decimal=3):
    '''Calculates the standardized mean difference (SMD) of a continuous variable
    stratified by a binary variable. This can be used to assess for 
    collinearity between the continuous and binary variables of interest.
    A SMD greater than 2 suggests potential collinearity issues. It does NOT
    mean that there will be collinearity issues in the full model though. This 
    functions works for summary data
    
    n1:
        -number of observations in the first group stratified by variable
    n2:
        -number of observations in the second group stratified by variable
    mean1:
        -mean of the continuous variable in the first group stratified by categorical
         variable
    mean2:
        -mean of the continuous variable in the second group stratified by categorical
         variable
    sd1:
        -standard deviation in the first group stratified by variable
    sd2:
        -standard deviation in the second group stratified by variable
    decimal:
        -Number of decimal places to display. Default is 3
    '''
    pooled_sd = math.sqrt((((n1 - 1) * (sd1**2)) + ((n2 - 1) * (sd2**2))) / n1 + n2 - 2)
    smd = abs(((mean1 - mean2) / pooled_sd))
    print('----------------------------------------------------------------------')
    print('Standardized Mean Difference: '+str(round(smd,decimal)))
    print('----------------------------------------------------------------------')


