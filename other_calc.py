#####################################################################
# Random calculations
#####################################################################
import math

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
    from scipy.stats import norm
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


def StandMeanDiff(df,binary,continuous,decimal=3):
    '''Calculates the standardized mean difference (SMD) of a continuous variable
    stratified by a binary variable. This can be used to assess for 
    collinearity between the continuous and binary variables of interest.
    A SMD greater than 2 suggests potential collinearity issues. It does NOT
    mean that there will be collinearity issues in the full model though.
    
    df:
        -specify the dataframe that contains the variables of interest
    binary:
        -binary variable of interest. Input can be string or number
    continuous:
        -continuous variable of interest. Can be discrete or continuous.
         Input can be string or number
    decimal:
        -Number of decimal places to display. Default is 3
    '''
    #divide dataframe by binary variable
    v0 = df.loc[df[binary]==0]
    v1 = df.loc[df[binary]==1]
    #calculate mean and standard deviation of continuous variable, by binary variable
    m0 = np.mean(v0[continuous])
    m1 = np.mean(v1[continuous])
    sd0 = np.std(v0[continuous])
    sd1 = np.std(v1[continuous])
    #calculate the pooled standard deviation
    pooled_sd = math.sqrt((((len(v0) - 1) * (sd0**2)) + ((len(v1) - 1) * (sd1**2))) / (len(v0) + len(v1) - 2))
    #calculate the SMD
    smd = abs(((m0 - m1) / pooled_sd))
    print('Standardized Mean Difference: '+str(round(smd,decimal)))

    
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
    import math
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
    import numpy as np
    from scipy.stats import norm
    import math
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
    from scipy.stats import norm
    import math
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
        post_mean = math.log(post_mean)
        post_lcl = math.log(post_lcl)
        post_ucl = math.log(post_ucl)
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
