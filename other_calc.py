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
