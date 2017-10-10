#####################################################################
# Random calculations
#####################################################################
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
    

def risk_ci(events,total):
    '''Calculate 95% Confidence interval of Risk'''
    return


def ir_ci(events,time,alpha=0.05,decimal=3):
    '''Calculate two-sided (1-alpha)% Confidence interval of Incidence Rate
    
    events:
        -number of events/outcomes that occurred
    time:
        -total person-time contributed in this group
    alpha:
        -alpha level. Default is 0.05
    decimal:
        -amount of decimal places to display. Default is three decimal places
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

