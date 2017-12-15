import math
import numpy as np
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import family
from statsmodels.genmod.families import links

def InteractContrast(df,outcome,exposure,modifier,adjust='',decimal=5):
    '''Calculate the Interaction Contrast (IC) using a pandas dataframe and
    statsmodels to fit a linear binomial regression. Can ONLY be used for 
    a 0,1 coded exposure and modifier (exposure = {0,1}, modifier = {0,1}, 
    outcome = {0,1}). Can handle missing data and adjustment for other confounders
    in the regression model. Prints the fit of the linear binomial regression,
    the IC, and the corresponding IC 95% confidence interval.
    
    NOTE: statsmodels may produce a domain error in some versions. This can
    be ignored if the linear binomial model is a properly fitting model
    
    df: 
        -pandas dataframe object containing all variables of interest
    outcome:
        -str object of the outcome variable name from pandas df. Ex) 'Outcome'
    exposure:
        -str object of the exposure variable name from pandas df. Ex) 'Exposure'
    modifier:
        -str object of the modifier variable name from pandas df. Ex) 'Modifier'
    adjust:
        -str object of other variables to adjust for in correct statsmodels format.
        Note: variables can NOT be named {E1M0,E0M1,E1M1} since this function creates
              variables with those names. Answers will be incorrect
         Ex) '+ C1 + C2 + C3 + Z'
    decimal:
        -Number of decimals to display in result. Default is 3
    '''
    df.loc[((df[exposure]==1)&(df[modifier]==1)),'E1M1'] = 1
    df.loc[((df[exposure]!=1)|(df[modifier]!=1)),'E1M1'] = 0
    df.loc[((df[exposure].isnull())|(df[modifier].isnull())),'E1M1'] = np.nan
    eq = outcome + '~'+exposure+'+'+modifier+'+E1M1' + adjust
    f = sm.families.family.Binomial(sm.families.links.identity)
    model = smf.glm(eq,df,family=f).fit()
    print(model.summary())
    ic = model.params['E1M1']
    print(ic)
    lcl = model.conf_int().loc['E1M1'][0]
    ucl = model.conf_int().loc['E1M1'][1]
    print('\n----------------------------------------------------------------------')
    print('Interaction Contrast')
    print('----------------------------------------------------------------------')
    print('\nIC:\t\t'+str(round(ic,decimal)))
    print('95% CI:\t\t('+str(round(lcl,decimal))+', '+str(round(ucl,decimal))+')')
    print('----------------------------------------------------------------------')

  

def InteractContrastRatio(df,outcome,exposure,modifier,adjust='',regression='log',ci='delta',b_sample=1000,alpha=0.05,decimal=5):
    '''Calculate the Interaction Contrast Ratio (ICR) using a pandas dataframe, and 
    conducts either log binomial or logistic regression through statsmodels. Can ONLY be 
    used for a 0,1 coded exposure and modifier (exposure = {0,1}, modifier = {0,1}, 
    outcome = {0,1}). Can handle missing data and adjustment for other confounders
    in the regression model. Prints the fit of the binomial regression,
    the ICR, and the corresponding ICR confidence interval. Confidence intervals can be 
    generated using the delta method or bootstrap method
    
    NOTE: statsmodels may produce a domain error for log binomial models in some 
    versions. This can be ignored if the log binomial model is a properly fitting model
    
    df: 
        -pandas dataframe object containing all variables of interest
    outcome:
        -str object of the outcome variable name from pandas df. Ex) 'Outcome'
    exposure:
        -str object of the exposure variable name from pandas df. Ex) 'Exposure'
    modifier:
        -str object of the modifier variable name from pandas df. Ex) 'Modifier'
    adjust:
        -str object of other variables to adjust for in correct statsmodels format.
        Note: variables can NOT be named {E1M0,E0M1,E1M1} since this function creates
              variables with those names. Answers will be incorrect
         Ex) '+ C1 + C2 + C3 + Z'
    regression:
        -Type of regression model (and relative measure) to estimate. Default is log binomial.
         Options include:
            'log':      Log-binomial model. Estimates the Relative Risk (RR)
            'logit':    Logistic (logit) model. Estimates the Odds Ratio (OR). Note, this is 
                        only valid when the OR approximates the RR 
    ci:
        -Type of confidence interval to return. Default is the delta method. Options include:
            'delta':      Delta method as described by Hosmer and Lemeshow (1992)
            'bootstrap':  bootstrap method (Assmann et al. 1996). Depending on the number of 
                          resampling requested and the sample size, can take a long time. 
                          The delta method is more time efficient
    b_sample:
        -Number of times to resample to generate bootstrap confidence intervals. Only important
         if bootstrap confidence intervals are requested. Default is 1000
    alpha:
        -Alpha level for confidence interval. Default is 0.05
    decimal:
        -Number of decimal places to display in result. Default is 3
    '''
    df.loc[((df[exposure]==1)&(df[modifier]==0)),'E1M0'] = 1
    df.loc[((df[exposure]!=1)|(df[modifier]!=0)),'E1M0'] = 0
    df.loc[((df[exposure].isnull())|(df[modifier].isnull())),'E1M0'] = 0
    df.loc[((df[exposure]==0)&(df[modifier]==1)),'E0M1'] = 1
    df.loc[((df[exposure]!=0)|(df[modifier]!=1)),'E0M1'] = 0
    df.loc[((df[exposure].isnull())|(df[modifier].isnull())),'E0M1'] = 0
    df.loc[((df[exposure]==1)&(df[modifier]==1)),'E1M1'] = 1
    df.loc[((df[exposure]!=1)|(df[modifier]!=1)),'E1M1'] = 0
    df.loc[((df[exposure].isnull())|(df[modifier].isnull())),'E1M1'] = np.nan
    model = outcome + ' ~ E1M0 + E0M1 + E1M1' + adjust
    if regression == 'logit':
        f = sm.families.family.Binomial(sm.families.links.logit)
        print('Note: Using the Odds Ratio to calculate the ICR is only valid when\nthe OR approximates the RR')
    elif regression == 'log':
        f = sm.families.family.Binomial(sm.families.links.log)
    eq = outcome + ' ~ E1M0 + E0M1 + E1M1' + adjust
    model = smf.glm(eq,df,family=f).fit()
    print(model.summary())
    em10 = math.exp(model.params['E1M0'])
    em01 = math.exp(model.params['E0M1'])
    em11 = math.exp(model.params['E1M1'])
    em_expect = em10 + em01 - 1
    icr = em11 - em_expect
    zalpha = norm.ppf((1-alpha/2),loc=0,scale=1)
    if ci == 'delta':
        cov_matrix = model.cov_params()
        vb10 = cov_matrix.loc['E1M0']['E1M0']
        vb01 = cov_matrix.loc['E0M1']['E0M1']
        vb11 = cov_matrix.loc['E1M1']['E1M1']
        cvb10_01 = cov_matrix.loc['E1M0']['E0M1']
        cvb10_11 = cov_matrix.loc['E1M0']['E1M1']
        cvb01_11 = cov_matrix.loc['E0M1']['E1M1']
        varICR = (((em10**2)*vb10)+((em01**2)*vb01)+((em11**2)*vb11)+((em10*em01*2*cvb10_01))+(-1*em10*em11*2*cvb10_11)+(-1*em01*em11*2*cvb01_11))
        icr_lcl = icr - zalpha*math.sqrt(varICR)
        icr_ucl = icr + zalpha*math.sqrt(varICR)
    elif ci == 'bootstrap':
        bse_icr = []
        ul = 1 - alpha/2
        ll = 0 + alpha/2
        for i in range(b_sample):
            dfs = df.sample(n=len(df),replace=True)
            try:
                bmodel = smf.glm(eq,dfs,family=f).fit()
                em_bexpect = math.exp(bmodel.params['E1M0']) + math.exp(bmodel.params['E0M1']) - 1
                bicr = math.exp(bmodel.params['E1M1']) - em_bexpect
                sigma = bicr - icr
                bse_icr.append(sigma)
            except:
                bse_icr.append(np.nan)
        bsdf = pd.DataFrame()
        bsdf['sigma'] = bse_icr
        lsig,usig = bsdf['sigma'].dropna().quantile(q=[ll,ul])
        icr_lcl = lsig + icr
        icr_ucl = usig + icr
    else:
        raise ValueError('Please specify a supported confidence interval type')
    print('\n----------------------------------------------------------------------')
    if regression == 'logit':
        print('ICR based on Odds Ratio\t\tAlpha = '+str(alpha))
        print('Note: Using the Odds Ratio to calculate the ICR is only valid when\nthe OR approximates the RR')
    elif regression == 'log':
        print('ICR based on Risk Ratio\t\tAlpha = '+str(alpha))
    print('\nICR:\t\t'+str(round(icr,decimal)))
    print('CI:\t\t('+str(round(icr_lcl,decimal))+', '+str(round(icr_ucl,decimal))+')')
    print('----------------------------------------------------------------------')
