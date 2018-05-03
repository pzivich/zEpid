import warnings
import math 
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import family
from statsmodels.genmod.families import links
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tabulate import tabulate
from zepid.calc.calc import rr,rd,nnt,oddsratio,ird,irr,acr,paf,stand_mean_diff

def RelRisk(df,exposure,outcome,alpha=0.05,decimal=3,print_result=True,return_result=False):
    '''Estimate of Relative Risk with a (1-alpha)*100% Confidence interval. Missing data is ignored by 
    this function. 
    
    WARNING: Exposure & Outcome must be coded as (1: yes, 0:no). Only works for binary exposures and outcomes
    
    df:
        -pandas dataframe containing variables of interest
    exposure:
        -column name of exposure variable. Must be coded as binary (0,1) where 1 is exposed
    outcome:
        -column name of outcome variable. Must be coded as binary (0,1) where 1 is the outcome of interest
    alpha:
        -Alpha value to calculate two-sided Wald confidence intervals. Default is 95% confidence interval
    decimal:
        -amount of decimal points to display. Default is 3
    print_result:
        -Whether to print the results. Default is True
    return_result:
        -Whether to return the RR as a object. Default is False
    
    Example)
    >zepid.RelRisk(df=data,exposure='exposure',outcome='outcome')
    '''
    zalpha = norm.ppf((1-alpha/2),loc=0,scale=1)
    a = len(df.loc[(df[exposure]==1)&(df[outcome]==1)])
    b = len(df.loc[(df[exposure]==1)&(df[outcome]==0)])
    c = len(df.loc[(df[exposure]==0)&(df[outcome]==1)])
    d = len(df.loc[(df[exposure]==0)&(df[outcome]==0)])
    rr(a=a,b=b,c=c,d=d,alpha=alpha,decimal=decimal,print_result=print_result,return_result=return_result)
 

def RiskDiff(df,exposure,outcome,alpha=0.05,decimal=3,print_result=True,return_result=False):
    '''Estimate of Risk Difference with a (1-alpha)*100% Confidence interval. Missing data is ignored by this 
    function. 
    
    WARNING: Exposure & Outcome must be coded as 1 and 0 for this to work properly (1: yes, 0:no)
    
    df:
        -pandas dataframe containing the variables of interest
    exposure:
        -column name of exposure variable. Must be coded as binary (0,1) where 1 is exposed
    outcome:
        -column name of outcome variable. Must be coded as binary (0,1) where 1 is the outcome of interest
    alpha:
        -Alpha value to calculate two-sided Wald confidence intervals. Default is 95% onfidence interval
    decimal:
        -amount of decimal points to display. Default is 3
    print_result:
        -Whether to print the results. Default is True, which prints the results
    return_result:
        -Whether to return the RD as a object. Default is False
    
    Example)
    >zepid.RiskDiff(df=data,exposure='exposure',outcome='outcome')
    '''
    zalpha = norm.ppf((1-alpha/2),loc=0,scale=1)
    a = len(df.loc[(df[exposure]==1)&(df[outcome]==1)])
    b = len(df.loc[(df[exposure]==1)&(df[outcome]==0)])
    c = len(df.loc[(df[exposure]==0)&(df[outcome]==1)])
    d = len(df.loc[(df[exposure]==0)&(df[outcome]==0)])
    rd(a=a,b=b,c=c,d=d,alpha=alpha,decimal=decimal,print_result=print_result,return_result=return_result)


def NNT(df,exposure,outcome,alpha=0.05,decimal=3,print_result=True,return_result=False):
    '''Estimates of Number Needed to Treat. NNT (1-alpha)*100% confidence interval presentation is based on 
    Altman, DG (BMJ 1998). Missing data is ignored by this function. 
    
    WARNING: Exposure & Outcome must be coded as 1 and 0 for this to work properly (1: yes, 0:no). 
    
    df:
        -pandas dataframe containing the variables of interest
    exposure:
        -column name of exposure variable. Must be coded as binary (0,1) where 1 is exposed
    outcome:
        -column name of outcome variable. Must be coded as binary (0,1) where 1 is the outcome of interest
    alpha:
        -Alpha value to calculate two-sided Wald confidence intervals. Default is 95% onfidence interval
    decimal:
        -amount of decimal points to display. Default is 3
    print_result:
        -Whether to print the results. Default is True
    return_result:
        -Whether to return the NNT as a object. Default is False
    
        Example)
    >zepid.NNT(df=data,exposure='exposure',outcome='outcome')
    '''
    zalpha = norm.ppf((1-alpha/2),loc=0,scale=1)
    a = len(df.loc[(df[exposure]==1)&(df[outcome]==1)])
    b = len(df.loc[(df[exposure]==1)&(df[outcome]==0)])
    c = len(df.loc[(df[exposure]==0)&(df[outcome]==1)])
    d = len(df.loc[(df[exposure]==0)&(df[outcome]==0)])
    nnt(a=a,b=b,c=c,d=d,alpha=alpha,decimal=decimal,print_result=print_result,return_result=return_result)
    

def OddsRatio(df,exposure,outcome,alpha=0.05,decimal=3,print_result=True,return_result=False):
    '''Estimates of Odds Ratio with a (1-alpha)*100% Confidence interval. Missing data is ignored by this function. 

    WARNING: Exposure & Outcome must be coded as 1 and 0 for this to work properly (1: yes, 0:no). 
    
    df:
        -pandas dataframe containing the variables of interest
    exposure:
        -column name of exposure variable. Must be coded as binary (0,1) where 1 is exposed
    outcome:
        -column name of outcome variable. Must be coded as binary (0,1) where 1 is the outcome of interest
    alpha:
        -Alpha value to calculate two-sided Wald confidence intervals. Default is 95% onfidence interval
    decimal:
        -amount of decimal points to display. Default is 3
    print_result:
        -Whether to print the results. Default is True
    return_result:
        -Whether to return the RR as a object. Default is False
    
    Example)
    >zepid.OddsRatio(df=data,exposure='exposure',outcome='outcome')
    '''
    zalpha = norm.ppf((1-alpha/2),loc=0,scale=1)
    a = len(df.loc[(df[exposure]==1)&(df[outcome]==1)])
    b = len(df.loc[(df[exposure]==1)&(df[outcome]==0)])
    c = len(df.loc[(df[exposure]==0)&(df[outcome]==1)])
    d = len(df.loc[(df[exposure]==0)&(df[outcome]==0)])
    if ((a<=5) | (b<=5) | (c<=5) | (d<=5)):
        warnings.warn('At least one cell count is less than 5, therefore confidence interval approximation is invalid')
    o1 = (a/b)
    o2 = (c/d)
    oddsratio = o1/o2
    SE=math.sqrt((1/a)+(1/b)+(1/c)+(1/d))
    lnor=math.log(oddsratio)
    lcl=lnor-(zalpha*SE)
    ucl=lnor+(zalpha*SE)
    oddsratio(a=a,b=b,c=c,d=d,alpha=alpha,decimal=decimal,print_result=print_result,return_result=return_result)


def IncRateRatio(df,exposure,outcome,time,alpha=0.05,decimal=3,print_result=True,return_result=False):
    '''Produces the estimate of the Incidence Rate Ratio with a (1-*alpha)*100% Confidence Interval. 
    Missing data is ignored by this function. 

    WARNING: Exposure & Outcome must be coded as 1 and 0 (1: yes, 0: no).
    
    df:
        -pandas dataframe containing the variables of interest
    exposure:
        -column name of exposure variable. Must be coded as binary (0,1) where 1 is exposed
    outcome:
        -column name of outcome variable. Must be coded as binary (0,1) where 1 is the outcome of interest
    time:
        -column name of person-time contributed by each individual. Must all be greater than 0
    alpha:
        -Alpha value to calculate two-sided Wald confidence intervals. Default is 95% onfidence interval
    decimal:
        -amount of decimal points to display. Default is 3
    print_result:
        -Whether to print the results. Default is True
    return_result:
        -Whether to return the RR as a object. Default is False
    
    Example)
    >zepid.IncRateRatio(df=data,exposure='exposure',outcome='outcome',time='person_time')
    '''
    zalpha = norm.ppf((1-alpha/2),loc=0,scale=1)
    a = len(df.loc[(df[exposure]==1)&(df[outcome]==1)])
    c = len(df.loc[(df[exposure]==0)&(df[outcome]==1)])
    time_a = df.loc[df[exposure]==1][time].sum()
    time_c = df.loc[df[exposure]==0][time].sum()
    irr(a=a,c=c,T1=time_a,T2=time_c,alpha=alpha,decimal=decimal,print_result=print_result,return_result=return_result)
    

def IncRateDiff(df,exposure,outcome,time,alpha=0.05,decimal=3,print_result=True,return_result=False):
    '''Produces the estimate of the Incidence Rate Difference with a (1-alpha)*100% confidence interval.
    Missing data is ignored by this function. 
    
    WARNING: Exposure & Outcome must be coded as 1 and 0 (1: yes, 0: no)
    
    exposure:
        -column name of exposure variable. Must be coded as binary (0,1) where 1 is exposed
    outcome:
        -column name of outcome variable. Must be coded as binary (0,1) where 1 is the outcome of interest
    time:
        -column name of person-time contributed by individual. Must be greater than 0
    alpha:
        -Alpha value to calculate two-sided Wald confidence intervals. Default is 95% onfidence interval
    decimal:
        -amount of decimal points to display. Default is 3
    print_result:
        -Whether to print the results. Default is True
    return_result:
        -Whether to return the RR as a object. Default is False
    
    Example)
    >zepid.IncRateDiff(df=data,exposure='exposure',outcome='outcome',time='person_time')
    '''
    zalpha = norm.ppf((1-alpha/2),loc=0,scale=1)
    a = len(df.loc[(df[exposure]==1)&(df[outcome]==1)])
    c = len(df.loc[(df[exposure]==0)&(df[outcome]==1)])
    time_a = df.loc[df[exposure]==1][time].sum()
    time_c = df.loc[df[exposure]==0][time].sum()
    ird(a=a,c=c,T1=time_a,T2=time_c,alpha=alpha,decimal=decimal,print_result=print_result,return_result=return_result)
 

def ACR(df,exposure,outcome,decimal=3):
    '''Produces the estimated Attributable Community Risk (ACR). ACR is also known as Population Attributable 
    Risk. Since this is commonly confused with the population attributable fraction, the name ACR is used to 
    clarify differences in the formulas. Missing data is ignored by this function. 
    
    WARNING: Exposure & Outcome must be coded as 1 and 0 for this to work properly (1: yes, 0:no)
    
    df:
        -pandas dataframe containing the variables of interest
    exposure:
        -column name of exposure variable. Must be coded as binary (0,1) where 1 is exposed
    outcome:
        -column name of outcome variable. Must be coded as binary (0,1) where 1 is the outcome of interest
    decimal:
        -amount of decimal points to display. Default is 3

    Example)
    >zepid.ACR(df=data,exposure='exposure',outcome='outcome')
    '''
    a = len(df.loc[(df[exposure]==1)&(df[outcome]==1)])
    b = len(df.loc[(df[exposure]==1)&(df[outcome]==0)])
    c = len(df.loc[(df[exposure]==0)&(df[outcome]==1)])
    d = len(df.loc[(df[exposure]==0)&(df[outcome]==0)])
    acr(a=a,b=b,c=c,d=d,decimal=decimal)


def PAF(df,exposure, outcome,decimal=3):
    '''Produces the estimated Population Attributable Fraction. Missing data is ignored by this function. 
    
    WARNING: Exposure & Outcome must be coded as 1 and 0 for this to work properly (1: yes, 0:no)
    
    exposure:
        -column name of exposure variable. Must be coded as binary (0,1) where 1 is exposed
    outcome:
        -column name of outcome variable. Must be coded as binary (0,1) where 1 is the outcome of interest
    decimal:
        -amount of decimal points to display. Default is 3
    
    Example)
    >zepid.PAF(df=data,exposure='exposure',outcome='outcome')
    '''
    a = len(df.loc[(df[exposure]==1)&(df[outcome]==1)])
    b = len(df.loc[(df[exposure]==1)&(df[outcome]==0)])
    c = len(df.loc[(df[exposure]==0)&(df[outcome]==1)])
    d = len(df.loc[(df[exposure]==0)&(df[outcome]==0)])
    paf(a=a,b=b,c=c,d=d,decimal=decimal)


def IC(df,exposure,outcome,modifier,adjust='',decimal=5):
    '''Calculate the Interaction Contrast (IC) using a pandas dataframe and statsmodels to fit a linear 
    binomial regression. Can ONLY be used for a 0,1 coded exposure and modifier (exposure = {0,1}, modifier = {0,1}, 
    outcome = {0,1}). Can handle adjustment for other confounders in the regression model. Prints the fit 
    of the linear binomial regression, the IC, and the corresponding IC 95% confidence interval.
    
    NOTE: statsmodels may produce a domain error in some versions. 
    
    df: 
        -pandas dataframe containing variables of interest
    exposure:
        -column name of exposure variable. Must be coded as (0,1) where 1 is exposure
    outcome:
        -column name of outcome variable. Must be coded as (0,1) where 1 is outcome of interest
    modifier:
        -column name of modifier variable. Must be coded as (0,1) where 1 is modifier
    adjust:
        -string of other variables to adjust for, in correct statsmodels format. Default is none
        NOTE: variables can NOT be named {E1M0,E0M1,E1M1} since this function creates variables with those names. 
              Answers will be incorrect
         Ex) '+ C1 + C2 + C3 + Z'
    decimal:
        -Number of decimals to display in result. Default is 3
    
    Example)
    zepid.IC(df=data,exposure='expo',outcome='event',modifier='modi',adjust='+ Z') 
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


def ICR(df,exposure,outcome,modifier,adjust='',regression='log',ci='delta',b_sample=1000,alpha=0.05,decimal=5):
    '''Calculate the Interaction Contrast Ratio (ICR) using a pandas dataframe, and conducts either log binomial 
    or logistic regression through statsmodels. Can ONLY be used for a 0,1 coded exposure and modifier (exposure = {0,1}, 
    modifier = {0,1}, outcome = {0,1}). Can handle missing data and adjustment for other confounders in the regression 
    model. Prints the fit of the binomial regression, the ICR, and the corresponding ICR confidence interval. Confidence 
    intervals can be generated using the delta method or bootstrap method
    
    NOTE: statsmodels may produce a domain error for log binomial models in some versions
    
    df: 
        -pandas dataframe containing variables of interest
    exposure:
        -column name of exposure variable. Must be coded as (0,1) where 1 is exposure
    outcome:
        -column name of outcome variable. Must be coded as (0,1) where 1 is outcome of interest
    modifier:
        -column name of modifier variable. Must be coded as (0,1) where 1 is modifier
    adjust:
        -string of other variables to adjust for, in correct statsmodels format. Default is none
        NOTE: variables can NOT be named {E1M0,E0M1,E1M1} since this function creates variables with those names. 
              Answers will be incorrect
         Ex) '+ C1 + C2 + C3 + Z'
    regression:
        -Type of regression model to fit. Default is log binomial.
         Options include:
            'log':      Log-binomial model. Estimates the Relative Risk (RR)
            'logit':    Logistic (logit) model. Estimates the Odds Ratio (OR). Note, this is only valid when the 
                        OR approximates the RR 
    ci:
        -Type of confidence interval to return. Default is the delta method. Options include:
            'delta':      Delta method as described by Hosmer and Lemeshow (1992)
            'bootstrap':  bootstrap method (Assmann et al. 1996). The delta method is more time efficient than bootstrap
    b_sample:
        -Number of times to resample to generate bootstrap confidence intervals. Only important if bootstrap confidence 
         intervals are requested. Default is 1000
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


def Sensitivity(df,test,disease,alpha=0.05,decimal=3,print_result=True,return_result=False):
    '''Generates the Sensitivity and (1-alpha)% confidence interval, comparing test results to disease status 
    from pandas dataframe

    WARNING: Disease & Test must be coded as (1: yes, 0:no)
    
    test:
        -column name of test results to detect the outcome. Needs to be coded as binary (0,1), where 1 indicates a 
        positive test for the individual
    disease:
        -column name of true outcomes status. Needs to be coded as binary (0,1), where 1 indicates the individual 
         has the outcome
    alpha:
        -Alpha value to calculate two-sided Wald confidence intervals. Default is 95% onfidence interval
    decimal:
        -amount of decimal points to display. Default is 3
    print_result:
        -Whether to print the results. Default is True
    return_result:
        -Whether to return the RR as a object. Default is False.
    
    Example)
    >zepid.Sensitivity(df,test='test_result',disease='outcome_stat')
    '''
    a = len(df.loc[(df[test]==1)&(df[disease]==1)])
    b = len(df.loc[(df[test]==1)&(df[disease]==0)])
    c = len(df.loc[(df[test]==0)&(df[disease]==1)])
    d = len(df.loc[(df[test]==0)&(df[disease]==0)])
    sens = a/(a+c)
    zalpha = norm.ppf((1-alpha/2),loc=0,scale=1)
    se = math.sqrt((sens*(1-sens)) / (a+c))
    lower = sens - zalpha*se
    upper = sens + zalpha*se
    if print_result == True:
        print(tabulate([["T+",a,b],["T-",c,d]],headers=['','D+','D-'],tablefmt='grid'))
        print('----------------------------------------------------------------------')
        print('Sensitivity: ',(round(sens,decimal)*100),'%','\n')
        print(str(round(100*(1-alpha)))+'% two-sided CI: (',round(lower,decimal),', ',round(upper,decimal),')')
        print('----------------------------------------------------------------------')
    if return_result == True:
        return sens


def Specificity(test,disease,alpha=0.05,decimal=3,print_result=True,return_result=False):
    '''Generates the Specificity and (1-alpha)% confidence interval, comparing test results to disease status 
    from pandas dataframe

    WARNING: Disease & Test must be coded as (1: yes, 0:no)
    
    test:
        -column name of test results to detect the outcome. Needs to be coded as binary (0,1), where 1 indicates a 
        positive test for the individual
    disease:
        -column name of true outcomes status. Needs to be coded as binary (0,1), where 1 indicates the individual 
         has the outcome
    alpha:
        -Alpha value to calculate two-sided Wald confidence intervals. Default is 95% onfidence interval
    decimal:
        -amount of decimal points to display. Default is 3
    print_result:
        -Whether to print the results. Default is True
    return_result:
        -Whether to return the RR as a object. Default is False.
    
    Example)
    >zepid.Specificity(df,test='test_result',disease='outcome_stat')    
    '''
    a = len(df.loc[(df[test]==1)&(df[disease]==1)])
    b = len(df.loc[(df[test]==1)&(df[disease]==0)])
    c = len(df.loc[(df[test]==0)&(df[disease]==1)])
    d = len(df.loc[(df[test]==0)&(df[disease]==0)])
    spec = d/(d+b)
    zalpha = norm.ppf((1-alpha/2),loc=0,scale=1)
    se = math.sqrt((spec*(1-spec)) / (b+d))
    lower = spec - zalpha*se
    upper = spec + zalpha*se
    if print_result == True:
        print(tabulate([["T+",a,b],["T-",c,d]],headers=['','D+','D-'],tablefmt='grid'))
        print('----------------------------------------------------------------------')
        print('Specificity: ',(round(spec,decimal)*100),'%','\n')
        print(str(round(100*(1-alpha)))+'% two-sided CI: (',round(lower,decimal),', ',round(upper,decimal),')')
        print('----------------------------------------------------------------------')
    if return_result == True:
        return spec


def StandMeanDiff(df,binary,continuous,decimal=3):
    '''Calculates the standardized mean difference (SMD) of a continuous variable stratified by a binary variable (0,1). 
    This can be used to assess for collinearity between the continuous and binary variables of interest. A SMD greater 
    than 2 suggests potential collinearity issues. It does NOT mean that there will be collinearity issues in the full 
    model though.
    
    df:
        -pandas dataframe containing variables of interest
    binary:
        -column name of binary variable. Must be coded as (0,1)
    continuous:
        -column name of continuous variable of interest
    decimal:
        -Number of decimal places to display. Default is 3
    
    Example)
    zepid.StandMeanDiff(df=data,binary='expo',continuous='age')
    '''
    v0 = df.loc[df[binary]==0]
    v1 = df.loc[df[binary]==1]
    m0 = np.mean(v0[continuous])
    m1 = np.mean(v1[continuous])
    sd0 = np.std(v0[continuous])
    sd1 = np.std(v1[continuous])
    pooled_sd = math.sqrt((((len(v0) - 1) * (sd0**2)) + ((len(v1) - 1) * (sd1**2))) / (len(v0) + len(v1) - 2))
    smd = abs(((m0 - m1) / pooled_sd))
    print('----------------------------------------------------------------------')
    print('Standardized Mean Difference: '+str(round(smd,decimal)))
    print('----------------------------------------------------------------------')


def spline(df,var,n_knots=3,knots=None,term=1,restricted=False):
    '''Creates spline dummy variables based on either user specified knot locations or automatically
    determines knot locations based on percentiles. Options are available to set the number of knots, 
    location of knots (value), term (linear, quadratic, etc.), and restricted/unrestricted.
    
    Returns a pandas dataframe containing the spline variables (labeled 0 to n_knots)
    
    df:
        -pandas dataframe containing the variables of interest
    var:
        -continuous variable to generate spline for
    n_knots:
        -number of knots requested. Options for knots include any positive integer if the location of 
         knots are specified. If knot locations are not specified, n_knots must be an integer between 
         1 to 7, including both. Default is set to 3
    knots:
        -Location of specified knots in a list. To specify the location of knots, put desired numbers for 
         knots into a list. Be sure that the length of the list is the same as the specified number of knots. 
         Default is None, so that the function will automatically determine knot locations without user specification
    term:
        -High order term for the spline terms. To calculate a quadratic spline change to 2, cubic spline 
         change to 3, etc. Default is 1, so a linear spline is returned
    restricted:
        -Whether to return a restricted spline. Note that the restricted spline returns one less column
         than the number of knots. An unrestricted spline returns the same number of columns as the number of knots.
         Default is False, providing an unrestricted spline
    
    Example)
    >zepid.spline(df=data,var='age',n_knots=4,knots=[18,30,50,65],term=2,restricted=True)
    '''
    if knots == None:
        if n_knots == 1:
            knots = [0.5]
        elif n_knots == 2:
            knots = [1/3,2/3]
        elif n_knots == 3:
            knots = [0.05,0.5,0.95]
        elif n_knots == 4:
            knots = [0.05,0.35,0.65,0.95]
        elif n_knots == 5:
            knots = [0.05,0.275,0.50,0.725,0.95]
        elif n_knots == 6:
            knots = [0.05,0.23,0.41,0.59,0.77,0.95]
        elif n_knots == 7:
            knots = [0.025,11/60,26/75,0.50,79/120,49/60,0.975]
        else:
            raise ValueError('When the knot locations are not pre-specified, the number of specified knots must be an integer between 1 and 7')
        pts = list(df[var].quantile(q=knots))
    else: 
        if n_knots != len(knots):
            raise ValueError('The number of knots and the number of specified knots must match')
        else:
            pass
        pts = knots
    if sorted(pts) != pts:
        raise ValueError('Knots must be in ascending order')
    colnames = []
    sf = df.copy()
    for i in range(len(pts)):
        colnames.append('spline'+str(i))
        sf['ref'+str(i)] = (sf[var] - pts[i])**term
        sf['spline'+str(i)] = [j if x > pts[i] else 0 for x,j in zip(sf[var],sf['ref'+str(i)])]
        sf.loc[sf[var].isnull(),'spline'+str(i)] = np.nan
    if restricted == False:
        return sf[colnames]
    elif restricted == True:
        rsf = sf.copy()
        colnames = []
        for i in range(len(pts)-1):
            colnames.append('rspline'+str(i))
            rsf['ref'+str(i)] = (rsf['spline'+str(i)] - rsf['spline'+str(len(pts)-1)])
            rsf['rspline'+str(i)] = [j if x > pts[i] else 0 for x,j in zip(rsf[var],rsf['ref'+str(i)])]
            rsf.loc[rsf[var].isnull(),'rspline'+str(i)] = np.nan
        return rsf[colnames]
    else:
        raise ValueError('restricted must be set to either True or False')


