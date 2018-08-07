import warnings
import math
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import links
from tabulate import tabulate
from zepid.calc.utils import rr, rd, nnt, oddsratio, ird, irr, acr, paf, sensitivity, specificity


def RiskRatio(df, exposure, outcome, reference=0, alpha=0.05, decimal=3, print_result=True, return_result=False):
    """Estimate of Relative Risk with a (1-alpha)*100% Confidence interval. Missing data is ignored by
    this function.

    WARNING: Exposure & Outcome must be coded as (1: yes, 0:no). Only works for binary exposures and outcomes

    df:
        -pandas dataframe containing variables of interest
    exposure:
        -column name of exposure variable. Must be coded as binary (0,1) where 1 is exposed
    outcome:
        -column name of outcome variable. Must be coded as binary (0,1) where 1 is the outcome of interest
    reference:
        -reference category for comparisons
    alpha:
        -Alpha value to calculate two-sided Wald confidence intervals. Default is 95% confidence interval
    decimal:
        -amount of decimal points to display. Default is 3
    print_result:
        -Whether to print the results. Default is True
    return_result:
        -Whether to return the RR as a object. Default is False

    Example)
    >>>zepid.RiskRatio(df=data,exposure='X',outcome='D')
    """
    c = df.loc[(df[exposure] == reference) & (df[outcome] == 1)].shape[0]
    d = df.loc[(df[exposure] == reference) & (df[outcome] == 0)].shape[0]
    vals = set(df[exposure].dropna().unique())
    vals.remove(reference)
    for i in vals:
        print('======================================================================')
        print('Comparing ' + exposure + '=' + str(i) + ' to ' + exposure + '=' + str(reference))
        print('======================================================================')
        a = df.loc[(df[exposure] == i) & (df[outcome] == 1)].shape[0]
        b = df.loc[(df[exposure] == i) & (df[outcome] == 0)].shape[0]
        rr(a=a, b=b, c=c, d=d, alpha=alpha, decimal=decimal, print_result=print_result, return_result=return_result)


def RiskDiff(df, exposure, outcome, reference=0, alpha=0.05, decimal=3, print_result=True, return_result=False):
    """Estimate of Risk Difference with a (1-alpha)*100% Confidence interval. Missing data is ignored by this
    function.

    WARNING: Exposure & Outcome must be coded as 1 and 0 for this to work properly (1: yes, 0:no)

    df:
        -pandas dataframe containing the variables of interest
    exposure:
        -column name of exposure variable. Must be coded as binary (0,1) where 1 is exposed
    outcome:
        -column name of outcome variable. Must be coded as binary (0,1) where 1 is the outcome of interest
    reference:
        -reference category for comparisons
    alpha:
        -Alpha value to calculate two-sided Wald confidence intervals. Default is 95% onfidence interval
    decimal:
        -amount of decimal points to display. Default is 3
    print_result:
        -Whether to print the results. Default is True, which prints the results
    return_result:
        -Whether to return the RD as a object. Default is False

    Example)
    >>>zepid.RiskDiff(df=data,exposure='X',outcome='D')
    """
    c = df.loc[(df[exposure] == reference) & (df[outcome] == 1)].shape[0]
    d = df.loc[(df[exposure] == reference) & (df[outcome] == 0)].shape[0]
    vals = set(df[exposure].dropna().unique())
    vals.remove(reference)
    for i in vals:
        print('======================================================================')
        print('Comparing ' + exposure + '=' + str(i) + ' to ' + exposure + '=' + str(reference))
        print('======================================================================')
        a = df.loc[(df[exposure] == i) & (df[outcome] == 1)].shape[0]
        b = df.loc[(df[exposure] == i) & (df[outcome] == 0)].shape[0]
        rd(a=a, b=b, c=c, d=d, alpha=alpha, decimal=decimal, print_result=print_result, return_result=return_result)


def NNT(df, exposure, outcome, reference=0, alpha=0.05, decimal=3, print_result=True, return_result=False):
    """Estimates of Number Needed to Treat. NNT (1-alpha)*100% confidence interval presentation is based on
    Altman, DG (BMJ 1998). Missing data is ignored by this function.

    WARNING: Exposure & Outcome must be coded as 1 and 0 for this to work properly (1: yes, 0:no).

    df:
        -pandas dataframe containing the variables of interest
    exposure:
        -column name of exposure variable. Must be coded as binary (0,1) where 1 is exposed
    outcome:
        -column name of outcome variable. Must be coded as binary (0,1) where 1 is the outcome of interest
    reference:
        -reference category for comparisons
    alpha:
        -Alpha value to calculate two-sided Wald confidence intervals. Default is 95% onfidence interval
    decimal:
        -amount of decimal points to display. Default is 3
    print_result:
        -Whether to print the results. Default is True
    return_result:
        -Whether to return the NNT as a object. Default is False

        Example)
    >>>zepid.NNT(df=data,exposure='X',outcome='D')
    """
    c = df.loc[(df[exposure] == reference) & (df[outcome] == 1)].shape[0]
    d = df.loc[(df[exposure] == reference) & (df[outcome] == 0)].shape[0]
    vals = set(df[exposure].dropna().unique())
    vals.remove(reference)
    for i in vals:
        print('======================================================================')
        print('Comparing ' + exposure + '=' + str(i) + ' to ' + exposure + '=' + str(reference))
        print('======================================================================')
        a = df.loc[(df[exposure] == i) & (df[outcome] == 1)].shape[0]
        b = df.loc[(df[exposure] == i) & (df[outcome] == 0)].shape[0]
        nnt(a=a, b=b, c=c, d=d, alpha=alpha, decimal=decimal, print_result=print_result, return_result=return_result)


def OddsRatio(df, exposure, outcome, reference=0, alpha=0.05, decimal=3, print_result=True, return_result=False):
    """Estimates of Odds Ratio with a (1-alpha)*100% Confidence interval. Missing data is ignored by this function.

    WARNING: Exposure & Outcome must be coded as 1 and 0 for this to work properly (1: yes, 0:no).

    df:
        -pandas dataframe containing the variables of interest
    exposure:
        -column name of exposure variable. Must be coded as binary (0,1) where 1 is exposed
    outcome:
        -column name of outcome variable. Must be coded as binary (0,1) where 1 is the outcome of interest
    reference:
        -reference category for comparisons
    alpha:
        -Alpha value to calculate two-sided Wald confidence intervals. Default is 95% onfidence interval
    decimal:
        -amount of decimal points to display. Default is 3
    print_result:
        -Whether to print the results. Default is True
    return_result:
        -Whether to return the RR as a object. Default is False

    Example)
    >>>zepid.OddsRatio(df=data,exposure='X',outcome='D')
    """
    c = df.loc[(df[exposure] == reference) & (df[outcome] == 1)].shape[0]
    d = df.loc[(df[exposure] == reference) & (df[outcome] == 0)].shape[0]
    vals = set(df[exposure].dropna().unique())
    vals.remove(reference)
    for i in vals:
        print('======================================================================')
        print('Comparing ' + exposure + '=' + str(i) + ' to ' + exposure + '=' + str(reference))
        print('======================================================================')
        a = df.loc[(df[exposure] == i) & (df[outcome] == 1)].shape[0]
        b = df.loc[(df[exposure] == i) & (df[outcome] == 0)].shape[0]
        oddsratio(a=a, b=b, c=c, d=d, alpha=alpha, decimal=decimal, print_result=print_result,
                  return_result=return_result)


def IncRateRatio(df, exposure, outcome, time, reference=0, alpha=0.05, decimal=3, print_result=True,
                 return_result=False):
    """Produces the estimate of the Incidence Rate Ratio with a (1-*alpha)*100% Confidence Interval.
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
    reference:
        -reference category for comparisons
    alpha:
        -Alpha value to calculate two-sided Wald confidence intervals. Default is 95% onfidence interval
    decimal:
        -amount of decimal points to display. Default is 3
    print_result:
        -Whether to print the results. Default is True
    return_result:
        -Whether to return the RR as a object. Default is False
    """
    c = df.loc[(df[exposure] == reference) & (df[outcome] == 1)].shape[0]
    time_c = df.loc[df[exposure] == reference][time].sum()
    vals = set(df[exposure].dropna().unique())
    vals.remove(reference)
    for i in vals:
        print('======================================================================')
        print('Comparing ' + exposure + '=' + str(i) + ' to ' + exposure + '=' + str(reference))
        print('======================================================================')
        a = df.loc[(df[exposure] == i) & (df[outcome] == 1)].shape[0]
        time_a = df.loc[df[exposure] == i][time].sum()
        irr(a=a, c=c, t1=time_a, t2=time_c, alpha=alpha, decimal=decimal, print_result=print_result,
            return_result=return_result)


def IncRateDiff(df, exposure, outcome, time, reference=0, alpha=0.05, decimal=3, print_result=True,
                return_result=False):
    """Produces the estimate of the Incidence Rate Difference with a (1-alpha)*100% confidence interval.
    Missing data is ignored by this function.

    WARNING: Exposure & Outcome must be coded as 1 and 0 (1: yes, 0: no)

    exposure:
        -column name of exposure variable. Must be coded as binary (0,1) where 1 is exposed
    outcome:
        -column name of outcome variable. Must be coded as binary (0,1) where 1 is the outcome of interest
    time:
        -column name of person-time contributed by individual. Must be greater than 0
    reference:
        -reference category for comparisons
    alpha:
        -Alpha value to calculate two-sided Wald confidence intervals. Default is 95% onfidence interval
    decimal:
        -amount of decimal points to display. Default is 3
    print_result:
        -Whether to print the results. Default is True
    return_result:
        -Whether to return the RR as a object. Default is False
    """
    c = df.loc[(df[exposure] == reference) & (df[outcome] == 1)].shape[0]
    time_c = df.loc[df[exposure] == reference][time].sum()
    vals = set(df[exposure].dropna().unique())
    vals.remove(reference)
    for i in vals:
        print('======================================================================')
        print('Comparing ' + exposure + '=' + str(i) + ' to ' + exposure + '=' + str(reference))
        print('======================================================================')
        a = df.loc[(df[exposure] == i) & (df[outcome] == 1)].shape[0]
        time_a = df.loc[df[exposure] == i][time].sum()
        ird(a=a, c=c, t1=time_a, t2=time_c, alpha=alpha, decimal=decimal, print_result=print_result,
            return_result=return_result)


def ACR(df, exposure, outcome, decimal=3):
    """Produces the estimated Attributable Community Risk (ACR). ACR is also known as Population Attributable
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
    """
    a = df.loc[(df[exposure] == 1) & (df[outcome] == 1)].shape[0]
    b = df.loc[(df[exposure] == 1) & (df[outcome] == 0)].shape[0]
    c = df.loc[(df[exposure] == 0) & (df[outcome] == 1)].shape[0]
    d = df.loc[(df[exposure] == 0) & (df[outcome] == 0)].shape[0]
    acr(a=a, b=b, c=c, d=d, decimal=decimal)


def PAF(df, exposure, outcome, decimal=3):
    """Produces the estimated Population Attributable Fraction. Missing data is ignored by this function.

    WARNING: Exposure & Outcome must be coded as 1 and 0 for this to work properly (1: yes, 0:no)

    exposure:
        -column name of exposure variable. Must be coded as binary (0,1) where 1 is exposed
    outcome:
        -column name of outcome variable. Must be coded as binary (0,1) where 1 is the outcome of interest
    decimal:
        -amount of decimal points to display. Default is 3
    """
    a = df.loc[(df[exposure] == 1) & (df[outcome] == 1)].shape[0]
    b = df.loc[(df[exposure] == 1) & (df[outcome] == 0)].shape[0]
    c = df.loc[(df[exposure] == 0) & (df[outcome] == 1)].shape[0]
    d = df.loc[(df[exposure] == 0) & (df[outcome] == 0)].shape[0]
    paf(a=a, b=b, c=c, d=d, decimal=decimal)


def IC(df, exposure, outcome, modifier, adjust=None, decimal=3):
    """Calculate the Interaction Contrast (IC) using a pandas dataframe and statsmodels to fit a linear
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
        -string of other variables to adjust for, in correct statsmodels format. Default is None
        NOTE: variables can NOT be named {E1M0,E0M1,E1M1} since this function creates variables with those names.
              Answers will be incorrect
         Ex) '+ C1 + C2 + C3 + Z'
    decimal:
        -Number of decimals to display in result. Default is 3

    Example)
    >>>zepid.IC(df=data,exposure='X',outcome='D',modifier='Z',adjust='var1 + var2')
                     Generalized Linear Model Regression Results
    ==============================================================================
    Dep. Variable:                      D   No. Observations:                  210
    Model:                            GLM   Df Residuals:                      204
    Model Family:                Binomial   Df Model:                            5
    Link Function:               identity   Scale:                             1.0
    Method:                          IRLS   Log-Likelihood:                -97.450
    Date:                Thu, 03 May 2018   Deviance:                       194.90
    Time:                        18:46:13   Pearson chi2:                     198.
    No. Iterations:                    79
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept      0.6101      0.542      1.125      0.260      -0.453       1.673
    X              0.2049      0.056      3.665      0.000       0.095       0.314
    Z              0.1580      0.049      3.207      0.001       0.061       0.255
    E1M1          -0.2105      0.086     -2.447      0.014      -0.379      -0.042
    var1        7.544e-05    6.7e-05      1.125      0.260    -5.6e-05       0.000
    var2          -0.0248      0.022     -1.125      0.260      -0.068       0.018
    ==============================================================================

    ----------------------------------------------------------------------
    Interaction Contrast
    ----------------------------------------------------------------------

    IC:		-0.21047
    95% CI:		(-0.37908, -0.04186)
    ----------------------------------------------------------------------
    """
    df.loc[((df[exposure] == 1) & (df[modifier] == 1)), 'E1M1'] = 1
    df.loc[((df[exposure] != 1) | (df[modifier] != 1)), 'E1M1'] = 0
    df.loc[((df[exposure].isnull()) | (df[modifier].isnull())), 'E1M1'] = np.nan
    if adjust is None:
        eq = outcome + ' ~ ' + exposure + ' + ' + modifier + ' + E1M1'
    else:
        eq = outcome + ' ~ ' + exposure + ' + ' + modifier + ' + E1M1 + ' + adjust
    f = sm.families.family.Binomial(sm.families.links.identity)
    model = smf.glm(eq, df, family=f).fit()
    print(model.summary())
    ic = model.params['E1M1']
    lcl = model.conf_int().loc['E1M1'][0]
    ucl = model.conf_int().loc['E1M1'][1]
    print('\n----------------------------------------------------------------------')
    print('Interaction Contrast')
    print('----------------------------------------------------------------------')
    print('\nIC:\t\t' + str(round(ic, decimal)))
    print('95% CI:\t\t(' + str(round(lcl, decimal)) + ', ' + str(round(ucl, decimal)) + ')')
    print('----------------------------------------------------------------------')


def ICR(df, exposure, outcome, modifier, adjust=None, regression='log', ci='delta', b_sample=200, alpha=0.05,
        decimal=5):
    """Calculate the Interaction Contrast Ratio (ICR) using a pandas dataframe, and conducts either log binomial
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

    Example)
    >>>zepid.ICR(df=data,exposure='X',outcome='D',modifier='Z',adjust='var1 + var2')
    """
    df.loc[((df[exposure] == 1) & (df[modifier] == 0)), 'E1M0'] = 1
    df.loc[((df[exposure] != 1) | (df[modifier] != 0)), 'E1M0'] = 0
    df.loc[((df[exposure].isnull()) | (df[modifier].isnull())), 'E1M0'] = 0
    df.loc[((df[exposure] == 0) & (df[modifier] == 1)), 'E0M1'] = 1
    df.loc[((df[exposure] != 0) | (df[modifier] != 1)), 'E0M1'] = 0
    df.loc[((df[exposure].isnull()) | (df[modifier].isnull())), 'E0M1'] = 0
    df.loc[((df[exposure] == 1) & (df[modifier] == 1)), 'E1M1'] = 1
    df.loc[((df[exposure] != 1) | (df[modifier] != 1)), 'E1M1'] = 0
    df.loc[((df[exposure].isnull()) | (df[modifier].isnull())), 'E1M1'] = np.nan
    if regression == 'logit':
        f = sm.families.family.Binomial(sm.families.links.logit)
        print('Note: Using the Odds Ratio to calculate the ICR is only valid when\nthe OR approximates the RR')
    elif regression == 'log':
        f = sm.families.family.Binomial(sm.families.links.log)
    if adjust == None:
        eq = outcome + ' ~ E1M0 + E0M1 + E1M1'
    else:
        eq = outcome + ' ~ E1M0 + E0M1 + E1M1 + ' + adjust
    model = smf.glm(eq, df, family=f).fit()
    print(model.summary())
    em10 = math.exp(model.params['E1M0'])
    em01 = math.exp(model.params['E0M1'])
    em11 = math.exp(model.params['E1M1'])
    em_expect = em10 + em01 - 1
    icr = em11 - em_expect
    zalpha = norm.ppf((1 - alpha / 2), loc=0, scale=1)
    if ci == 'delta':
        cov_matrix = model.cov_params()
        vb10 = cov_matrix.loc['E1M0']['E1M0']
        vb01 = cov_matrix.loc['E0M1']['E0M1']
        vb11 = cov_matrix.loc['E1M1']['E1M1']
        cvb10_01 = cov_matrix.loc['E1M0']['E0M1']
        cvb10_11 = cov_matrix.loc['E1M0']['E1M1']
        cvb01_11 = cov_matrix.loc['E0M1']['E1M1']
        varICR = (((em10 ** 2) * vb10) + ((em01 ** 2) * vb01) + ((em11 ** 2) * vb11) + (
        (em10 * em01 * 2 * cvb10_01)) + (-1 * em10 * em11 * 2 * cvb10_11) + (-1 * em01 * em11 * 2 * cvb01_11))
        icr_lcl = icr - zalpha * math.sqrt(varICR)
        icr_ucl = icr + zalpha * math.sqrt(varICR)
    elif ci == 'bootstrap':
        print('Running bootstrap... please wait...')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bse_icr = []
            ul = 1 - alpha / 2
            ll = 0 + alpha / 2
            for i in range(b_sample):
                dfs = df.sample(n=df.shape[0], replace=True)
                try:
                    bmodel = smf.glm(eq, dfs, family=f).fit()
                    em_bexpect = math.exp(bmodel.params['E1M0']) + math.exp(bmodel.params['E0M1']) - 1
                    bicr = math.exp(bmodel.params['E1M1']) - em_bexpect
                    sigma = bicr - icr
                    bse_icr.append(sigma)
                except:
                    bse_icr.append(np.nan)
            bsdf = pd.DataFrame()
            bsdf['sigma'] = bse_icr
            lsig, usig = bsdf['sigma'].dropna().quantile(q=[ll, ul])
            icr_lcl = lsig + icr
            icr_ucl = usig + icr
    else:
        raise ValueError('Please specify a supported confidence interval type')
    print('\n----------------------------------------------------------------------')
    if regression == 'logit':
        print('ICR based on Odds Ratio\t\tAlpha = ' + str(alpha))
        print('Note: Using the Odds Ratio to calculate the ICR is only valid when\nthe OR approximates the RR')
    elif regression == 'log':
        print('ICR based on Risk Ratio\t\tAlpha = ' + str(alpha))
    print('\nICR:\t\t' + str(round(icr, decimal)))
    print('CI:\t\t(' + str(round(icr_lcl, decimal)) + ', ' + str(round(icr_ucl, decimal)) + ')')
    print('----------------------------------------------------------------------')


def Sensitivity(df, test, disease, alpha=0.05, decimal=3, print_result=True, return_result=False):
    """Generates the sensitivity and (1-alpha)% confidence interval, comparing test results to disease status
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
        -Whether to return the calculated sensitivity. Default is False.

    Example)
    >zepid.Sensitivity(df=data,test='test_disease',disease='true_disease')
    """
    a = df.loc[(df[test] == 1) & (df[disease] == 1)].shape[0]
    b = df.loc[(df[test] == 1) & (df[disease] == 0)].shape[0]
    if print_result is True:
        print(tabulate([["T+", a, b]], headers=['', 'D+', 'D-'], tablefmt='grid'))
        print('----------------------------------------------------------------------')
        sensitivity(a, a + b, alpha=alpha, decimal=decimal, confint='wald', print_result=True)
        print('----------------------------------------------------------------------')
    if return_result is True:
        se = sensitivity(a, a + b, print_result=False, return_result=True)
        return se


def Specificity(df, test, disease, alpha=0.05, decimal=3, print_result=True, return_result=False):
    """Generates the Specificity and (1-alpha)% confidence interval, comparing test results to disease status
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
        -Whether to return the calculated specificity. Default is False.

    Example)
    >zepid.Specificity(df=data,test='test_disease',disease='true_disease')
    """
    c = df.loc[(df[test] == 0) & (df[disease] == 1)].shape[0]
    d = df.loc[(df[test] == 0) & (df[disease] == 0)].shape[0]
    if print_result:
        print(tabulate([["T-", c, d]], headers=['', 'D+', 'D-'], tablefmt='grid'))
        print('----------------------------------------------------------------------')
        specificity(c, c + d, alpha=alpha, decimal=decimal, confint='wald', print_result=True)
        print('----------------------------------------------------------------------')
    if return_result:
        sp = specificity(c, c + d, print_result=False, return_result=True)
        return sp


def Diagnostics(df, test, disease, alpha=0.05, decimal=3, print_result=True, return_result=False):
    """
    Generates the Sensitivity, Specificity and corresponding (1-alpha)% confidence intervals, comparing test results
    to disease status from pandas dataframe

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
        -Whether to return the calculates sensitivity and specificity. Default is False.
    """
    a = df.loc[(df[test] == 1) & (df[disease] == 1)].shape[0]
    b = df.loc[(df[test] == 1) & (df[disease] == 0)].shape[0]
    c = df.loc[(df[test] == 0) & (df[disease] == 1)].shape[0]
    d = df.loc[(df[test] == 0) & (df[disease] == 0)].shape[0]
    if print_result:
        print(tabulate([["T+", a, b], ["T-", c, d]], headers=['', 'D+', 'D-'], tablefmt='grid'))
        print('----------------------------------------------------------------------')
        sensitivity(a, a + b, alpha=alpha, decimal=decimal, confint='wald', print_result=True)
        specificity(c, c + d, alpha=alpha, decimal=decimal, confint='wald', print_result=True)
        print('----------------------------------------------------------------------')
    if return_result:
        se = sensitivity(a, a + b, print_result=False, return_result=True)
        sp = specificity(c, c + d, print_result=False, return_result=True)
        return se, sp


def spline(df, var, n_knots=3, knots=None, term=1, restricted=False):
    """Creates spline dummy variables based on either user specified knot locations or automatically
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
    >>>zepid.spline(df=data,var='var1',n_knots=4,term=2,restricted=True)
           rspline0     rspline1   rspline2
    0   9839.409066  1234.154601   2.785600
    1    446.391437     0.000000   0.000000
    2   7107.550298   409.780251   0.000000
    3   4465.272901     7.614501   0.000000
    4  10972.041543  1655.208555  52.167821
    ..          ...          ...        ...
    """
    if knots == None:
        if n_knots == 1:
            knots = [0.5]
        elif n_knots == 2:
            knots = [1 / 3, 2 / 3]
        elif n_knots == 3:
            knots = [0.05, 0.5, 0.95]
        elif n_knots == 4:
            knots = [0.05, 0.35, 0.65, 0.95]
        elif n_knots == 5:
            knots = [0.05, 0.275, 0.50, 0.725, 0.95]
        elif n_knots == 6:
            knots = [0.05, 0.23, 0.41, 0.59, 0.77, 0.95]
        elif n_knots == 7:
            knots = [0.025, 11 / 60, 26 / 75, 0.50, 79 / 120, 49 / 60, 0.975]
        else:
            raise ValueError(
                'When the knot locations are not pre-specified, the number of specified knots must be'
                ' an integer between 1 and 7')
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
        colnames.append('spline' + str(i))
        sf['spline' + str(i)] = np.where(sf[var] > pts[i], (sf[var] - pts[i]) ** term, 0)
        sf['spline' + str(i)] = np.where(sf[var].isnull(), np.nan, sf['spline' + str(i)])
    if restricted is False:
        return sf[colnames]
    elif restricted is True:
        rsf = sf.copy()
        colnames = []
        for i in range(len(pts) - 1):
            colnames.append('rspline' + str(i))
            rsf['rspline' + str(i)] = np.where(rsf[var] > pts[i],
                                               rsf['spline' + str(i)] - rsf['spline' + str(len(pts) - 1)], 0)
            rsf['rspline' + str(i)] = np.where(rsf[var].isnull(), np.nan, rsf['rspline' + str(i)])
        return rsf[colnames]
    else:
        raise ValueError('restricted must be set to either True or False')


def Table1(df, cols, variable_type, continuous_measure='median', strat_by=None, decimal=3):
    """Code to automatically generate a descriptive table of your study population (often referred to as a
    Table 1). Personally, I hate copying SAS/R/Python output from the interpreter to an Excel or other
    spreadsheet software. This code will generate a pandas dataframe object. This object will be a formatted
    table which can be exported as a CSV, opened in Excel, then final formatting changes/renaming can be done.
    Variables with np.nan values are counted as missing

    Categorical variables will be divided into the unique numbers and have a percent calculated. Additionally,
    missing data will be counted (but is not included in the percent). Additionally, a single categorical variable
    can be used to present the results

    Continuous variables either have median/IQR or mean/SE calculated depending on what is requested. Missing are
    counted as a separate category

    Returns a pandas dataframe object containing a formatted Table 1. It is not recommended that this table is used
    in any part of later analysis, since is id difficult to parse through the table. This function is only meant to
    reduce the amount of copying from output needed.

    df:
        -pandas dataframe object containing all variables of interest
    cols:
        -list of columns of variable names to include in the table. Ex) ['X',var1','var2']
    variable_types:
        -list of strings indicating the variable types. Ex) ['category','continuous','continuous']
         Options
            'category'      :   variable with categories only
            'continuous'    :   continuous variable
    continuous_measure:
        -Whether to use the medians or the means. Default is median
         Options
            'median'    :   returns medians and IQR for continuous variables
            'mean'      :   returns means and SE for continuous variables
    strat_by:
        -What categorical variable to stratify by. Default is None (no stratification)
    decimal:
        -Number of decimals to display in the table. Default is 3

    Example)
    >>>var_types = ['category','category','continuous','continuous','continuous']
    >>>zepid.Table1(df=data,cols=['X','Z','var1','var2','var3'],variable_type=var_types,strat_by='D')
    _                                D=0                             D=1
    __                           % / IQR           n             % / IQR          n

    Variable
    TOTAL                                 310.000000                      74.000000
    X        1.0                0.608187  104.000000            0.692308  27.000000
             0.0                0.391813   67.000000            0.307692  12.000000
             Missing                      139.000000                      35.000000
    Z        1.0                0.722581  224.000000            0.635135  47.000000
             0.0                0.277419   86.000000            0.364865  27.000000
             Missing                        0.000000                       0.000000
    var1              [468.231, 525.312]  497.262978  [481.959, 538.964] 507.286133
             Missing                        0.000000                       0.000000
    var2                [24.454, 25.731]   25.058982      [24.1, 25.607]  24.816898
             Missing                        0.000000                       0.000000
    var3                [24.446, 25.685]   25.037731    [24.388, 25.563]  24.920583
             Missing                        0.000000
    >>>_.to_csv('path/filename.csv')
    """
    # Unstratificed Table 1
    if strat_by == None:
        rlist = []
        for i in cols:
            vn = cols.index(i)
            if continuous_measure == 'median':
                if variable_type[vn] == 'continuous':
                    rf = pd.DataFrame({'n / Median': [np.median(df[i].dropna()), df[i].isnull().sum()],
                                       '% / IQR': [np.percentile(df[i].dropna(), [25, 75]).round(decimals=decimal),
                                                   '']}, index=['', 'Missing'])
                if variable_type[vn] == 'category':
                    x = df[i].value_counts()
                    m = df[i].isnull().sum()
                    rf = pd.DataFrame({'n / Median': x, '% / IQR': x / x.sum()})
                    rf = rf.append(pd.DataFrame({'n / Median': m, '% / IQR': ''}, index=['Missing']))
            elif continuous_measure == 'mean':
                if variable_type[vn] == 'continuous':
                    rf = pd.DataFrame({'n / Mean': [np.mean(df[i].dropna()), df[i].isnull().sum()],
                                       '% / SE': [np.std(df[i].dropna()).round(decimals=decimal), '']},
                                      index=['', 'Missing'])
                if variable_type[vn] == 'category':
                    x = df[i].value_counts(dropna=False)
                    y = df[i].value_counts(dropna=False)
                    rf = pd.DataFrame({'n / Mean': x, '% / SE': y / y.sum()})
            else:
                raise ValueError('median or mean must be specified')
            rlist.append(rf)
        srf = pd.concat(rlist, keys=cols, names=['Variable'])
        if continuous_measure == 'median':
            return srf[['n / Median', '% / IQR']]
        if continuous_measure == 'mean':
            return srf[['n / Mean', '% / SE']]

    # Stratified Table 1
    if strat_by != None:
        v = df[strat_by].dropna().unique()
        slist = []
        nlist = []
        for j in v:
            sf = df.loc[df[strat_by] == j].copy()
            rlist = []
            for i in cols:
                vn = cols.index(i)
                if continuous_measure == 'median':
                    if variable_type[vn] == 'continuous':
                        rf = pd.DataFrame({'n / Median': [np.median(sf[i].dropna()), sf[i].isnull().sum()],
                                           '% / IQR': [np.percentile(sf[i].dropna(), [25, 75]).round(decimals=decimal),
                                                       '']}, index=['', 'Missing'])
                    if variable_type[vn] == 'category':
                        x = sf[i].value_counts()
                        m = sf[i].isnull().sum()
                        rf = pd.DataFrame({'n / Median': x, '% / IQR': x / x.sum()})
                        rf = rf.append(pd.DataFrame({'n / Median': m, '% / IQR': ''}, index=['Missing']))
                if continuous_measure == 'mean':
                    if variable_type[vn] == 'continuous':
                        rf = pd.DataFrame({'n / Mean': [np.mean(sf[i].dropna()), sf[i].isnull().sum()],
                                           '% / SE': [np.std(sf[i].dropna()).round(decimals=decimal), '']},
                                          index=['', 'Missing'])
                    if variable_type[vn] == 'category':
                        x = sf[i].value_counts()
                        m = sf[i].isnull().sum()
                        rf = pd.DataFrame({'n / Mean': x, '% / SD': x / x.sum()})
                        rf = rf.append(pd.DataFrame({'n / Mean': m, '% / SD': ''}, index=['Missing']))
                rlist.append(rf)
            if continuous_measure == 'median':
                c = pd.DataFrame({'n / Median': len(sf), '% / IQR': ''}, index=[''])
            if continuous_measure == 'mean':
                c = pd.DataFrame({'n / Mean': len(sf), '% / SD': ''}, index=[''])
            rff = pd.concat([c] + rlist, keys=['TOTAL'] + cols, names=['Variable'], axis=0)
            slist.append(rff)
            if continuous_measure == 'median':
                nlist.append((strat_by + '=' + str(j), '% / IQR'))
            if continuous_measure == 'mean':
                nlist.append((strat_by + '=' + str(j), '% / SD'))
            nlist.append((strat_by + '=' + str(j), 'n'))
        index = pd.MultiIndex.from_tuples(nlist, names=['_', '__'])
        srf = pd.concat(slist, keys=cols, names=['Variable'], axis=1)
        srf.columns = index
        return srf
