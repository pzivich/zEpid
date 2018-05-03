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


def rr_corr(rr_obs,rr_conf,p1,p0):
    '''Simple Sensitivity analysis calculator for Risk Ratios. Estimates the impact of 
    an unmeasured confounder on the results of a conducted study. Observed RR comes from 
    the data analysis, while the RR between the unmeasured confounder and the outcome should
    be obtained from prior literature or constitute an reasonable guess. Probability of exposure
    between the groups should also be reasonable numbers. This function can be adapted to be part
    of a sensitivity analysis with repeated sampling of rr_conf, p1, and p0 from set distributions
    to obtain a range of effects. 
    
    rr_obs:
        -Observed RR from the data
    rr_conf:
        -Value of RR between unmeasured confounder and the outcome of interest
    p1:
        -Estimated proportion of those with unmeasured confounder in the exposed group
    p0:
        -Estimated porportion of those with unmeasured confounder in the unexposed group
    
    Example)
    >>>zepid.sens_analysis.rr_corr(rr_obs=1.5,rr_conf=1.1,p1=0.6,p0=0.4)
    '''
    denom = (p1*(rr_conf-1)+1) / (p0*(rr_conf-1)+1)
    rr_adj = rr_obs / denom
    return rr_adj


def trapezoidal(mini,mode1,mode2,maxi,size=100000,seed=None):
    '''Creates trapezoidal distribution based on Fox & Lash 2005. This function 
    can be used to generate distributions of probabilities and effect measures for
    sensitivity analyses. It is particularly useful when used in conjunction with 
    rr_corr to determine a distribution of potential results due to a single unadjusted
    confounder
    
    mini:
        -minimum value of trapezoidal distribution
    mode1:
        -Start of uniform distribution
    mode2:
        -End of uniform distribution
    maxi:
        -maximum value of trapezoidal distribution
    size:
        -number of observations to generate
    seed:
        -specify a seed for reproducible results. Default is None
    
    Example)
    >>>zepid.sens_analysis.trapezoidal(mini=0.2,mode1=0.3,mode2=0.5,maxi=0.6,size=3,seed=1234)
    '''
    if seed != None:
        np.random.seed(seed)
    tzf = pd.DataFrame()
    tzf['p'] = np.random.uniform(size=size)
    tzf['v'] = (tzf.p*(maxi+mode2-mini-mode1)+(mini+mode1)) / 2
    tzf.loc[tzf['v'] < mode1,'v'] = mini + np.sqrt((mode1-mini)*(2*tzf.v-mini-mode1))
    tzf.loc[tzf['v'] > mode2,'v'] = maxi - np.sqrt(2*(maxi-mode2)*(tzf.v-mode2))
    return tzf['v']


def delta_beta(df,eq,beta,model='glm',match='',family=sm.families.family.Binomial(sm.families.links.logit),group=False,groupvar=''):
    '''Delta-beta is a sensitivity analysis that tracks the change in the beta estimate(s) of interest 
    when a single observation is excluded from the dataframe for all the observations. This function 
    uses statsmodels to calculate betas. All observations and the difference in estimates is stored 
    in a pandas dataframe that is returned by the function. Multiple delta betas are able to estimated
    at once. All beta(s) of interest should be included in a list. Difference in estimates is calculated 
    by substracting the full model results from the reduced results. Returns dataframe of requested betas

    NOTE: that if a delta beta is missing in the returned dataframe, this indicates the model had convergence
    issues when that observation (or group of observations) was removed from the model.
    
    Currently supported model options include:
        -Generalized linear model
        -Generalized estimation equations
    
    df:
        -Dataframe of observations to use for delta beta analysis
    eq:
        -Regression model formula
    beta:
        -Which beta's are of interest. Single string (variable name) or a list of strings that designate
         variable names of beta(s) of interest
    model:
        -Whether to use GLM or GEE. Default is GLM
    match:
        -Variable to match observations on for a GEE model
    group:
        -Whether to drop groups. Default is False, which drops individual observations rather than by 
         dropping groups of observations. If group is set to True, groupvar must be specified
    groupvar:
        -Variable which to group by to conduct the delta beta analysis. group must be set to True for this variable to be used.
    
    Example)
    >>>zepid.sens_analysis.delta_beta(df=data,eq='D ~ X + var1 + var2',beta='X')
    '''
    if type(beta) is not list:
        raise ValueError("Input 'beta' must be a list object")
    if model=='glm':
        fmodel = smf.glm(eq,df,family=family).fit()
    elif model=='gee':
        if match == '':
            raise ValueError('Please specify matching variable for GEE')
        else:
            fmodel = smf.gee(eq,df,match,family=family).fit()
    else:
        raise ValueError('Please specify a supported model')
    dbr = {}
    if group == False:
        for i in range(len(df)):
            dfs = df.drop(df.index[i])
            try:
                if model=='glm':
                    rmodel = smf.glm(eq,dfs,family=family).fit()
                elif model=='gee':
                    rmodel = smf.gee(eq,dfs,match,family=family).fit()
                for b in beta:
                    dbr.setdefault(b,[]).append(rmodel.params[b])
            except:
                for b in beta:
                    dbr.setdefault(b,[]).append(np.nan)
        rf = pd.DataFrame.from_dict(dbr)
    if group == True:
        if groupvar == '':
            raise ValueError('Must specify group variable to drop observations by')
        for i in list(df[groupvar].unique()):
            dfs = df[df[groupvar]!=i]
            try:
                if model=='glm':
                    rmodel = smf.glm(eq,dfs,family=family).fit()
                elif model=='gee':
                    rmodel = smf.gee(eq,dfs,match,family=family).fit()
                for b in beta:
                    dbr.setdefault(b,[]).append(rmodel.params[b])
            except:
                for b in beta:
                    dbr.setdefault(b,[]).append(np.nan)
        rf = pd.DataFrame.from_dict(dbr)
    for b in beta:
        rf[b] -= fmodel.params[b]
    return rf


