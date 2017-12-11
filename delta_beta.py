import numpy as np 
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import family
from statsmodels.genmod.families import links

def delta_beta(df,eq,beta,model='glm',match='',family=sm.families.family.Binomial(sm.families.links.logit),group=False,groupvar=''):
    '''Delta-beta is a sensitivity analysis that tracks the change in the beta estimate(s) of interest 
    when a single observation is excluded from the dataframe for all the observations. This function 
    uses statsmodels to calculate betas. All observations and the difference in estimates is stored 
    in a pandas dataframe that is returned by the function. Multiple delta betas are able to estimated
    at once. All beta(s) of interest should be included in a list. Difference in estimates is calculated 
    by substracting the full model results from the reduced results
    
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
    group:
        -Whether to drop groups. Default is False, which drops individual observations rather than by 
         dropping groups of observations. If group is set to True, groupvar must be specified
    groupvar:
        -Variable which to group by to conduct the delta beta analysis. group must be set to True for this variable to be used.
        
    Return: dataframe of requested betas in a pandas dataframe
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


