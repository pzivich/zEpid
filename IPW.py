import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import family
from statsmodels.genmod.families import links

def ipw(df,miss_model,model,idv,model_type='logistic'):
    '''Calculate the main effect measure of interest using Inverse Probability 
    Weighting (IPW). Weights are calculated from the inverse predicted probabilities 
    from logistic regression model. Fits a GEE model with indepedent dependence structure 
    (same process used in SAS to obtain IPW estimate) to generate the estimate.
    Based on statsmodels to fit the models.
    
    Returns fitted regression model. Prints model fit summary
    
    df:
        -pandas dataframe containing the variables of interest
    miss_model:
        -model specification for predicting missing data by the 
         variables to include in the weights. Uses statsmodels format:
         'missing ~ var1 + var2 + var3'
    model:
        -regression model of interest based on the weights generated from
         IPW. Uses statsmodels format: 'outcome ~ exposure'
    idv:
        -unique ID variable for observations. Used for clusters in GEE.
    model_type:
        -type of regression model to fit. Options are:
                Model                       Keyword
            Logistic regression (OR):            'logistic'
            Log-risk regression (PR/RR):         'log-risk'
            Linear-risk regression (PD/RD):      'linear-risk'
    '''
    mm = sm.families.family.Binomial(sm.families.links.logit) #specify logistic regression for IPW 
    log = smf.glm(miss_model,df,family=mm).fit()
    w = log.predict()
    w = w**-1
    print(w)
    if model_type == 'logistic':
        efm = sm.families.family.Binomial(sm.families.links.logit)
    elif model_type == 'log-risk':
        efm = sm.families.family.Binomial(sm.families.links.log)
    elif model_type == 'linear-risk':
        efm = sm.families.family.Binomial(sm.families.links.identity)
    else:
        print('Please use a valid model')
    ind = sm.cov_struct.Independence()
    ipw = smf.gee(model,idv,df,cov_struct=ind,family=efm,weights=w).fit()
    print(ipw.summary())
    return ipw


