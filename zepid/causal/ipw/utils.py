# utilities for IPW methods. Basically for propensity_score(), which they all use

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import links

def propensity_score(df, model, print_results=True):
    '''Generate propensity scores (probability) based on the model input. Uses logistic regression model
    to calculate

    Returns fitted propensity score model

    df:
        -Dataframe for the model
    model:
        -Model to fit the logistic regression to. Example) 'y ~ var1 + var2'
    print_results:
        -Whether to print the logistic regression results. Default is True

    Example)
    >>>zepid.ipw.propensity_score(df=data,model='X ~ Z + var1 + var2')
    '''
    f = sm.families.family.Binomial(sm.families.links.logit)
    log = smf.glm(model, df, family=f).fit()
    if print_results == True:
        print('\n----------------------------------------------------------------')
        print('MODEL: ' + model)
        print('-----------------------------------------------------------------')
        print(log.summary())
    return log
