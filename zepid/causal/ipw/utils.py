# utilities for IPW methods. Basically for propensity_score(), which they all use

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import links


def propensity_score(df, model, print_results=True):
    """Generate propensity scores (probability) based on the model input. Uses logistic regression model
    to calculate

    Parameters
    -----------
    df : DataFrame
        Pandas Dataframe containing the variables of interest
    model : str
        Model to fit the logistic regression to. For example, 'y ~ var1 + var2'
    print_results : bool, optional
        Whether to print the logistic regression results. Default is True

    Returns
    -------------
    Fitted statsmodels GLM object

    Example
    ------------
    >>>import zepid as ze
    >>>df = ze.load_sample_data(timevary=False)
    >>>ze.causal.ipw.propensity_score(df=df,model='dead ~ art0 + male + dvl0')
    """
    f = sm.families.family.Binomial(sm.families.links.logit)
    log = smf.glm(model, df, family=f).fit()
    if print_results:
        print('\n----------------------------------------------------------------')
        print('MODEL: ' + model)
        print('-----------------------------------------------------------------')
        print(log.summary())
    return log
