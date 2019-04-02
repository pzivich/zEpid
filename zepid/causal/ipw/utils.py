# utilities for IPW methods. Basically for propensity_score(), which they all use

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import links


def propensity_score(df, model, weights=None, print_results=True):
    """Generate propensity scores (probability) based on the model input. Uses logistic regression model
    to calculate

    Parameters
    -----------
    df : DataFrame
        Pandas Dataframe containing the variables of interest
    model : str
        Model to fit the logistic regression to. For example, 'y ~ var1 + var2'
    weights : str, optional
        Whether to estimate the model using weights. Default is None (unweighted)
    print_results : bool, optional
        Whether to print the logistic regression results. Default is True

    Returns
    -------------
    Fitted statsmodels GLM object

    Example
    ------------
    >>>from zepid import load_sample_data
    >>>from zepid.causal.ipw import propensity_score
    >>>df = load_sample_data(timevary=False)
    >>>propensity_score(df=df,model='dead ~ art0 + male + dvl0')
    """
    f = sm.families.family.Binomial()
    if weights is None:
        log = smf.glm(model, df, family=f).fit()
    else:
        log = smf.gee(model, df.index, df, weights=df[weights], family=f).fit()

    if print_results:
        print('\n----------------------------------------------------------------')
        print('MODEL: ' + model)
        print('-----------------------------------------------------------------')
        print(log.summary())
    return log
