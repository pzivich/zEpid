import statsmodels.api as sm
import statsmodels.formula.api as smf


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
    >>>from zepid.causal.utils import propensity_score
    >>>df = load_sample_data(timevary=False)
    >>>propensity_score(df=df,model='dead ~ art0 + male + dvl0')
    """
    f = sm.families.family.Binomial()
    if weights is None:
        log = smf.glm(model, df, family=f).fit()
    else:
        log = smf.glm(model, df, freq_weights=df[weights], family=f).fit()

    if print_results:
        print('\n----------------------------------------------------------------')
        print('MODEL: ' + model)
        print('-----------------------------------------------------------------')
        print(log.summary())
    return log


def exposure_machine_learner(xdata, ydata, ml_model, print_results=True):
    """Function to fit machine learning predictions. Used by TMLE to generate predicted probabilities of being
    treated (i.e. Pr(A=1 | L))
    """
    # Trying to fit the Machine Learning model
    try:
        fm = ml_model.fit(X=xdata, y=ydata)
    except TypeError:
        raise TypeError("Currently custom_model must have the 'fit' function with arguments 'X', 'y'. This "
                        "covers both sklearn and supylearner. If there is a predictive model you would "
                        "like to use, please open an issue at https://github.com/pzivich/zepid and I "
                        "can work on adding support")
    if print_results and hasattr(fm, 'summarize'):  # SuPyLearner has a nice summarize function
        fm.summarize()

    # Generating predictions
    if hasattr(fm, 'predict_proba'):
        g = fm.predict_proba(xdata)[:, 1]
        return g
    elif hasattr(fm, 'predict'):
        g = fm.predict(xdata)
        return g
    else:
        raise ValueError("Currently custom_model must have 'predict' or 'predict_proba' attribute")


def outcome_machine_learner(xdata, ydata, all_a, none_a, ml_model, continuous, print_results=True):
    """Function to fit machine learning predictions. Used by TMLE to generate predicted probabilities of outcome
    (i.e. Pr(Y=1 | A=1, L) and Pr(Y=1 | A=0, L)). Future update will include continuous Y functionality (i.e. E(Y))
    """
    # Trying to fit Machine Learning model
    try:
        fm = ml_model.fit(X=xdata, y=ydata)
    except TypeError:
        raise TypeError("Currently custom_model must have the 'fit' function with arguments 'X', 'y'. This "
                        "covers both sklearn and supylearner. If there is a predictive model you would "
                        "like to use, please open an issue at https://github.com/pzivich/zepid and I "
                        "can work on adding support")
    if print_results and hasattr(fm, 'summarize'):  # Nice summarize option from SuPyLearner
        fm.summarize()

    # Generating predictions
    if continuous:
        if hasattr(fm, 'predict'):
            qa1 = fm.predict(all_a)
            qa0 = fm.predict(none_a)
            return qa1, qa0
        else:
            raise ValueError("Currently custom_model must have 'predict' or 'predict_proba' attribute")

    else:
        if hasattr(fm, 'predict_proba'):
            qa1 = fm.predict_proba(all_a)[:, 1]
            qa0 = fm.predict_proba(none_a)[:, 1]
            return qa1, qa0
        elif hasattr(fm, 'predict'):
            qa1 = fm.predict(all_a)
            qa0 = fm.predict(none_a)
            return qa1, qa0
        else:
            raise ValueError("Currently custom_model must have 'predict' or 'predict_proba' attribute")


def missing_machine_learner(xdata, mdata, all_a, none_a, ml_model, print_results=True):
    """Function to fit machine learning predictions. Used by TMLE to generate predicted probabilities of missing
     outcome data, Pr(M=1|A,L)
    """
    # Trying to fit the Machine Learning model
    try:
        fm = ml_model.fit(X=xdata, y=mdata)
    except TypeError:
        raise TypeError("Currently custom_model must have the 'fit' function with arguments 'X', 'y'. This "
                        "covers both sklearn and supylearner. If there is a predictive model you would "
                        "like to use, please open an issue at https://github.com/pzivich/zepid and I "
                        "can work on adding support")
    if print_results and hasattr(fm, 'summarize'):  # SuPyLearner has a nice summarize function
        fm.summarize()

    # Generating predictions
    if hasattr(fm, 'predict_proba'):
        ma1 = fm.predict_proba(all_a)[:, 1]
        ma0 = fm.predict_proba(none_a)[:, 1]
        return ma1, ma0
    elif hasattr(fm, 'predict'):
        ma1 = fm.predict(all_a)
        ma0 = fm.predict(none_a)
        return ma1, ma0
    else:
        raise ValueError("Currently custom_model must have 'predict' or 'predict_proba' attribute")
