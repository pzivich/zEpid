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
