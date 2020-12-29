import warnings
import patsy
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import logistic
from scipy.stats.kde import gaussian_kde
from statsmodels.stats.weightstats import DescrStatsW
import matplotlib.pyplot as plt

from zepid.calc import probability_to_odds, probability_bounds


def check_input_data(data, exposure, outcome, estimator, drop_censoring, drop_missing, binary_exposure_only):
    """Background function used by the various estimators to check the input data for possible issues or
    inconsistencies with the format expected by the estimator.

    Parameters
    ----------
    data : DataFrame
        Input dataframe to be processed
    exposure : str
        Exposure/treatment column name
    outcome : str
        Outcome column
    estimator : str
        Name of the input estimator. Reads back the estimator name in the warning system
    drop_censoring : bool
        Whether to drop censored observations. Some estimators do not handle censoring (so this is set to True)
    drop_missing : bool
        Argument currently does nothing. Will be utilized in the future (when some estimators support missing data
        aside from censoring)
    binary_exposure_only : bool
        Argument for whether only binary exposure / treatments are supported. This is currently True for most
        estimators

    Returns
    -------
    formatted DataFrame, boolean for missing data flag
    """
    # Handling missing data and censoring
    if drop_censoring:  # Censoring is not always supported by all estimators
        valid_obs = data.dropna(subset=[d for d in data.columns]).shape[0]
        if valid_obs != data.shape[0]:
            warnings.warn("There is missing data in the dataset. By default, " + str(estimator) +
                          " will drop all missing data (including missing outcome data). " + str(estimator) +
                          " will fit " + str(valid_obs) + ' of ' + str(data.shape[0]) + ' observations', UserWarning)
            data = data.copy().dropna().reset_index()
        else:
            data = data.copy().reset_index()
        miss_flag = False
        data['__missing_indicator__'] = 1
    else:
        valid_obs = data.dropna(subset=[d for d in data.columns if d != outcome]).shape[0]
        # Checking for other missing data
        if valid_obs != data.shape[0]:
            warnings.warn("There is missing data that is not the outcome in the data set. " + str(estimator) +
                          " will drop all missing data that is not missing outcome data. " + str(estimator) +
                          " will fit " + str(valid_obs) + " of " + str(data.shape[0]) + " observations", UserWarning)
            data = data.copy().dropna(subset=[d for d in data.columns if d != outcome]).reset_index()
        else:
            data = data.copy().reset_index()
        # Checking for censored data
        if valid_obs != data.dropna(subset=[outcome]).shape[0]:
            miss_flag = True
            data['__missing_indicator__'] = np.where(data[outcome].isna(), 0, 1)
        else:
            miss_flag = False
            data['__missing_indicator__'] = 1

    # Checking for only binary exposures
    if binary_exposure_only:
        if not data[exposure].value_counts().index.isin([0, 1]).all():
            raise ValueError(str(estimator) + " only supports binary exposures currently")

    # Check outcome data type
    if data[outcome].value_counts().index.isin([0, 1]).all():
        continuous = False
    else:
        continuous = True

    return data, miss_flag, continuous


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
        print('==============================================================================')
        print('Propensity Score Model')
        print(log.summary())
        print('==============================================================================')
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
    if print_results and hasattr(fm, 'summary'):  # SuPyLearner has a nice summarize function
        print('==============================================================================')
        print('Propensity Score Model')
        fm.summary()
        print('==============================================================================')

    # Generating predictions
    if hasattr(fm, 'predict_proba'):
        g = fm.predict_proba(xdata)
        # this allows support for pygam LogisticGAM, which only returns only 1 probability
        if g.ndim == 1:
            return g
        else:
            return g[:, 1]
    elif hasattr(fm, 'predict'):
        return fm.predict(xdata)
    else:
        raise ValueError("Currently custom_model must have 'predict' or 'predict_proba' attribute")


def outcome_machine_learner(xdata, ydata, all_a, none_a, ml_model, continuous, print_results=True):
    """Function to fit machine learning predictions. Used by TMLE to generate predicted probabilities of outcome
    (i.e. Pr(Y=1 | A=1, L) and Pr(Y=1 | A=0, L)).
    """
    # Trying to fit Machine Learning model
    try:
        fm = ml_model.fit(X=xdata, y=ydata)
    except TypeError:
        raise TypeError("Currently custom_model must have the 'fit' function with arguments 'X', 'y'. This "
                        "covers both sklearn and supylearner. If there is a predictive model you would "
                        "like to use, please open an issue at https://github.com/pzivich/zepid and I "
                        "can work on adding support")
    if print_results and hasattr(fm, 'summary'):  # Nice summarize option from SuPyLearner
        print('==============================================================================')
        print('Outcome Model')
        fm.summary()
        print('==============================================================================')

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
            qa1 = fm.predict_proba(all_a)
            qa0 = fm.predict_proba(none_a)
            if (qa1.ndim == 1) and (qa0.ndim == 1):
                return qa1, qa0
            else:
                return qa1[:, 1], qa0[:, 1]
        elif hasattr(fm, 'predict'):
            qa1 = fm.predict(all_a)
            qa0 = fm.predict(none_a)
            return qa1, qa0
        else:
            raise ValueError("Currently custom_model must have 'predict' or 'predict_proba' attribute")


def stochastic_outcome_machine_learner(xdata, ydata, ml_model, continuous, print_results=True):
    """Function to fit machine learning predictions. Used by StochasticTMLE to generate predicted probabilities of
    outcome (i.e. Pr(Y=1 | A, L)
    """
    # Trying to fit Machine Learning model
    try:
        fm = ml_model.fit(X=xdata, y=ydata)
    except TypeError:
        raise TypeError("Currently custom_model must have the 'fit' function with arguments 'X', 'y'. This "
                        "covers both sklearn and supylearner. If there is a predictive model you would "
                        "like to use, please open an issue at https://github.com/pzivich/zepid and I "
                        "can work on adding support")
    if print_results and hasattr(fm, 'summary'):  # Nice summarize option from SuPyLearner
        print('==============================================================================')
        print('Outcome Model')
        fm.summary()
        print('==============================================================================')

    # Generating predictions
    if continuous:
        if hasattr(fm, 'predict'):
            qa = fm.predict(xdata)
            return qa, fm
        else:
            raise ValueError("Currently custom_model must have 'predict' or 'predict_proba' attribute")

    else:
        if hasattr(fm, 'predict_proba'):
            qa = fm.predict_proba(xdata)
            if qa.ndim == 1:  # Allows for PyGAM
                return qa, fm
            else:
                return qa[:, 1], fm
        elif hasattr(fm, 'predict'):
            qa = fm.predict(xdata)
            return qa, fm
        else:
            raise ValueError("Currently custom_model must have 'predict' or 'predict_proba' attribute")


def stochastic_outcome_predict(xdata, fit_ml_model, continuous):
    """Function to generate predictions machine learning predictions. Used by StochasticTMLE to generate predicted
    probabilities of outcome (i.e. Pr(Y=1 | A=a*, L) in the Monte-Carlo integration procedure
    """
    # Generating predictions
    if continuous:
        if hasattr(fit_ml_model, 'predict'):
            qa_star = fit_ml_model.predict(xdata)
            return qa_star
        else:
            raise ValueError("Currently custom_model must have 'predict' or 'predict_proba' attribute")
    else:
        if hasattr(fit_ml_model, 'predict_proba'):
            qa_star = fit_ml_model.predict_proba(xdata)
            if qa_star.ndim == 1:  # Allows for PyGAM
                return qa_star
            else:
                return qa_star[:, 1]
        elif hasattr(fit_ml_model, 'predict'):
            qa_star = fit_ml_model.predict(xdata)
            return qa_star
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
    if print_results and hasattr(fm, 'summary'):  # SuPyLearner has a nice summarize function
        print('==============================================================================')
        print('Censoring Model')
        fm.summary()
        print('==============================================================================')

    # Generating predictions
    if hasattr(fm, 'predict_proba'):
        ma1 = fm.predict_proba(all_a)
        ma0 = fm.predict_proba(none_a)
        if (ma1.ndim == 1) and (ma0.ndim == 1):
            return ma1, ma0
        else:
            return ma1[:, 1], ma0[:, 1]
    elif hasattr(fm, 'predict'):
        ma1 = fm.predict(all_a)
        ma0 = fm.predict(none_a)
        return ma1, ma0
    else:
        raise ValueError("Currently custom_model must have 'predict' or 'predict_proba' attribute")


def iptw_calculator(df, treatment, model_denom, model_numer, weight, stabilized, standardize, bound, print_results):
    """Background function to calculate inverse probability of treatment weights. Used by `IPTW`, `AIPTW`, `IPSW`,
    `AIPSW`
    """
    denominator_model = propensity_score(df, treatment + ' ~ ' + model_denom,
                                         weights=weight, print_results=print_results)
    d = denominator_model.predict(df)

    # Calculating numerator probabilities (if stabilized)
    if stabilized is True:
        numerator_model = propensity_score(df, treatment + ' ~ ' + model_numer,
                                           weights=weight, print_results=print_results)
        n = numerator_model.predict(df)
    else:
        if model_numer != '1':
            raise ValueError('Argument for model_numerator is only used for stabilized=True')
        n = 1

    # Bounding predicted probabilities if requested
    if bound:
        d = probability_bounds(d, bounds=bound)
        n = probability_bounds(n, bounds=bound)

    # Calculating weights
    if stabilized:  # Stabilized weights
        if standardize == 'population':
            iptw = np.where(df[treatment] == 1, (n / d), ((1 - n) / (1 - d)))
            iptw = np.where(df[treatment].isna(), np.nan, iptw)
        # Stabilizing to exposed (compares all exposed if they were exposed versus unexposed)
        elif standardize == 'exposed':
            iptw = np.where(df[treatment] == 1, 1, (d / (1 - d)) * ((1 - n) / n))
            iptw = np.where(df[treatment].isna(), np.nan, iptw)
        # Stabilizing to unexposed (compares all unexposed if they were exposed versus unexposed)
        else:
            iptw = np.where(df[treatment] == 1, (((1 - d) / d) * (n / (1 - n))), 1)
            iptw = np.where(df[treatment].isna(), np.nan, iptw)

    else:  # Unstabilized weights
        if standardize == 'population':
            iptw = np.where(df[treatment] == 1, 1 / d, 1 / (1 - d))
            iptw = np.where(df[treatment].isna(), np.nan, iptw)
        # Stabilizing to exposed (compares all exposed if they were exposed versus unexposed)
        elif standardize == 'exposed':
            iptw = np.where(df[treatment] == 1, 1, (d / (1 - d)))
            iptw = np.where(df[treatment].isna(), np.nan, iptw)
        # Stabilizing to unexposed (compares all unexposed if they were exposed versus unexposed)
        else:
            iptw = np.where(df[treatment] == 1, ((1 - d) / d), 1)
            iptw = np.where(df[treatment].isna(), np.nan, iptw)
    return d, n, iptw


def plot_kde(df, treatment, probability,
             measure='probability', bw_method='scott', fill=True, color_e='b', color_u='r'):
    """Generates a density plot that can be used to check whether positivity may be violated qualitatively. The
    kernel density used is SciPy's Gaussian kernel. Either Scott's Rule or Silverman's Rule can be implemented.
    Alternative option to the boxplot of probabilities

    Parameters
    ------------
    df : DataFrame
        Pandas dataframe containing the variables of interest
    treatment : str
        Column name of the treatment variable
    probability : str
        Column name of the predicted probability of treatment
    measure : str, optional
        Measure to plot. Options include either the probabilities or log-odds stratified by treatment received.
        Default is probabilities (measure='probability'). Log-odds can be requested via measure='logit'
    bw_method : str, optional
        Method used to estimate the bandwidth. Following SciPy, either 'scott' or 'silverman' are valid options
    fill : bool, optional
        Whether to color the area under the density curves. Default is true
    color_e : str, optional
        Color of the line/area for the treated group. Default is Blue
    color_u : str, optional
        Color of the line/area for the treated group. Default is Red

    Returns
    ---------------
    matplotlib axes
    """
    if measure == 'probability':
        x = np.linspace(0, 1, 10000)
        density_t = gaussian_kde(df.loc[df[treatment] == 1][probability].dropna(),
                                 bw_method=bw_method)
        density_u = gaussian_kde(df.loc[df[treatment] == 0][probability].dropna(),
                                 bw_method=bw_method)
    elif measure == 'logit':
        t = np.log(probability_to_odds(df.loc[df[treatment] == 1][probability].dropna()))
        density_t = gaussian_kde(t, bw_method=bw_method)

        u = np.log(probability_to_odds(df.loc[df[treatment] == 0][probability].dropna()))
        density_u = gaussian_kde(u, bw_method=bw_method)
        x = np.linspace(np.min((np.min(t), np.min(u))) - 1, np.max((np.max(t), np.max(u))) + 1, 10000)
    else:
        raise ValueError("Only plots of probabilities or log-odds are supported. Please specify either "
                         "'probability' or 'logit'")

    ax = plt.gca()
    if fill:
        ax.fill_between(x, density_t(x), color=color_e, alpha=0.2, label=None)
        ax.fill_between(x, density_u(x), color=color_u, alpha=0.2, label=None)
    ax.plot(x, density_t(x), color=color_e, label='Treat = 1')
    ax.plot(x, density_u(x), color=color_u, label='Treat = 0')
    if measure == 'probability':
        ax.set_xlabel('Probability')
    else:
        ax.set_xlabel('Log-Odds')
    ax.set_ylabel('Density')
    ax.legend()
    return ax


def plot_boxplot(df, treatment, probability, measure='probability'):
    """Generates a stratified boxplot that can be used to visually check whether positivity may be violated,
    qualitatively. Alternative option to the kernel density plot.

    Parameters
    ----------
    df : DataFrame
        Pandas dataframe containing the variables of interest
    treatment : str
        Column name of the treatment variable
    probability : str
        Column name of the predicted probability of treatment

    measure : str, optional
        Measure to plot. Options include either the probabilities or log-odds stratified by treatment received.
        Default is probabilities (measure='probability'). Log-odds can be requested via measure='logit'

    Returns
    -------------
    matplotlib axes
    """
    if measure == 'probability':
        boxes = (df.loc[df[treatment] == 1][probability].dropna(),
                 df.loc[df[treatment] == 0][probability].dropna())

    elif measure == 'logit':
        boxes = (np.log(probability_to_odds(df.loc[df[treatment] == 1][probability].dropna())),
                 np.log(probability_to_odds(df.loc[df[treatment] == 0][probability].dropna())))
    else:
        raise ValueError("Only plots of probabilities or log-odds are supported. Please specify either "
                         "'probability' or 'logit")

    labs = ['A = 1', 'A = 0']
    meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='black')
    ax = plt.gca()
    ax.boxplot(boxes, labels=labs, meanprops=meanpointprops, showmeans=True)
    if measure == 'probability':
        ax.set_ylabel('Probability')
        ax.set_ylim([0, 1])
    else:
        ax.set_ylabel('Log-Odds')
    return ax


def positivity(df, weights):
    """Use this to assess whether positivity is a valid assumption. For stabilized weights, the mean weight should
    be approximately 1. For unstabilized weights, the mean weight should be approximately 2. If there are extreme
    outliers, this may indicate problems with the calculated weights

    Parameters
    --------------
    df : DataFrame
        Pandas dataframe containing the variables of interest
    weights : str
        Column name of the inverse probability of treatment weights

    Returns
    --------------
    tuple
        Tuple of positivity results; mean, SD, min, max
    """
    pos_avg = float(np.mean(df[weights].dropna()))
    pos_max = np.max(df[weights].dropna())
    pos_min = np.min(df[weights].dropna())
    pos_sd = float(np.std(df[weights].dropna()))
    return pos_avg, pos_sd, pos_min, pos_max


def standardized_mean_differences(df, treatment, weight, formula):
    """Calculates the standardized mean differences for all variables. Default calculates the standardized mean
    difference for all variables included in the IPTW denominator

    Parameters
    ----------
    df : DataFrame
        Pandas dataframe containing the variables of interest
    treatment : str
        Column name of the treatment variable
    weight : str
        Column name of the inverse probability of treatment weights
    formula : str
        Formula for weights model following patsy syntax

    Returns
    -------
    DataFrame
        Returns pandas DataFrame of calculated standardized mean differences. Columns are labels (variables labels),
        smd_u (unweighted standardized difference), and smd_w (weighted standardized difference)
    iptw_only : bool, optional
        Whether the diagnostic should be run on IPTW only or the weights multiplied together. Default is IPTW only
    """
    def _standardized_difference_(df, treatment, var_type, weight, weighted=True):
        """Background function to calculate the standardized mean difference between the treat and untreated for a
        specified variable. Useful for checking whether a confounder was balanced between the two treatment groups
        by the specified IPTW model SMD based on: Austin PC 2011; https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3144483/
        """
        def _categorical_cov_(a, b):
            """Turns out, pandas and numpy don't have the correct covariance matrix I need for categorical variables.
            The covariance matrix is defined as

            S = [S_{kl}] = (P_{1k}*(1-P_{1k}) + P_{2k}*(1-P{2k})) / 2     if k == l
                           (P_{1k}*P_{1l} + P_{2k}*P_{2l}) / 2            if k != l
            """
            cv2 = []
            for i, v in enumerate(a):
                cv1 = []
                if i == 0:
                    pass
                else:
                    for j, w in enumerate(b):
                        if j == 0:
                            pass
                        elif i == j:
                            cv1.append((v * (1 - v) + w * (1 - w)) / 2)
                        else:
                            cv1.append((a[i] * a[j] + b[i] * b[j]) / -2)
                    cv2.append(cv1)

            return np.array(cv2)

        # Pulling out relevant data
        dft = df.loc[(df[treatment] == 1) & (df[weight].notnull())].copy()
        dfn = df.loc[(df[treatment] == 0) & (df[weight].notnull())].copy()
        vcols = list(df.columns)
        vcols.remove(treatment)
        vcols.remove(weight)

        if var_type == 'binary':
            if weighted:
                dwt = DescrStatsW(dft[vcols], weights=dft[weight])
                wt = dwt.mean
                dwn = DescrStatsW(dfn[vcols], weights=dfn[weight])
                wn = dwn.mean
            else:
                wt = np.mean(dft[vcols].dropna(), axis=0)
                wn = np.mean(dfn[vcols].dropna(), axis=0)
            return float((wt - wn) / np.sqrt((wt * (1 - wt) + wn * (1 - wn)) / 2))

        elif var_type == 'continuous':
            if weighted:
                dwt = DescrStatsW(dft[vcols], weights=dft[weight], ddof=1)
                wmt = dwt.mean
                wst = dwt.std
                dwn = DescrStatsW(dfn[vcols], weights=dfn[weight], ddof=1)
                wmn = dwn.mean
                wsn = dwn.std
            else:
                dwt = DescrStatsW(dft[vcols], ddof=1)
                wmt = dwt.mean
                wst = dwt.std
                dwn = DescrStatsW(dfn[vcols], ddof=1)
                wmn = dwn.mean
                wsn = dwn.std
            return float((wmt - wmn) / np.sqrt((wst ** 2 + wsn ** 2) / 2))

        elif var_type == 'categorical':
            if weighted:
                wt = np.average(dft[vcols], weights=dft[weight], axis=0)
                wn = np.average(dfn[vcols], weights=dfn[weight], axis=0)
            else:
                wt = np.mean(dft[vcols], axis=0)
                wn = np.mean(dfn[vcols], axis=0)

            t_c = wt - wn
            s_inv = np.linalg.inv(_categorical_cov_(a=wt, b=wn))
            return float(np.sqrt(np.dot(np.transpose(t_c[1:]), np.dot(s_inv, t_c[1:]))))

        else:
            raise ValueError('Not supported')

    variables = patsy.dmatrix(formula + ' - 1', df, return_type='dataframe')
    w_diff = []
    u_diff = []
    vlabel = []

    # Pull out list of terms and the corresponding dataframe slice(s)
    term_dict = variables.design_info.term_name_slices

    # Looping through the terms
    for term in variables.design_info.terms:
        # Adding term labels
        vlabel.append(term.name())

        # Pulling out data corresponding to term
        chunk = term_dict[term.name()]
        v = variables.iloc[:, chunk].copy()

        # Detecting variable type
        if v.shape[1] != 1:
            vtype = 'categorical'
        elif np.all(v.dropna().isin([0, 1])):
            vtype = 'binary'
        else:
            vtype = 'continuous'

        # calculate the absolute standardized difference
        dat = pd.concat([v, df[[treatment, weight]]], axis=1)
        wsmd = _standardized_difference_(df=dat, treatment=treatment, var_type=vtype, weight=weight, weighted=True)
        w_diff.append(wsmd)
        usmd = _standardized_difference_(df=dat, treatment=treatment, var_type=vtype, weight=weight, weighted=False)
        u_diff.append(usmd)

    # Setting up DataFrame to return with calculated differences
    s = pd.DataFrame()
    s['labels'] = vlabel
    s['smd_w'] = w_diff
    s['smd_u'] = u_diff
    return s


def plot_love(df, treatment, weight, formula,
              color_unweighted='r', color_weighted='b', shape_unweighted='o', shape_weighted='o'):
    """Generates a Love-plot to detail covariate balance based on the IPTW weights. Further details on the usage of
    this plot are available in Austin PC & Stuart EA 2015 https://onlinelibrary.wiley.com/doi/full/10.1002/sim.6607

    The Love plot generates a dashed line at standardized mean difference of 0.10. Ideally, weighted SMD are below
    this level. Below 0.20 may also be sufficient. Variables above this level may be unbalanced despite the
    weighting procedure. Different functional forms (or approaches like machine learning) may be worth considering

    Parameters
    ----------
    df : dataframe
        Pandas DataFrame object containing variables of interest
    treatment : str
        Treatment/exposure variable column label
    weight : str
        Column label for the weights
    formula : str
        Right-hand side of the equation for the weights
    color_unweighted : str, optional
        Color for the unweighted standardized mean differences. Default is red
    color_weighted : str, optional
        Color for the weighted standardized mean differences. Default is blue
    shape_unweighted : str, optional
        Shape of points for the unweighted standardized mean differences. Default is circles
    shape_weighted:
        Shape of points for the weighted standardized mean differences. Default is circles

    Returns
    -------
    axes
        Matplotlib axes of the Love plot
    """
    to_plot = standardized_mean_differences(df=df, treatment=treatment, weight=weight, formula=formula)
    to_plot['smd_w'] = np.absolute(to_plot['smd_w'])
    to_plot['smd_u'] = np.absolute(to_plot['smd_u'])
    to_plot = to_plot.sort_values(by='smd_u', ascending=True).reset_index(drop=True)

    # Generate plot
    ax = plt.gca()
    ax.plot(to_plot.smd_u, to_plot.index, shape_unweighted, c=color_unweighted, label='Unweighted')
    ax.plot(to_plot.smd_w, to_plot.index, shape_weighted, c=color_weighted, label='Weighted')
    ax.set_xlim([0, np.max([np.max(to_plot['smd_w']), np.max(to_plot['smd_u'])]) + 0.5])
    ax.set_xlabel('Absolute Standardized Difference')
    ax.axvline(0.1, color='gray')
    ax.set_yticks([i for i in range(to_plot.shape[0])])
    ax.set_yticklabels(to_plot['labels'])
    ax.legend()
    return ax


def outcome_accuracy(true, predicted, decimal=3):
    """Diagnostic for Q-models. Compares the observed outcome to the predicted.

    Parameters
    ----------
    true:
        True (observed) values
    predicted:
        Predicted values
    decimal : int, optional
        Number of decimal places to display. Default is three
    """
    value = predicted - true

    print('======================================================================')
    print('                 Natural Course Prediction Accuracy')
    print('======================================================================')
    print('Outcome model accuracy summary statistics. Defined as the predicted\n'
          'outcome value minus the observed outcome value')
    print('----------------------------------------------------------------------')
    print('Mean value:           ', np.round(np.mean(value), decimal))
    print('Standard Deviation:    ', np.round(np.std(value, ddof=1), decimal))
    print('Minimum value:        ', np.round(np.min(value), decimal))
    print('Maximum value:        ', np.round(np.max(value), decimal))
    print('======================================================================')


def plot_kde_accuracy(values, bw_method='scott', fill=True, color='b'):
    """Generates a density plot that can be used to check whether positivity may be violated qualitatively. The
    kernel density used is SciPy's Gaussian kernel. Either Scott's Rule or Silverman's Rule can be implemented.
    Alternative option to the boxplot of probabilities

    Parameters
    ----------
    values : str, optional
        Difference between predicted value and observed value
    bw_method : str, optional
        Method used to estimate the bandwidth. Following SciPy, either 'scott' or 'silverman' are valid options
    fill : bool, optional
        Whether to color the area under the density curves. Default is true
    color : str, optional
        Color of the line/area. Default is blue

    Returns
    -------
    matplotlib axes
    """
    x = np.linspace(np.min(values) - 0.5, np.max(values) + 0.5, 10000)
    density = gaussian_kde(values, bw_method=bw_method)

    ax = plt.gca()
    if fill:
        ax.fill_between(x, density(x), color=color, alpha=0.2, label=None)
    ax.plot(x, density(x), color=color)
    ax.set_xlabel(r'$Y_{pred} - Y$')
    ax.set_ylabel('Density')
    return ax


def stochastic_check_conditional(df, conditional):
    """Check that conditionals are exclusive for the stochastic fit process. Generates a warning if not true
    """
    a = np.zeros(df.shape[0])
    for c in conditional:
        a = a + np.where(eval(c), 1, 0)

    if np.any(a > 1):
        warnings.warn("It looks like your conditional categories are NOT exclusive. For appropriate estimation, "
                      "the conditions that designate each category should be exclusive", UserWarning)


def aipw_calculator(y, a, py_a, py_n, pa1, pa0, difference=True, weights=None, splits=None, continuous=False):
    """Function to calculate AIPW estimates. Called by AIPTW, SingleCrossfitAIPTW, and DoubleCrossfitAIPTW
    """
    # Point estimate calculation
    y1 = np.where(a == 1, (y - py_a*(1 - pa1)) / pa1, py_a)
    y0 = np.where(a == 0, (y - py_n*(1 - pa0)) / pa0, py_n)

    # Warning system if values are out of range
    if not continuous:
        if np.mean(y1) > 1 or np.mean(y1) < 0:
            warnings.warn("The estimated probability for all-exposed is out of the bounds (less than zero or greater "
                          "than 1). This may indicate positivity issues resulting from extreme weights, too small of a "
                          "sample size, or too flexible of models. Try setting the optional `bound` argument. If using "
                          "DoubleCrossfitAIPTW, try SingleCrossfitAIPTW or the TMLE estimators instead.", UserWarning)
        if np.mean(y0) > 1 or np.mean(y0) < 0:
            warnings.warn("The estimated probability for none-exposed is out of the bounds (less than zero or greater "
                          "than 1). This may indicate positivity issues resulting from extreme weights, too small of a "
                          "sample size, or too flexible of models. Try setting the optional `bound` argument. If using "
                          "DoubleCrossfitAIPTW, try SingleCrossfitAIPTW or the TMLE estimators instead.", UserWarning)

    # Calculating ACE as a difference
    if difference:
        if weights is None:
            estimate = np.nanmean(y1 - y0)
            if splits is None:
                var = np.nanvar((y1 - y0) - estimate, ddof=1) / y.shape[0]
            else:
                var_rd = []
                for i in set(splits):
                    y1s = y1[splits == i]
                    y0s = y0[splits == i]
                    var_rd.append(np.var((y1s - y0s) - estimate, ddof=1))
                var = np.mean(var_rd) / y.shape[0]
        else:
            estimate = DescrStatsW(y1, weights=weights).mean - DescrStatsW(y0, weights=weights).mean
            var = np.nan

    # Calculating ACE as a ratio
    else:
        if weights is None:
            estimate = np.nanmean(y1) / np.nanmean(y0)
            if estimate < 0:
                warnings.warn("lower than 0", UserWarning)
            py_o = a*py_a + (1-a)*py_n
            ic = ((a*(y-py_o)) / (np.mean(py_a)*pa1) + (py_a - np.mean(py_a)) -
                  ((1-a)*(y-py_o)) / (np.mean(py_n)*pa0) + (py_n - np.mean(py_n)))
            var = np.nanvar(ic, ddof=1) / y.shape[0]
        else:
            estimate = DescrStatsW(y1, weights=weights).mean / DescrStatsW(y0, weights=weights).mean
            var = np.nan

    return estimate, var
