import copy
import warnings
import patsy
import numpy as np
import pandas as pd
from scipy.stats import logistic, norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from numpy.random import RandomState

from zepid.calc.utils import probability_bounds, probability_to_odds, odds_to_probability
from zepid.causal.utils import check_input_data, aipw_calculator
from zepid.causal.doublyrobust.utils import tmle_unit_unbound, tmle_unit_bounds


class SingleCrossfitAIPTW:
    """Implementation of the Augmented Inverse Probability Weighting estimator with a cross-fit procedure. The purpose
    of the cross-fit procedure is to all for non-Donsker nuisance function estimators. Some of machine learning
    algorithms are non-Donsker. In practice this means that confidence interval coverage can be incorrect when certain
    nuisance function estimators are used. Additionally, bias may persist as well. Cross-fitting is meant to alleviate
    this issue, therefore cross-fitting with a doubly-robust estimator is recommended when using machine learning.

    `SingleCrossfitAIPTW` uses a single cross-fit procedure, where the data set is paritioned into at least two
    non-overlapping splits. The nuisance function estimators are then estimated in each split. The estimated nuisance
    functions are then used to predict values in a non-overlapping split. This decouple the nuisance function estimation
    from the data used to estimate it

    Note
    ----
    Because of the repetitions of the procedure are needed to reduce variance determined by a particular partition, it
    can take a long time to run this code.

    Parameters
    ----------
    df : DataFrame
        Pandas dataframe containing all necessary variables
    exposure : str
        Label for treatment column in the pandas data frame
    outcome : str
        Label for outcome column in the pandas data frame
    alpha : float, optional
        Alpha for confidence interval level. Default is 0.05

    Examples
    --------
    Setting up environment

    >>> from sklearn.linear_model import LogisticRegression
    >>> from zepid import load_sample_data
    >>> from zepid.causal.doublyrobust import SingleCrossfitAIPTW
    >>> df = load_sample_data(False).drop(columns='cd4_wk45').dropna()

    Estimating the single cross-fit AIPTW

    >>> scaipw = SingleCrossfitAIPTW(df, exposure='art', outcome='dead')
    >>> scaipw.exposure_model("male + age0 + cd40 + dvl0", estimator=LogisticRegression(solver='lbfgs'))
    >>> scaipw.outcome_model("art + male + age0 + cd40 + dvl0", estimator=LogisticRegression(solver='lbfgs'))
    >>> scaipw.fit(n_splits=5, n_partitions=100)
    >>> scaipw.summary()

    References
    ----------
    Chernozhukov V, Chetverikov D, Demirer M, Duflo E, Hansen C, Newey W, & Robins J. (2018). "Double/debiased machine
    learning for treatment and structural parameters". The Econometrics Journal 21:1; pC1–C6
    """
    def __init__(self, df, exposure, outcome, alpha=0.05):
        self.exposure = exposure
        self.outcome = outcome
        self.df, self._miss_flag, self._continuous_outcome_ = check_input_data(data=df,
                                                                               exposure=exposure,
                                                                               outcome=outcome,
                                                                               estimator="SingleCrossfitAIPTW",
                                                                               drop_censoring=True,
                                                                               drop_missing=True,
                                                                               binary_exposure_only=True)
        self.alpha = alpha

        self._a_covariates = None
        self._y_covariates = None
        self._a_estimator = None
        self._y_estimator = None
        self._fit_treatment_ = False
        self._fit_outcome_ = False
        self._gbounds = None
        self._n_splits_ = 0
        self._n_partitions = 0
        self._combine_method_ = None

        self.ace_vector = None
        self.ace_var_vector = None
        self.ace = None
        self.ace_ci = None
        self.ace_se = None

        self.risk_difference_vector = None
        self.risk_difference_var_vector = None
        self.risk_difference = None
        self.risk_difference_ci = None
        self.risk_difference_se = None

        self.risk_ratio_vector = None
        self.risk_ratio_var_vector = None
        self.risk_ratio = None
        self.risk_ratio_ci = None
        self.risk_ratio_se = None

    def exposure_model(self, covariates, estimator, bound=False):
        """Specify the treatment nuisance model variables and estimator(s) to use. These parameters are held
        in the background until the .fit() function is called. These approaches are for used each sample split

        Parameters
        ----------
        covariates : str
            Confounders to include in the propensity score model. Follows patsy notation
        estimator :
            Estimator to use for prediction of the propensity score
        bound : float, list, optional
            Whether to bound predicted probabilities. Default is False, which does not bound
        """
        self._a_estimator = estimator
        self._a_covariates = covariates
        self._fit_treatment_ = True
        self._gbounds = bound

    def outcome_model(self, covariates, estimator):
        """Specify the outcome nuisance model variables and estimator(s) to use. These parameters are held
        in the background until the .fit() function is called. These approaches are for used each sample split

        Parameters
        ----------
        covariates : str
            Confounders to include in the propensity score model. Follows patsy notation
        estimator :
            Estimator to use for prediction of the propensity score
        """
        self._y_estimator = estimator
        self._y_covariates = covariates
        self._fit_outcome_ = True

    def fit(self, n_splits=2, n_partitions=100, method='median', random_state=None):
        """Runs the crossfit estimation procedure with augmented inverse probability weighting estimator. The
        estimation process is completed for multiple different splits during the procedure. The final estimate is
        defined as either the median or mean of the causal measure from each of the different splits. Median is
        used as the default since it is more stable.

        Note
        ----
        `n_partition` should be kept high to reduce dependency of results on the chosen number of splits

        Confidence intervals come from influences curves and incorporates the within-split variance and between-split
        variance.

        Parameters
        ----------
        n_splits : int
            Number of splits to use with a default of 2. The number of splits must be greater than or equal to 2.
        n_partitions : int
            Number of times to repeat the partition process. The default is 100, which I have seen good performance
            with in the past. Note that this algorithm can take a long time to run for high values of this parameter.
            It is best to test out run-times on small numbers first. Also if running in parallel, it can be reduced
        method : str, optional
            Method to obtain point estimates and standard errors. Median method takes the median (which is more robust)
            and the mean takes the mean. It has been remarked that the median is preferred, since it is more stable to
            extreme outliers, which may happen in finite samples
        random_state : None, int, optional
            Whether to set a seed for the partitions. Default is None (which does not use a user-set seed). Any valid
            NumPy seed can be input. Note that you should also state the random_state of all (applicable) estimators
            to ensure replicability. Seeds are chosen by the following procedure. The input random_state is based to
            np.random.choice to select n_partitions between 0 and 5million. That list of n_partition-length is then
            passed to each iteration of the cross-fitting pandas.DataFrame.sample(random_state).
        """
        # Checking for various issues
        if not self._fit_treatment_:
            raise ValueError("exposure_model() must be called before fit()")
        if not self._fit_outcome_:
            raise ValueError("outcome_model() must be called before fit()")
        if n_splits < 2:
            raise ValueError("SingleCrossfitAIPTW requires that n_splits > 1")

        # Storing some information
        self._n_splits_ = n_splits
        self._n_partitions = n_partitions
        self._combine_method_ = method

        # Creating blank lists
        diff_est, diff_var, ratio_est, ratio_var = [], [], [], []

        # Conducts the re-sampling procedure
        if random_state is None:
            random_state = [None] * n_partitions
        else:
            random_state = RandomState(random_state).choice(range(5000000), size=n_partitions, replace=False)

        for j in range(self._n_partitions):
            # Estimating for a particular split (lots of functions happening in the background)
            result = self._single_crossfit_(random_state=random_state[j])

            # Appending results of this particular split combination
            diff_est.append(result[0])
            diff_var.append(result[1])
            if not self._continuous_outcome_:
                ratio_est.append(result[2])
                ratio_var.append(result[3])

        # Obtaining overall estimate and (1-alpha)% CL from all splits
        zalpha = norm.ppf(1 - self.alpha / 2, loc=0, scale=1)

        est, var = calculate_joint_estimate(diff_est, diff_var, method=method)
        if self._continuous_outcome_:
            self.ace_vector = diff_est
            self.ace_var_vector = diff_var
            self.ace = est
            self.ace_se = np.sqrt(var)
            self.ace_ci = (self.ace - zalpha*self.ace_se,
                           self.ace + zalpha*self.ace_se)
        else:
            # Risk Difference
            self.risk_difference_vector = diff_est
            self.risk_difference_var_vector = diff_var
            self.risk_difference = est
            self.risk_difference_se = np.sqrt(var)
            self.risk_difference_ci = (self.risk_difference - zalpha*self.risk_difference_se,
                                       self.risk_difference + zalpha*self.risk_difference_se)
            # Risk Ratio
            self.risk_ratio_vector = ratio_est
            self.risk_ratio_var_vector = ratio_var
            ln_rr, ln_rr_var = calculate_joint_estimate(np.log(self.risk_ratio_vector),
                                                        self.risk_ratio_var_vector, method=method)
            self.risk_ratio = np.exp(ln_rr)
            self.risk_ratio_se = np.sqrt(ln_rr_var)
            self.risk_ratio_ci = (np.exp(ln_rr - zalpha*self.risk_ratio_se),
                                  np.exp(ln_rr + zalpha*self.risk_ratio_se))

    def summary(self, decimal=3):
        """Prints summary of model results

        Parameters
        ----------
        decimal : int, optional
            Number of decimal places to display. Default is 3
        """
        if (self._fit_outcome_ is False) or (self._fit_treatment_ is False):
            raise ValueError('exposure_model and outcome_model must be specified before the estimate can '
                             'be generated')

        print('======================================================================')
        print('                     Single Cross-fit AIPTW                           ')
        print('======================================================================')
        fmt = 'Treatment:        {:<15} No. Observations:     {:<20}'
        print(fmt.format(self.exposure, self.df.shape[0]))
        fmt = 'Outcome:          {:<15} No. of Splits:        {:<20}'
        print(fmt.format(self.outcome, self._n_splits_))
        fmt = 'Method:           {:<15} No. of Partitions:    {:<20}'
        print(fmt.format(self._combine_method_, self._n_partitions))

        print('======================================================================')
        if self._continuous_outcome_:
            print('Average Causal Effect: ', round(float(self.ace), decimal))
            print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: (' +
                  str(round(self.ace_ci[0], decimal)), ',',
                  str(round(self.ace_ci[1], decimal)) + ')')
        else:
            print('Risk Difference:    ', round(float(self.risk_difference), decimal))
            print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: (' +
                  str(round(self.risk_difference_ci[0], decimal)), ',',
                  str(round(self.risk_difference_ci[1], decimal)) + ')')
            print('----------------------------------------------------------------------')
            print('Risk Ratio:         ', round(float(self.risk_ratio), decimal))
            print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: (' +
                  str(round(self.risk_ratio_ci[0], decimal)), ',',
                  str(round(self.risk_ratio_ci[1], decimal)) + ')')

        print('======================================================================')

    def run_diagnostics(self, color='gray'):
        """Runs available diagnostics for the plots. Currently diagnostics consist of a plot of the different point
        estimates and variance estimates across different partitions. Diagnostics for cross-fit estimators is ongoing.
        If you have any suggestions, please feel free to contact me on GitHub

        Parameters
        ----------
        color : str, optional
            Controls color of the plots. Default is gray

        Returns
        -------
        Plot to console
        """
        # Continuous outcomes have less plots to generate
        if self._continuous_outcome_:
            _run_diagnostic_(diff=self.ace_vector, diff_var=self.ace_var_vector,
                             color=color)

        # Binary outcomes have plots for all measures
        else:
            _run_diagnostic_(diff=self.risk_difference_vector, diff_var=self.risk_difference_var_vector,
                             rratio=self.risk_ratio_vector, rratio_var=self.risk_ratio_var_vector,
                             color=color)

    def _single_crossfit_(self, random_state):
        """Background function that runs a single crossfit of the split samples
        """
        # Dividing into s different splits
        sample_split = _sample_split_(self.df, n_splits=self._n_splits_, random_state=random_state)

        # Determining pairings to use for each sample split and each combination
        pairing_exposure = [i - 1 for i in range(self._n_splits_)]
        pairing_outcome = pairing_exposure

        # Estimating treatment nuisance model
        a_models = _treatment_nuisance_(treatment=self.exposure, estimator=self._a_estimator,
                                        samples=sample_split, covariates=self._a_covariates)
        # Estimating outcome nuisance model
        y_models = _outcome_nuisance_(outcome=self.outcome, estimator=self._y_estimator,
                                      samples=sample_split, covariates=self._y_covariates)

        # Generating predictions based on set pairs for cross-fit procedure
        predictions = []
        y_obs, a_obs = np.array([]), np.array([])
        split_index = []
        for id, ep, op in zip(range(self._n_splits_), pairing_exposure, pairing_outcome):
            predictions.append(self._generate_predictions_(sample_split[id],
                                                           a_model_v=a_models[ep],
                                                           y_model_v=y_models[op]))
            # Generating vector of Y in correct order
            y_obs = np.append(y_obs, np.asarray(sample_split[id][self.outcome]))
            # Generating vector of A in correct order
            a_obs = np.append(a_obs, np.asarray(sample_split[id][self.exposure]))
            # Generating index for splits
            split_index.extend([id]*sample_split[id].shape[0])

        # Stacking Predicted Pr(A=1), Y(a=1), Y(a=0)
        pred_a_array, pred_y1_array, pred_y0_array = np.array([]), np.array([]), np.array([])
        for preds in predictions:
            pred_a_array = np.append(pred_a_array, preds[0])
            pred_y1_array = np.append(pred_y1_array, preds[1])
            pred_y0_array = np.append(pred_y0_array, preds[2])

        # Applying bounds if requested
        if self._gbounds:  # Bounding g-model if requested
            pred_a_array = probability_bounds(pred_a_array, bounds=self._gbounds)

        # Calculating point estimates
        difference, var_diff = aipw_calculator(y=y_obs, a=a_obs,
                                               py_a=pred_y1_array, py_n=pred_y0_array,
                                               pa1=pred_a_array, pa0=1-pred_a_array,
                                               splits=np.asarray(split_index),
                                               difference=True, continuous=self._continuous_outcome_)
        if self._continuous_outcome_:
            return difference, var_diff
        else:
            ratio, var_ratio = aipw_calculator(y=y_obs, a=a_obs,
                                               py_a=pred_y1_array, py_n=pred_y0_array,
                                               pa1=pred_a_array, pa0=1 - pred_a_array,
                                               splits=np.asarray(split_index),
                                               difference=False, continuous=False)
            return difference, var_diff, ratio, var_ratio

    def _generate_predictions_(self, sample, a_model_v, y_model_v):
        """Generates predictions from fitted functions (in background of _single_crossfit()
        """
        s = sample.copy()

        # Predicting Pr(A=1|L)
        xdata = np.asarray(patsy.dmatrix(self._a_covariates + ' - 1', s))
        a_pred = _ml_predictor(xdata, fitted_algorithm=a_model_v)

        # Predicting E(Y|A=1, L)
        s[self.exposure] = 1
        xdata = np.asarray(patsy.dmatrix(self._y_covariates + ' - 1', s))
        y_treat = _ml_predictor(xdata, fitted_algorithm=y_model_v)

        # Predicting E(Y|A=0, L)
        s[self.exposure] = 0
        xdata = np.asarray(patsy.dmatrix(self._y_covariates + ' - 1', s))
        y_none = _ml_predictor(xdata, fitted_algorithm=y_model_v)

        return a_pred, y_treat, y_none


class DoubleCrossfitAIPTW:
    """Implementation of the augmented inverse probability weighted estimator with a double cross-fit procedure. The
    purpose of the cross-fit procedure is to all for non-Donsker nuisance function estimators. Some of machine learning
    algorithms are non-Donsker. In practice this means that confidence interval coverage can be incorrect when certain
    nuisance function estimators are used. Additionally, bias may persist as well. Cross-fitting is meant to alleviate
    this issue, therefore cross-fitting with a doubly-robust estimator is recommended when using machine learning.

    `DoubleCrossfitAIPTW` allows for double cross-fitting, where the data set is partitioned into at least three
    non-overlapping splits. The nuisance function estimators are then estimated in each split. The estimated nuisance
    functions are then used to predict values in the opposing split. Different splits are used for each nuisance
    function. A double cross-fit procedure further de-couples the nuisance function estimation compared to single
    cross-fit procedures.

    Note
    ----
    Because of the repetitions of the procedure are needed to reduce variance determined by a particular partition, it
    can take a long time to run this code. On a data set of 3000 observations with 100 different partitions it takes
    about an hour. The advantage is that the code can be ran in parallel. See the documentation for an example.

    Parameters
    ----------
    df : DataFrame
        Pandas dataframe containing all necessary variables
    exposure : str
        Label for treatment column in the pandas data frame
    outcome : str
        Label for outcome column in the pandas data frame
    alpha : float, optional
        Alpha for confidence interval level. Default is 0.05

    Examples
    --------
    Setting up environment

    >>> from sklearn.linear_model import LogisticRegression
    >>> from zepid import load_sample_data
    >>> from zepid.causal.doublyrobust import SingleCrossfitAIPTW
    >>> df = load_sample_data(False).drop(columns='cd4_wk45').dropna()

    Estimating the double cross-fit AIPTW

    >>> dcaipw = DoubleCrossfitAIPTW(df, exposure='art', outcome='dead')
    >>> dcaipw.exposure_model("male + age0 + cd40 + dvl0", estimator=LogisticRegression(solver='lbfgs'))
    >>> dcaipw.outcome_model("art + male + age0 + cd40 + dvl0", estimator=LogisticRegression(solver='lbfgs'))
    >>> dcaipw.fit(n_splits=5, n_partitions=100)
    >>> dcaipw.summary()

    References
    ----------
    Newey WK, Robins JR. (2018) "Cross-fitting and fast remainder rates for semiparametric estimation".
    arXiv:1801.09138

    Zivich PN, & Breskin A. (2020). Machine learning for causal inference: on the use of cross-fit estimators.
    arXiv preprint arXiv:2004.10337.

    Chernozhukov V, Chetverikov D, Demirer M, Duflo E, Hansen C, Newey W, & Robins J. (2018). "Double/debiased machine
    learning for treatment and structural parameters". The Econometrics Journal 21:1; pC1–C6
    """
    def __init__(self, df, exposure, outcome, alpha=0.05):
        self.exposure = exposure
        self.outcome = outcome
        self.df, self._miss_flag, self._continuous_outcome_ = check_input_data(data=df,
                                                                               exposure=exposure,
                                                                               outcome=outcome,
                                                                               estimator="DoubleCrossfitAIPTW",
                                                                               drop_censoring=True,
                                                                               drop_missing=True,
                                                                               binary_exposure_only=True)
        self.alpha = alpha

        self._a_covariates = None
        self._y_covariates = None
        self._a_estimator = None
        self._y_estimator = None
        self._fit_treatment_ = False
        self._fit_outcome_ = False
        self._gbounds = None
        self._n_splits_ = 0
        self._n_partitions = 0
        self._combine_method_ = None

        self.ace_vector = None
        self.ace_var_vector = None
        self.ace = None
        self.ace_ci = None
        self.ace_se = None

        self.risk_difference_vector = None
        self.risk_difference_var_vector = None
        self.risk_difference = None
        self.risk_difference_ci = None
        self.risk_difference_se = None

        self.risk_ratio_vector = None
        self.risk_ratio_var_vector = None
        self.risk_ratio = None
        self.risk_ratio_ci = None
        self.risk_ratio_se = None

    def exposure_model(self, covariates, estimator, bound=False):
        """Specify the treatment nuisance model variables and estimator(s) to use. These parameters are held
        in the background until the .fit() function is called. These approaches are for used each sample split

        Parameters
        ----------
        covariates : str
            Confounders to include in the propensity score model. Follows patsy notation
        estimator :
            Estimator to use for prediction of the propensity score
        bound : float, list, optional
            Whether to bound predicted probabilities. Default is False, which does not bound
        """
        self._a_estimator = estimator
        self._a_covariates = covariates
        self._fit_treatment_ = True
        self._gbounds = bound

    def outcome_model(self, covariates, estimator):
        """Specify the outcome nuisance model variables and estimator(s) to use. These parameters are held
        in the background until the .fit() function is called. These approaches are for used each sample split

        Parameters
        ----------
        covariates : str
            Confounders to include in the propensity score model. Follows patsy notation
        estimator :
            Estimator to use for prediction of the propensity score
        """
        self._y_estimator = estimator
        self._y_covariates = covariates
        self._fit_outcome_ = True

    def fit(self, n_splits=3, n_partitions=100, method='median', random_state=None):
        """Runs the crossfit estimation procedure with augmented inverse probability weighted estimator. The
        estimation process is completed for multiple different splits during the procedure. The final estimate is
        defined as either the median or mean of the average causal effect from each of the different splits. Median is
        used as the default since it is more stable.

        Note
        ----
        `n_partition` should be kept high to reduce dependency of results on the chosen number of splits

        Confidence intervals come from influences curves and incorporates the within-split variance and between-split
        variance.

        Parameters
        ----------
        n_splits : int
            Number of splits to use. The default is 3, which is valid for both single cross-fit and double cross-fit.
            Single cross-fit is also compatible with 2 as the the number of splits
        n_partitions : int
            Number of times to repeat the partition process. The default is 100, which I have seen good performance
            with in the past. Note that this algorithm can take a long time to run for high values of this parameter.
            It is best to test out run-times on small numbers first. Also if running in parallel, it can be reduced
        method : str, optional
            Method to obtain point estimates and standard errors. Median method takes the median (which is more robust)
            and the mean takes the mean. It has been remarked that the median is preferred, since it is more stable to
            extreme outliers, which may happen in finite samples
        random_state : None, int, optional
            Whether to set a seed for the partitions. Default is None (which does not use a user-set seed). Any valid
            NumPy seed can be input. Note that you should also state the random_state of all (applicable) estimators
            to ensure replicability. Seeds are chosen by the following procedure. The input random_state is based to
            np.random.choice to select n_partitions between 0 and 5million. That list of n_partition-length is then
            passed to each iteration of the cross-fitting pandas.DataFrame.sample(random_state).
        """
        # Checking for various issues
        if not self._fit_treatment_:
            raise ValueError("exposure_model() must be called before fit()")
        if not self._fit_outcome_:
            raise ValueError("outcome_model() must be called before fit()")
        if n_splits < 3:
            raise ValueError("DoubleCrossfitAIPTW requires that n_splits >= 3")

        # Storing some information
        self._n_splits_ = n_splits
        self._n_partitions = n_partitions
        self._combine_method_ = method

        # Creating blank lists
        diff_est, diff_var, ratio_est, ratio_var = [], [], [], []

        # Conducts the re-sampling procedure
        if random_state is None:
            random_state = [None] * n_partitions
        else:
            random_state = RandomState(random_state).choice(range(5000000), size=n_partitions, replace=False)
        for j in range(self._n_partitions):
            # Estimating for a particular split (lots of functions happening in the background)
            result = self._single_crossfit_(random_state=random_state[j])

            # Appending results of this particular split combination
            diff_est.append(result[0])
            diff_var.append(result[1])
            if not self._continuous_outcome_:
                ratio_est.append(result[2])
                ratio_var.append(result[3])

        # Obtaining overall estimate and (1-alpha)% CL from all splits
        zalpha = norm.ppf(1 - self.alpha / 2, loc=0, scale=1)

        est, var = calculate_joint_estimate(diff_est, diff_var, method=method)
        if self._continuous_outcome_:
            self.ace_vector = diff_est
            self.ace_var_vector = diff_var
            self.ace = est
            self.ace_se = np.sqrt(var)
            self.ace_ci = (self.ace - zalpha*self.ace_se,
                           self.ace + zalpha*self.ace_se)
        else:
            # Risk Difference
            self.risk_difference_vector = diff_est
            self.risk_difference_var_vector = diff_var
            self.risk_difference = est
            self.risk_difference_se = np.sqrt(var)
            self.risk_difference_ci = (self.risk_difference - zalpha*self.risk_difference_se,
                                       self.risk_difference + zalpha*self.risk_difference_se)
            # Risk Ratio
            self.risk_ratio_vector = ratio_est
            self.risk_ratio_var_vector = ratio_var
            ln_rr, ln_rr_var = calculate_joint_estimate(np.log(self.risk_ratio_vector),
                                                        self.risk_ratio_var_vector, method=method)
            self.risk_ratio = np.exp(ln_rr)
            self.risk_ratio_se = np.sqrt(ln_rr_var)
            self.risk_ratio_ci = (np.exp(ln_rr - zalpha*self.risk_ratio_se),
                                  np.exp(ln_rr + zalpha*self.risk_ratio_se))

    def summary(self, decimal=3):
        """Prints summary of model results

        Parameters
        ----------
        decimal : int, optional
            Number of decimal places to display. Default is 3
        """
        if (self._fit_outcome_ is False) or (self._fit_treatment_ is False):
            raise ValueError('exposure_model and outcome_model must be specified before the estimate can '
                             'be generated')

        print('======================================================================')
        print('                     Double Cross-fit AIPTW                           ')
        print('======================================================================')
        fmt = 'Treatment:        {:<15} No. Observations:     {:<20}'
        print(fmt.format(self.exposure, self.df.shape[0]))
        fmt = 'Outcome:          {:<15} No. of Splits:        {:<20}'
        print(fmt.format(self.outcome, self._n_splits_))
        fmt = 'Method:           {:<15} No. of Partitions:    {:<20}'
        print(fmt.format(self._combine_method_, self._n_partitions))

        print('======================================================================')
        if self._continuous_outcome_:
            print('Average Causal Effect: ', round(float(self.ace), decimal))
            print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: (' +
                  str(round(self.ace_ci[0], decimal)), ',',
                  str(round(self.ace_ci[1], decimal)) + ')')
        else:
            print('Risk Difference:    ', round(float(self.risk_difference), decimal))
            print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: (' +
                  str(round(self.risk_difference_ci[0], decimal)), ',',
                  str(round(self.risk_difference_ci[1], decimal)) + ')')
            print('----------------------------------------------------------------------')
            print('Risk Ratio:         ', round(float(self.risk_ratio), decimal))
            print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: (' +
                  str(round(self.risk_ratio_ci[0], decimal)), ',',
                  str(round(self.risk_ratio_ci[1], decimal)) + ')')

        print('======================================================================')

    def run_diagnostics(self, color='gray'):
        """Runs available diagnostics for the plots. Currently diagnostics consist of a plot of the different point
        estimates and variance estimates across different partitions. Diagnostics for cross-fit estimators is ongoing.
        If you have any suggestions, please feel free to contact me on GitHub

        Parameters
        ----------
        color : str, optional
            Controls color of the plots. Default is gray

        Returns
        -------
        Plot to console
        """
        # Continuous outcomes have less plots to generate
        if self._continuous_outcome_:
            _run_diagnostic_(diff=self.ace_vector, diff_var=self.ace_var_vector,
                             color=color)

        # Binary outcomes have plots for all measures
        else:
            _run_diagnostic_(diff=self.risk_difference_vector, diff_var=self.risk_difference_var_vector,
                             rratio=self.risk_ratio_vector, rratio_var=self.risk_ratio_var_vector,
                             color=color)

    def _single_crossfit_(self, random_state):
        """Background function that runs a single crossfit of the split samples
        """
        # Dividing into s different splits
        sample_split = _sample_split_(self.df, n_splits=self._n_splits_, random_state=random_state)

        # Determining pairings to use for each sample split and each combination
        pairing_exposure = [i - 1 for i in range(self._n_splits_)]
        pairing_outcome = [i - 2 for i in range(self._n_splits_)]

        # Estimating treatment nuisance model
        a_models = _treatment_nuisance_(treatment=self.exposure, estimator=self._a_estimator,
                                        samples=sample_split, covariates=self._a_covariates)
        # Estimating outcome nuisance model
        y_models = _outcome_nuisance_(outcome=self.outcome, estimator=self._y_estimator,
                                      samples=sample_split, covariates=self._y_covariates)

        # Generating predictions based on set pairs for cross-fit procedure
        predictions = []
        y_obs, a_obs = np.array([]), np.array([])
        split_index = []
        for id, ep, op in zip(range(self._n_splits_), pairing_exposure, pairing_outcome):
            predictions.append(self._generate_predictions_(sample_split[id],
                                                           a_model_v=a_models[ep],
                                                           y_model_v=y_models[op]))
            # Generating vector of Y in correct order
            y_obs = np.append(y_obs, np.asarray(sample_split[id][self.outcome]))
            # Generating vector of A in correct order
            a_obs = np.append(a_obs, np.asarray(sample_split[id][self.exposure]))
            # Generating index for splits
            split_index.extend([id]*sample_split[id].shape[0])

        # Stacking Predicted Pr(A=1), Y(a=1), Y(a=0)
        pred_a_array, pred_y1_array, pred_y0_array = np.array([]), np.array([]), np.array([])
        for preds in predictions:
            pred_a_array = np.append(pred_a_array, preds[0])
            pred_y1_array = np.append(pred_y1_array, preds[1])
            pred_y0_array = np.append(pred_y0_array, preds[2])

        # Applying bounds if requested
        if self._gbounds:  # Bounding g-model if requested
            pred_a_array = probability_bounds(pred_a_array, bounds=self._gbounds)

        # Calculating point estimates
        difference, var_diff = aipw_calculator(y=y_obs, a=a_obs,
                                               py_a=pred_y1_array, py_n=pred_y0_array,
                                               pa1=pred_a_array, pa0=1-pred_a_array,
                                               splits=np.asarray(split_index),
                                               difference=True, continuous=self._continuous_outcome_)
        if self._continuous_outcome_:
            return difference, var_diff
        else:
            ratio, var_ratio = aipw_calculator(y=y_obs, a=a_obs,
                                               py_a=pred_y1_array, py_n=pred_y0_array,
                                               pa1=pred_a_array, pa0=1 - pred_a_array,
                                               splits=np.asarray(split_index),
                                               difference=False, continuous=False)
            return difference, var_diff, ratio, var_ratio

    def _generate_predictions_(self, sample, a_model_v, y_model_v):
        """Generates predictions from fitted functions (in background of _single_crossfit()
        """
        s = sample.copy()

        # Predicting Pr(A=1|L)
        xdata = np.asarray(patsy.dmatrix(self._a_covariates + ' - 1', s))
        a_pred = _ml_predictor(xdata, fitted_algorithm=a_model_v)

        # Predicting E(Y|A=1, L)
        s[self.exposure] = 1
        xdata = np.asarray(patsy.dmatrix(self._y_covariates + ' - 1', s))
        y_treat = _ml_predictor(xdata, fitted_algorithm=y_model_v)

        # Predicting E(Y|A=0, L)
        s[self.exposure] = 0
        xdata = np.asarray(patsy.dmatrix(self._y_covariates + ' - 1', s))
        y_none = _ml_predictor(xdata, fitted_algorithm=y_model_v)

        return a_pred, y_treat, y_none


class SingleCrossfitTMLE:
    """Implementation of the Targeted Maximum Likelihood Estimator with a single cross-fit procedure. The purpose
    of the cross-fit procedure is to all for non-Donsker nuisance function estimators. Some of machine learning
    algorithms are non-Donsker. In practice this means that confidence interval coverage can be incorrect when certain
    nuisance function estimators are used. Additionally, bias may persist as well. Cross-fitting is meant to alleviate
    this issue, therefore cross-fitting with a doubly-robust estimator is recommended when using machine learning.

    `SingleCrossfitTMLE` uses a single cross-fit, where the data set is paritioned into at least two non-overlapping
    splits. The nuisance function estimators are then estimated in each split. The estimated nuisance functions are
    then used to predict values in a non-overlapping split. This decouple the nuisance function estimation from the
    data used to estimate it

    Note
    ----
    Because of the repetitions of the procedure are needed to reduce variance determined by a particular partition, it
    can take a long time to run this code.

    Parameters
    ----------
    df : DataFrame
        Pandas dataframe containing all necessary variables
    exposure : str
        Label for treatment column in the pandas data frame
    outcome : str
        Label for outcome column in the pandas data frame
    alpha : float, optional
        Alpha for confidence interval level. Default is 0.05
    continuous_bound : float, optional
        Optional argument to control the bounding feature for continuous outcomes. The bounding process may result
        in values of 0,1 which are undefined for logit(x). This parameter adds or substracts from the scenarios of
        0,1 respectively. Default value is 0.0005

    Examples
    --------
    Setting up environment

    >>> from sklearn.linear_model import LogisticRegression
    >>> from zepid import load_sample_data
    >>> from zepid.causal.doublyrobust import SingleCrossfitTMLE
    >>> df = load_sample_data(False).drop(columns='cd4_wk45').dropna()

    Estimating the single cross-fit TMLE

    >>> sctmle = SingleCrossfitTMLE(df, exposure='art', outcome='dead')
    >>> sctmle.exposure_model("male + age0 + cd40 + dvl0", estimator=LogisticRegression(solver='lbfgs'))
    >>> sctmle.outcome_model("art + male + age0 + cd40 + dvl0", estimator=LogisticRegression(solver='lbfgs'))
    >>> sctmle.fit(n_splits=5, n_partitions=100)
    >>> sctmle.summary()

    References
    ----------
    Chernozhukov V, Chetverikov D, Demirer M, Duflo E, Hansen C, Newey W, & Robins J. (2018). "Double/debiased machine
    learning for treatment and structural parameters". The Econometrics Journal 21:1; pC1–C6
    """
    def __init__(self, df, exposure, outcome, alpha=0.05, continuous_bound=0.0005):
        self.exposure = exposure
        self.outcome = outcome
        self.df, self._miss_flag, self._continuous_outcome_ = check_input_data(data=df,
                                                                               exposure=exposure,
                                                                               outcome=outcome,
                                                                               estimator="SingleCrossfitTMLE",
                                                                               drop_censoring=True,
                                                                               drop_missing=True,
                                                                               binary_exposure_only=True)
        self.alpha = alpha

        # bounding for continuous Y
        if self._continuous_outcome_:
            self._continuous_min = np.min(self.df[outcome])
            self._continuous_max = np.max(self.df[outcome])
            self._cb = continuous_bound
            self.df[outcome] = tmle_unit_bounds(y=self.df[outcome], mini=self._continuous_min,
                                                maxi=self._continuous_max, bound=self._cb)
        else:
            self._cb = 0.0

        self._a_covariates = None
        self._y_covariates = None
        self._a_estimator = None
        self._y_estimator = None
        self._fit_treatment_ = False
        self._fit_outcome_ = False
        self._gbounds = None
        self._n_splits_ = 0
        self._n_partitions = 0
        self._combine_method_ = None

        self.ace_vector = None
        self.ace_var_vector = None
        self.ace = None
        self.ace_ci = None
        self.ace_se = None

        self.risk_difference_vector = None
        self.risk_difference_var_vector = None
        self.risk_difference = None
        self.risk_difference_ci = None
        self.risk_difference_se = None

        self.risk_ratio_vector = None
        self.risk_ratio_var_vector = None
        self.risk_ratio = None
        self.risk_ratio_ci = None
        self.risk_ratio_se = None

        self.odds_ratio_vector = None
        self.odds_ratio_var_vector = None
        self.odds_ratio = None
        self.odds_ratio_se = None
        self.odds_ratio_ci = None

    def exposure_model(self, covariates, estimator, bound=False):
        """Specify the treatment nuisance model variables and estimator(s) to use. These parameters are held
        in the background until the .fit() function is called. These approaches are for used each sample split

        Parameters
        ----------
        covariates : str
            Confounders to include in the propensity score model. Follows patsy notation
        estimator :
            Estimator to use for prediction of the propensity score
        bound : float, list, optional
            Whether to bound predicted probabilities. Default is False, which does not bound
        """
        self._a_estimator = estimator
        self._a_covariates = covariates
        self._fit_treatment_ = True
        self._gbounds = bound

    def outcome_model(self, covariates, estimator):
        """Specify the outcome nuisance model variables and estimator(s) to use. These parameters are held
        in the background until the .fit() function is called. These approaches are for used each sample split

        Parameters
        ----------
        covariates : str
            Confounders to include in the propensity score model. Follows patsy notation
        estimator :
            Estimator to use for prediction of the propensity score
        """
        self._y_estimator = estimator
        self._y_covariates = covariates
        self._fit_outcome_ = True

    def fit(self, n_splits=2, n_partitions=100, method='median', random_state=None):
        """Runs the crossfit estimation procedure with the targeted maximum likelihood estimator. The estimation
        process is completed for multiple different splits during the procedure. The final estimate is defined as
        either the median or mean of the causal measure from each of the different splits. Median is used as the
        default since it is more stable.

        Note
        ----
        `n_partition` should be kept high to reduce dependency of results on the chosen number of splits

        Confidence intervals come from influences curves and incorporates the within-split variance and between-split
        variance.

        Parameters
        ----------
        n_splits : int
            Number of splits to use with a default of 2. The number of splits must be greater than or equal to 2.
        n_partitions : int
            Number of times to repeat the partition process. The default is 100, which I have seen good performance
            with in the past. Note that this algorithm can take a long time to run for high values of this parameter.
            It is best to test out run-times on small numbers first. Also if running in parallel, it can be reduced
        method : str, optional
            Method to obtain point estimates and standard errors. Median method takes the median (which is more robust)
            and the mean takes the mean. It has been remarked that the median is preferred, since it is more stable to
            extreme outliers, which may happen in finite samples
        random_state : None, int, optional
            Whether to set a seed for the partitions. Default is None (which does not use a user-set seed). Any valid
            NumPy seed can be input. Note that you should also state the random_state of all (applicable) estimators
            to ensure replicability. Seeds are chosen by the following procedure. The input random_state is based to
            np.random.choice to select n_partitions between 0 and 5million. That list of n_partition-length is then
            passed to each iteration of the cross-fitting pandas.DataFrame.sample(random_state).
        """
        # Checking for various issues
        if not self._fit_treatment_:
            raise ValueError("exposure_model() must be called before fit()")
        if not self._fit_outcome_:
            raise ValueError("outcome_model() must be called before fit()")
        if n_splits < 2:
            raise ValueError("SingleCrossfitTMLE requires that n_splits >= 2")

        # Storing some information
        self._n_splits_ = n_splits
        self._n_partitions = n_partitions
        self._combine_method_ = method

        # Creating blank lists
        diff_est, diff_var, rratio_est, rratio_var, oratio_est, oratio_var = [], [], [], [], [], []

        # Conducts the re-sampling procedure
        if random_state is None:
            random_state = [None] * n_partitions
        else:
            random_state = RandomState(random_state).choice(range(5000000), size=n_partitions, replace=False)
        for j in range(self._n_partitions):
            # Estimating for a particular split (lots of functions happening in the background)
            result = self._single_crossfit_(random_state=random_state[j])

            # Appending results of this particular split combination
            diff_est.append(result[0])
            diff_var.append(result[1])
            if not self._continuous_outcome_:
                rratio_est.append(result[2])
                rratio_var.append(result[3])
                oratio_est.append(result[4])
                oratio_var.append(result[5])

        # Obtaining overall estimate and (1-alpha)% CL from all splits
        zalpha = norm.ppf(1 - self.alpha / 2, loc=0, scale=1)

        est, var = calculate_joint_estimate(diff_est, diff_var, method=method)
        if self._continuous_outcome_:
            self.ace_vector = diff_est
            self.ace_var_vector = diff_var
            self.ace = est
            self.ace_se = np.sqrt(var)
            self.ace_ci = (self.ace - zalpha*self.ace_se,
                           self.ace + zalpha*self.ace_se)
        else:
            # Risk Difference
            self.risk_difference_vector = diff_est
            self.risk_difference_var_vector = diff_var
            self.risk_difference = est
            self.risk_difference_se = np.sqrt(var)
            self.risk_difference_ci = (self.risk_difference - zalpha*self.risk_difference_se,
                                       self.risk_difference + zalpha*self.risk_difference_se)
            # Risk Ratio
            self.risk_ratio_vector = rratio_est
            self.risk_ratio_var_vector = rratio_var
            ln_rr, ln_rr_var = calculate_joint_estimate(np.log(self.risk_ratio_vector),
                                                        self.risk_ratio_var_vector, method=method)
            self.risk_ratio = np.exp(ln_rr)
            self.risk_ratio_se = np.sqrt(ln_rr_var)
            self.risk_ratio_ci = (np.exp(ln_rr - zalpha*self.risk_ratio_se),
                                  np.exp(ln_rr + zalpha*self.risk_ratio_se))
            # Odds Ratio
            self.odds_ratio_vector = oratio_est
            self.odds_ratio_var_vector = oratio_var
            ln_or, ln_or_var = calculate_joint_estimate(np.log(self.odds_ratio_vector),
                                                        self.odds_ratio_var_vector, method=method)
            self.odds_ratio = np.exp(ln_or)
            self.odds_ratio_se = np.sqrt(ln_or_var)
            self.odds_ratio_ci = (np.exp(ln_or - zalpha*self.odds_ratio_se),
                                  np.exp(ln_or + zalpha*self.odds_ratio_se))

    def summary(self, decimal=3):
        """Prints summary of model results

        Parameters
        ----------
        decimal : int, optional
            Number of decimal places to display. Default is 3
        """
        if (self._fit_outcome_ is False) or (self._fit_treatment_ is False):
            raise ValueError('exposure_model and outcome_model must be specified before the estimate can '
                             'be generated')

        print('======================================================================')
        print('                     Single Cross-fit TMLE                            ')
        print('======================================================================')
        fmt = 'Treatment:        {:<15} No. Observations:     {:<20}'
        print(fmt.format(self.exposure, self.df.shape[0]))
        fmt = 'Outcome:          {:<15} No. of Splits:        {:<20}'
        print(fmt.format(self.outcome, self._n_splits_))
        fmt = 'Method:           {:<15} No. of Partitions:    {:<20}'
        print(fmt.format(self._combine_method_, self._n_partitions))

        print('======================================================================')
        if self._continuous_outcome_:
            print('Average Causal Effect: ', round(float(self.ace), decimal))
            print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: (' +
                  str(round(self.ace_ci[0], decimal)), ',',
                  str(round(self.ace_ci[1], decimal)) + ')')
        else:
            print('Risk Difference:    ', round(float(self.risk_difference), decimal))
            print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: (' +
                  str(round(self.risk_difference_ci[0], decimal)), ',',
                  str(round(self.risk_difference_ci[1], decimal)) + ')')
            print('----------------------------------------------------------------------')
            print('Risk Ratio:         ', round(float(self.risk_ratio), decimal))
            print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: (' +
                  str(round(self.risk_ratio_ci[0], decimal)), ',',
                  str(round(self.risk_ratio_ci[1], decimal)) + ')')
            print('----------------------------------------------------------------------')
            print('Odds Ratio:         ', round(float(self.odds_ratio), decimal))
            print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: (' +
                  str(round(self.odds_ratio_ci[0], decimal)), ',',
                  str(round(self.odds_ratio_ci[1], decimal)) + ')')
        print('======================================================================')

    def run_diagnostics(self, color='gray'):
        """Runs available diagnostics for the plots. Currently diagnostics consist of a plot of the different point
        estimates and variance estimates across different partitions. Diagnostics for cross-fit estimators is ongoing.
        If you have any suggestions, please feel free to contact me on GitHub

        Parameters
        ----------
        color : str, optional
            Controls color of the plots. Default is gray

        Returns
        -------
        Plot to console
        """
        # Continuous outcomes have less plots to generate
        if self._continuous_outcome_:
            _run_diagnostic_(diff=self.ace_vector, diff_var=self.ace_var_vector,
                             color=color)

        # Binary outcomes have plots for all measures
        else:
            _run_diagnostic_(diff=self.risk_difference_vector, diff_var=self.risk_difference_var_vector,
                             rratio=self.risk_ratio_vector, rratio_var=self.risk_ratio_var_vector,
                             oratio=self.odds_ratio_vector, oratio_var=self.odds_ratio_var_vector,
                             color=color)

    def _single_crossfit_(self, random_state):
        """Background function that runs a single crossfit of the split samples
        """
        # Dividing into s different splits
        sample_split = _sample_split_(self.df, n_splits=self._n_splits_, random_state=random_state)

        # Determining pairings to use for each sample split and each combination
        pairing_exposure = [i - 1 for i in range(self._n_splits_)]
        pairing_outcome = pairing_exposure

        # Estimating treatment nuisance model
        a_models = _treatment_nuisance_(treatment=self.exposure, estimator=self._a_estimator,
                                        samples=sample_split, covariates=self._a_covariates)
        # Estimating outcome nuisance model
        y_models = _outcome_nuisance_(outcome=self.outcome, estimator=self._y_estimator,
                                      samples=sample_split, covariates=self._y_covariates)

        # Generating predictions based on set pairs for cross-fit procedure
        predictions = []
        y_obs, a_obs = np.array([]), np.array([])
        split_index = []
        for id, ep, op in zip(range(self._n_splits_), pairing_exposure, pairing_outcome):
            predictions.append(self._generate_predictions_(sample_split[id],
                                                           a_model_v=a_models[ep],
                                                           y_model_v=y_models[op]))
            # Generating vector of Y in correct order
            y_obs = np.append(y_obs, np.asarray(sample_split[id][self.outcome]))
            # Generating vector of A in correct order
            a_obs = np.append(a_obs, np.asarray(sample_split[id][self.exposure]))
            # Generating index for splits
            split_index.extend([id]*sample_split[id].shape[0])

        # Stacking Predicted Pr(A=1), Y(a=1), Y(a=0)
        pred_a_array, pred_y1_array, pred_y0_array = np.array([]), np.array([]), np.array([])
        for preds in predictions:
            pred_a_array = np.append(pred_a_array, preds[0])
            pred_y1_array = np.append(pred_y1_array, preds[1])
            pred_y0_array = np.append(pred_y0_array, preds[2])

        # Applying bounds if requested
        if self._gbounds:  # Bounding g-model if requested
            pred_a_array = probability_bounds(pred_a_array, bounds=self._gbounds)

        # Calculating point estimates
        targeted_vals = targeting_step(y=y_obs, a=a_obs,
                                       py_a=pred_y1_array, py_n=pred_y0_array,
                                       pa1=pred_a_array, pa0=1-pred_a_array,
                                       splits=np.asarray(split_index))

        if self._continuous_outcome_:
            difference, var_diff = tmle_calculator(y=y_obs,
                                                   ystar1=targeted_vals[0], ystar0=targeted_vals[1],
                                                   ystara=targeted_vals[2],
                                                   h1w=targeted_vals[3], h0w=targeted_vals[4], haw=targeted_vals[5],
                                                   splits=np.asarray(split_index),
                                                   measure='ate',
                                                   lower_bound=self._continuous_min, upper_bound=self._continuous_max)
            return difference, var_diff
        else:
            difference, var_diff = tmle_calculator(y=y_obs,
                                                   ystar1=targeted_vals[0], ystar0=targeted_vals[1],
                                                   ystara=targeted_vals[2],
                                                   h1w=targeted_vals[3], h0w=targeted_vals[4], haw=targeted_vals[5],
                                                   splits=np.asarray(split_index),
                                                   measure='risk_difference')
            rratio, var_rratio = tmle_calculator(y=y_obs,
                                                 ystar1=targeted_vals[0], ystar0=targeted_vals[1],
                                                 ystara=targeted_vals[2],
                                                 h1w=targeted_vals[3], h0w=targeted_vals[4], haw=targeted_vals[5],
                                                 splits=np.asarray(split_index),
                                                 measure='risk_ratio')
            oratio, var_oratio = tmle_calculator(y=y_obs,
                                                 ystar1=targeted_vals[0], ystar0=targeted_vals[1],
                                                 ystara=targeted_vals[2],
                                                 h1w=targeted_vals[3], h0w=targeted_vals[4], haw=targeted_vals[5],
                                                 splits=np.asarray(split_index),
                                                 measure='odds_ratio')
            return difference, var_diff, rratio, var_rratio, oratio, var_oratio

    def _generate_predictions_(self, sample, a_model_v, y_model_v):
        """Generates predictions from fitted functions (in background of _single_crossfit()
        """
        s = sample.copy()

        # Predicting Pr(A=1|L)
        xdata = np.asarray(patsy.dmatrix(self._a_covariates + ' - 1', s))
        a_pred = _ml_predictor(xdata, fitted_algorithm=a_model_v)

        # Predicting E(Y|A=1, L)
        s[self.exposure] = 1
        xdata = np.asarray(patsy.dmatrix(self._y_covariates + ' - 1', s))
        y_treat = _ml_predictor(xdata, fitted_algorithm=y_model_v)

        # Predicting E(Y|A=0, L)
        s[self.exposure] = 0
        xdata = np.asarray(patsy.dmatrix(self._y_covariates + ' - 1', s))
        y_none = _ml_predictor(xdata, fitted_algorithm=y_model_v)

        return a_pred, y_treat, y_none


class DoubleCrossfitTMLE:
    """Implementation of the Targeted Maximum Likelihood Estimator with a double cross-fit procedure. The purpose
    of the cross-fit procedure is to all for non-Donsker nuisance function estimators. Some of machine learning
    algorithms are non-Donsker. In practice this means that confidence interval coverage can be incorrect when certain
    nuisance function estimators are used. Additionally, bias may persist as well. Cross-fitting is meant to alleviate
    this issue, therefore cross-fitting with a doubly-robust estimator is recommended when using machine learning.

    `DoubleCrossfitTMLE` uses a double cross-fit, where the data set is paritioned into at least three non-overlapping
    split. The nuisance function estimators are then estimated in each split. The estimated nuisance functions are
    then used to predict values in a non-overlapping split. This decouple the nuisance function estimation from the
    data used to estimate it

    Note
    ----
    Because of the repetitions of the procedure are needed to reduce variance determined by a particular partition, it
    can take a long time to run this code.

    Parameters
    ----------
    df : DataFrame
        Pandas dataframe containing all necessary variables
    exposure : str
        Label for treatment column in the pandas data frame
    outcome : str
        Label for outcome column in the pandas data frame
    alpha : float, optional
        Alpha for confidence interval level. Default is 0.05
    continuous_bound : float, optional
        Optional argument to control the bounding feature for continuous outcomes. The bounding process may result
        in values of 0,1 which are undefined for logit(x). This parameter adds or substracts from the scenarios of
        0,1 respectively. Default value is 0.0005

    Examples
    --------
    Setting up environment

    >>> from sklearn.linear_model import LogisticRegression
    >>> from zepid import load_sample_data
    >>> from zepid.causal.doublyrobust import DoubleCrossfitTMLE
    >>> df = load_sample_data(False).drop(columns='cd4_wk45').dropna()

    Estimating the double cross-fit TMLE

    >>> dctmle = DoubleCrossfitTMLE(df, exposure='art', outcome='dead')
    >>> dctmle.exposure_model("male + age0 + cd40 + dvl0", estimator=LogisticRegression(solver='lbfgs'))
    >>> dctmle.outcome_model("art + male + age0 + cd40 + dvl0", estimator=LogisticRegression(solver='lbfgs'))
    >>> dctmle.fit(n_splits=5, n_partitions=100)
    >>> dctmle.summary()

    References
    ----------
    Zivich PN, & Breskin A. (2020). Machine learning for causal inference: on the use of cross-fit estimators.
    arXiv preprint arXiv:2004.10337.

    Newey WK, Robins JR. (2018) "Cross-fitting and fast remainder rates for semiparametric estimation".
    arXiv:1801.09138

    Chernozhukov V, Chetverikov D, Demirer M, Duflo E, Hansen C, Newey W, & Robins J. (2018). "Double/debiased machine
    learning for treatment and structural parameters". The Econometrics Journal 21:1; pC1–C6
    """
    def __init__(self, df, exposure, outcome, alpha=0.05, continuous_bound=0.0005):
        self.exposure = exposure
        self.outcome = outcome
        self.df, self._miss_flag, self._continuous_outcome_ = check_input_data(data=df,
                                                                               exposure=exposure,
                                                                               outcome=outcome,
                                                                               estimator="DoubleCrossfitTMLE",
                                                                               drop_censoring=True,
                                                                               drop_missing=True,
                                                                               binary_exposure_only=True)
        self.alpha = alpha

        # bounding for continuous Y
        if self._continuous_outcome_:
            self._continuous_min = np.min(self.df[outcome])
            self._continuous_max = np.max(self.df[outcome])
            self._cb = continuous_bound
            self.df[outcome] = tmle_unit_bounds(y=self.df[outcome], mini=self._continuous_min,
                                                maxi=self._continuous_max, bound=self._cb)
        else:
            self._cb = 0.0

        self._a_covariates = None
        self._y_covariates = None
        self._a_estimator = None
        self._y_estimator = None
        self._fit_treatment_ = False
        self._fit_outcome_ = False
        self._gbounds = None
        self._n_splits_ = 0
        self._n_partitions = 0
        self._combine_method_ = None

        self.ace_vector = None
        self.ace_var_vector = None
        self.ace = None
        self.ace_ci = None
        self.ace_se = None

        self.risk_difference_vector = None
        self.risk_difference_var_vector = None
        self.risk_difference = None
        self.risk_difference_ci = None
        self.risk_difference_se = None

        self.risk_ratio_vector = None
        self.risk_ratio_var_vector = None
        self.risk_ratio = None
        self.risk_ratio_ci = None
        self.risk_ratio_se = None

        self.odds_ratio_vector = None
        self.odds_ratio_var_vector = None
        self.odds_ratio = None
        self.odds_ratio_se = None
        self.odds_ratio_ci = None

    def exposure_model(self, covariates, estimator, bound=False):
        """Specify the treatment nuisance model variables and estimator(s) to use. These parameters are held
        in the background until the .fit() function is called. These approaches are for used each sample split

        Parameters
        ----------
        covariates : str
            Confounders to include in the propensity score model. Follows patsy notation
        estimator :
            Estimator to use for prediction of the propensity score
        bound : float, list, optional
            Whether to bound predicted probabilities. Default is False, which does not bound
        """
        self._a_estimator = estimator
        self._a_covariates = covariates
        self._fit_treatment_ = True
        self._gbounds = bound

    def outcome_model(self, covariates, estimator):
        """Specify the outcome nuisance model variables and estimator(s) to use. These parameters are held
        in the background until the .fit() function is called. These approaches are for used each sample split

        Parameters
        ----------
        covariates : str
            Confounders to include in the propensity score model. Follows patsy notation
        estimator :
            Estimator to use for prediction of the propensity score
        """
        self._y_estimator = estimator
        self._y_covariates = covariates
        self._fit_outcome_ = True

    def fit(self, n_splits=3, n_partitions=100, method='median', random_state=None):
        """Runs the crossfit estimation procedure with the targeted maximum likelihood estimator. The
        estimation process is completed for multiple different splits during the procedure. The final estimate is
        defined as either the median or mean of the causal measure from each of the different splits. Median is
        used as the default since it is more stable.

        Note
        ----
        `n_partition` should be kept high to reduce dependency of results on the chosen number of splits

        Confidence intervals come from influences curves and incorporates the within-split variance and between-split
        variance.

        Parameters
        ----------
        n_splits : int
            Number of splits to use with a default of 3. The number of splits must be greater than or equal to 3.
        n_partitions : int
            Number of times to repeat the partition process. The default is 100, which I have seen good performance
            with in the past. Note that this algorithm can take a long time to run for high values of this parameter.
            It is best to test out run-times on small numbers first. Also if running in parallel, it can be reduced
        method : str, optional
            Method to obtain point estimates and standard errors. Median method takes the median (which is more robust)
            and the mean takes the mean. It has been remarked that the median is preferred, since it is more stable to
            extreme outliers, which may happen in finite samples
        random_state : None, int, optional
            Whether to set a seed for the partitions. Default is None (which does not use a user-set seed). Any valid
            NumPy seed can be input. Note that you should also state the random_state of all (applicable) estimators
            to ensure replicability. Seeds are chosen by the following procedure. The input random_state is based to
            np.random.choice to select n_partitions between 0 and 5million. That list of n_partition-length is then
            passed to each iteration of the cross-fitting pandas.DataFrame.sample(random_state).
        """
        # Checking for various issues
        if not self._fit_treatment_:
            raise ValueError("exposure_model() must be called before fit()")
        if not self._fit_outcome_:
            raise ValueError("outcome_model() must be called before fit()")
        if n_splits < 3:
            raise ValueError("DoubleCrossfitTMLE requires that n_splits >= 3")

        # Storing some information
        self._n_splits_ = n_splits
        self._n_partitions = n_partitions
        self._combine_method_ = method

        # Creating blank lists
        diff_est, diff_var, rratio_est, rratio_var, oratio_est, oratio_var = [], [], [], [], [], []

        # Conducts the re-sampling procedure
        if random_state is None:
            random_state = [None] * n_partitions
        else:
            random_state = RandomState(random_state).choice(range(5000000), size=n_partitions, replace=False)
        for j in range(self._n_partitions):
            # Estimating for a particular split (lots of functions happening in the background)
            result = self._single_crossfit_(random_state=random_state[j])

            # Appending results of this particular split combination
            diff_est.append(result[0])
            diff_var.append(result[1])
            if not self._continuous_outcome_:
                rratio_est.append(result[2])
                rratio_var.append(result[3])
                oratio_est.append(result[4])
                oratio_var.append(result[5])

        # Obtaining overall estimate and (1-alpha)% CL from all splits
        zalpha = norm.ppf(1 - self.alpha / 2, loc=0, scale=1)

        est, var = calculate_joint_estimate(diff_est, diff_var, method=method)
        if self._continuous_outcome_:
            self.ace_vector = diff_est
            self.ace_var_vector = diff_var
            self.ace = est
            self.ace_se = np.sqrt(var)
            self.ace_ci = (self.ace - zalpha*self.ace_se,
                           self.ace + zalpha*self.ace_se)
        else:
            # Risk Difference
            self.risk_difference_vector = diff_est
            self.risk_difference_var_vector = diff_var
            self.risk_difference = est
            self.risk_difference_se = np.sqrt(var)
            self.risk_difference_ci = (self.risk_difference - zalpha*self.risk_difference_se,
                                       self.risk_difference + zalpha*self.risk_difference_se)
            # Risk Ratio
            self.risk_ratio_vector = rratio_est
            self.risk_ratio_var_vector = rratio_var
            ln_rr, ln_rr_var = calculate_joint_estimate(np.log(self.risk_ratio_vector),
                                                        self.risk_ratio_var_vector, method=method)
            self.risk_ratio = np.exp(ln_rr)
            self.risk_ratio_se = np.sqrt(ln_rr_var)
            self.risk_ratio_ci = (np.exp(ln_rr - zalpha*self.risk_ratio_se),
                                  np.exp(ln_rr + zalpha*self.risk_ratio_se))
            # Odds Ratio
            self.odds_ratio_vector = oratio_est
            self.odds_ratio_var_vector = oratio_var
            ln_or, ln_or_var = calculate_joint_estimate(np.log(self.odds_ratio_vector),
                                                        self.odds_ratio_var_vector, method=method)
            self.odds_ratio = np.exp(ln_or)
            self.odds_ratio_se = np.sqrt(ln_or_var)
            self.odds_ratio_ci = (np.exp(ln_or - zalpha*self.odds_ratio_se),
                                  np.exp(ln_or + zalpha*self.odds_ratio_se))

    def summary(self, decimal=3):
        """Prints summary of model results

        Parameters
        ----------
        decimal : int, optional
            Number of decimal places to display. Default is 3
        """
        if (self._fit_outcome_ is False) or (self._fit_treatment_ is False):
            raise ValueError('exposure_model and outcome_model must be specified before the estimate can '
                             'be generated')

        print('======================================================================')
        print('                     Double Cross-fit TMLE                            ')
        print('======================================================================')
        fmt = 'Treatment:        {:<15} No. Observations:     {:<20}'
        print(fmt.format(self.exposure, self.df.shape[0]))
        fmt = 'Outcome:          {:<15} No. of Splits:        {:<20}'
        print(fmt.format(self.outcome, self._n_splits_))
        fmt = 'Method:           {:<15} No. of Partitions:    {:<20}'
        print(fmt.format(self._combine_method_, self._n_partitions))

        print('======================================================================')
        if self._continuous_outcome_:
            print('Average Causal Effect: ', round(float(self.ace), decimal))
            print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: (' +
                  str(round(self.ace_ci[0], decimal)), ',',
                  str(round(self.ace_ci[1], decimal)) + ')')
        else:
            print('Risk Difference:    ', round(float(self.risk_difference), decimal))
            print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: (' +
                  str(round(self.risk_difference_ci[0], decimal)), ',',
                  str(round(self.risk_difference_ci[1], decimal)) + ')')
            print('----------------------------------------------------------------------')
            print('Risk Ratio:         ', round(float(self.risk_ratio), decimal))
            print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: (' +
                  str(round(self.risk_ratio_ci[0], decimal)), ',',
                  str(round(self.risk_ratio_ci[1], decimal)) + ')')
            print('----------------------------------------------------------------------')
            print('Odds Ratio:         ', round(float(self.odds_ratio), decimal))
            print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: (' +
                  str(round(self.odds_ratio_ci[0], decimal)), ',',
                  str(round(self.odds_ratio_ci[1], decimal)) + ')')
        print('======================================================================')

    def run_diagnostics(self, color='gray'):
        """Runs available diagnostics for the plots. Currently diagnostics consist of a plot of the different point
        estimates and variance estimates across different partitions. Diagnostics for cross-fit estimators is ongoing.
        If you have any suggestions, please feel free to contact me on GitHub

        Parameters
        ----------
        color : str, optional
            Controls color of the plots. Default is gray

        Returns
        -------
        Plot to console
        """
        # Continuous outcomes have less plots to generate
        if self._continuous_outcome_:
            _run_diagnostic_(diff=self.ace_vector, diff_var=self.ace_var_vector,
                             color=color)

        # Binary outcomes have plots for all measures
        else:
            _run_diagnostic_(diff=self.risk_difference_vector, diff_var=self.risk_difference_var_vector,
                             rratio=self.risk_ratio_vector, rratio_var=self.risk_ratio_var_vector,
                             oratio=self.odds_ratio_vector, oratio_var=self.odds_ratio_var_vector,
                             color=color)

    def _single_crossfit_(self, random_state):
        """Background function that runs a single crossfit of the split samples
        """
        # Dividing into s different splits
        sample_split = _sample_split_(self.df, n_splits=self._n_splits_, random_state=random_state)

        # Determining pairings to use for each sample split and each combination
        pairing_exposure = [i - 1 for i in range(self._n_splits_)]
        pairing_outcome = [i - 2 for i in range(self._n_splits_)]

        # Estimating treatment nuisance model
        a_models = _treatment_nuisance_(treatment=self.exposure, estimator=self._a_estimator,
                                        samples=sample_split, covariates=self._a_covariates)
        # Estimating outcome nuisance model
        y_models = _outcome_nuisance_(outcome=self.outcome, estimator=self._y_estimator,
                                      samples=sample_split, covariates=self._y_covariates)

        # Generating predictions based on set pairs for cross-fit procedure
        predictions = []
        y_obs, a_obs = np.array([]), np.array([])
        split_index = []
        for id, ep, op in zip(range(self._n_splits_), pairing_exposure, pairing_outcome):
            predictions.append(self._generate_predictions_(sample_split[id],
                                                           a_model_v=a_models[ep],
                                                           y_model_v=y_models[op]))
            # Generating vector of Y in correct order
            y_obs = np.append(y_obs, np.asarray(sample_split[id][self.outcome]))
            # Generating vector of A in correct order
            a_obs = np.append(a_obs, np.asarray(sample_split[id][self.exposure]))
            # Generating index for splits
            split_index.extend([id]*sample_split[id].shape[0])

        # Stacking Predicted Pr(A=1), Y(a=1), Y(a=0)
        pred_a_array, pred_y1_array, pred_y0_array = np.array([]), np.array([]), np.array([])
        for preds in predictions:
            pred_a_array = np.append(pred_a_array, preds[0])
            pred_y1_array = np.append(pred_y1_array, preds[1])
            pred_y0_array = np.append(pred_y0_array, preds[2])

        # Applying bounds if requested
        if self._gbounds:  # Bounding g-model if requested
            pred_a_array = probability_bounds(pred_a_array, bounds=self._gbounds)

        # Calculating point estimates
        targeted_vals = targeting_step(y=y_obs, a=a_obs,
                                       py_a=pred_y1_array, py_n=pred_y0_array,
                                       pa1=pred_a_array, pa0=1-pred_a_array,
                                       splits=np.asarray(split_index))

        if self._continuous_outcome_:
            difference, var_diff = tmle_calculator(y=y_obs,
                                                   ystar1=targeted_vals[0], ystar0=targeted_vals[1],
                                                   ystara=targeted_vals[2],
                                                   h1w=targeted_vals[3], h0w=targeted_vals[4], haw=targeted_vals[5],
                                                   splits=np.asarray(split_index),
                                                   measure='ate',
                                                   lower_bound=self._continuous_min, upper_bound=self._continuous_max)
            return difference, var_diff
        else:
            difference, var_diff = tmle_calculator(y=y_obs,
                                                   ystar1=targeted_vals[0], ystar0=targeted_vals[1],
                                                   ystara=targeted_vals[2],
                                                   h1w=targeted_vals[3], h0w=targeted_vals[4], haw=targeted_vals[5],
                                                   splits=np.asarray(split_index),
                                                   measure='risk_difference')
            rratio, var_rratio = tmle_calculator(y=y_obs,
                                                 ystar1=targeted_vals[0], ystar0=targeted_vals[1],
                                                 ystara=targeted_vals[2],
                                                 h1w=targeted_vals[3], h0w=targeted_vals[4], haw=targeted_vals[5],
                                                 splits=np.asarray(split_index),
                                                 measure='risk_ratio')
            oratio, var_oratio = tmle_calculator(y=y_obs,
                                                 ystar1=targeted_vals[0], ystar0=targeted_vals[1],
                                                 ystara=targeted_vals[2],
                                                 h1w=targeted_vals[3], h0w=targeted_vals[4], haw=targeted_vals[5],
                                                 splits=np.asarray(split_index),
                                                 measure='odds_ratio')
            return difference, var_diff, rratio, var_rratio, oratio, var_oratio

    def _generate_predictions_(self, sample, a_model_v, y_model_v):
        """Generates predictions from fitted functions (in background of _single_crossfit()
        """
        s = sample.copy()

        # Predicting Pr(A=1|L)
        xdata = np.asarray(patsy.dmatrix(self._a_covariates + ' - 1', s))
        a_pred = _ml_predictor(xdata, fitted_algorithm=a_model_v)

        # Predicting E(Y|A=1, L)
        s[self.exposure] = 1
        xdata = np.asarray(patsy.dmatrix(self._y_covariates + ' - 1', s))
        y_treat = _ml_predictor(xdata, fitted_algorithm=y_model_v)

        # Predicting E(Y|A=0, L)
        s[self.exposure] = 0
        xdata = np.asarray(patsy.dmatrix(self._y_covariates + ' - 1', s))
        y_none = _ml_predictor(xdata, fitted_algorithm=y_model_v)

        return a_pred, y_treat, y_none


def calculate_joint_estimate(point_est, var_est, method):
    """Function that combines the different estimates across partitions into a single point estimate and standard
    error. This function can be directly called and have data directly input into. Use of this function allows for
    users to run the cross-fit procedure in parallel, output the answers, and then produce the final point estimate
    via this function.

    Note
    ----
    This function is only intended to make running the parallel cross-fit procedure easier to implement for users.
    It does not actually run any part of the cross-fitting procedure. It only calculates a summary from disparate
    partitions

    Parameters
    ----------
    point_est : container
        Container of point estimates
    var_est : container
        Container of variance estimates
    method : string
        Method to combine results. Options are 'median' or 'mean'. Median is recommended since it is more stable
        across a fewer number of partitions
    """
    if len(point_est) != len(var_est):
        raise ValueError("The length of point_est and var_est do not match")

    # Using the Median Method
    if method == 'median':
        single_point = np.median(point_est)
        single_point_var = np.median(var_est + (point_est - single_point)**2)

    # Using the Mean Method
    elif method == 'mean':
        single_point = np.mean(point_est)
        single_point_var = np.mean(var_est + (point_est - single_point)**2)

    # Error if neither exists
    else:
        raise ValueError("Either 'mean' or 'median' must be selected for the pooling of repeated sample splits")

    return single_point, single_point_var


def targeting_step(y, a, py_a, py_n, pa1, pa0, splits):
    f = sm.families.family.Binomial()
    h1w = a / pa1
    h0w = -(1 - a) / pa0
    haw = h1w + h0w
    py_o = a * py_a + (1 - a) * py_n

    ystar1, ystar0, ystara = [], [], []
    for s in set(splits):
        ys = y[splits == s]
        pa1s = pa1[splits == s]
        pa0s = pa0[splits == s]
        py_as = py_a[splits == s]
        py_ns = py_n[splits == s]
        py_os = py_o[splits == s]
        h1ws = h1w[splits == s]
        h0ws = h0w[splits == s]

        # Targeting Step
        log = sm.GLM(ys, np.column_stack((h1ws, h0ws)), offset=np.log(probability_to_odds(py_os)),
                     family=f, missing='drop').fit()
        epsilon = log.params

        # Getting updated predictions from targeting step
        ystar1 = np.append(ystar1, logistic.cdf(np.log(probability_to_odds(py_as)) + epsilon[0] / pa1s))
        ystar0 = np.append(ystar0, logistic.cdf(np.log(probability_to_odds(py_ns)) - epsilon[1] / pa0s))
        ystara = np.append(ystara, log.predict(np.column_stack((h1ws, h0ws)),
                                               offset=np.log(probability_to_odds(py_os))))
    return ystar1, ystar0, ystara, h1w, h0w, haw


def tmle_calculator(y, ystar1, ystar0, ystara, h1w, h0w, haw, splits,
                    measure='ate', lower_bound=None, upper_bound=None):
    """Function to calculate TMLE estimates for SingleCrossfitTMLE, and DoubleCrossfitTMLE
    """
    if measure in ["ate", "risk_difference"]:
        # Unbounding if continuous outcome (ate)
        if measure == "ate":
            # Unbounding continuous outcomes
            y = tmle_unit_unbound(y, mini=lower_bound, maxi=upper_bound)
            ystar1 = tmle_unit_unbound(ystar1, mini=lower_bound, maxi=upper_bound)
            ystar0 = tmle_unit_unbound(ystar0, mini=lower_bound, maxi=upper_bound)
            ystara = tmle_unit_unbound(ystara, mini=lower_bound, maxi=upper_bound)

        # Point Estimate
        estimate = np.mean(ystar1 - ystar0)
        # Variance estimate
        variance = []
        for s in set(splits):
            ys = y[splits == s]
            ystar1s = ystar1[splits == s]
            ystar0s = ystar0[splits == s]
            ystaras = ystara[splits == s]
            haws = haw[splits == s]

            ic = haws * (ys - ystaras) + (ystar1s - ystar0s) - estimate
            variance.append(np.var(ic, ddof=1))

        return estimate, (np.mean(variance) / y.shape[0])

    elif measure == 'risk_ratio':
        # Point Estimate
        estimate = np.mean(ystar1) / np.mean(ystar0)
        variance = []
        for s in set(splits):
            ys = y[splits == s]
            ystar1s = ystar1[splits == s]
            ystar0s = ystar0[splits == s]
            ystaras = ystara[splits == s]
            h1ws = h1w[splits == s]
            h0ws = h0w[splits == s]

            ic = (1/np.mean(ystar1s) * (h1ws * (ys - ystaras)) + ystar1s - np.mean(ystar1s) -
                  (1/np.mean(ystar0s) * (-1 * h0ws * (ys - ystaras)) + ystar0s - np.mean(ystar0s)))
            variance.append(np.var(ic, ddof=1))

        return estimate, (np.mean(variance) / y.shape[0])

    elif measure == 'odds_ratio':
        # Point Estimate
        estimate = (np.mean(ystar1) / (1-np.mean(ystar1))) / (np.mean(ystar0) / (1-np.mean(ystar0)))
        variance = []
        for s in set(splits):
            ys = y[splits == s]
            ystar1s = ystar1[splits == s]
            ystar0s = ystar0[splits == s]
            ystaras = ystara[splits == s]
            h1ws = h1w[splits == s]
            h0ws = h0w[splits == s]

            ic = ((1-np.mean(ystar1s))/np.mean(ystar1s)*(h1ws*(ys - ystaras) + ystar1s) -
                  (1-np.mean(ystar0s))/np.mean(ystar0s)*(-1*h0ws*(ys - ystaras) + ystar0s))
            variance.append(np.var(ic, ddof=1))

        return estimate, (np.mean(variance) / y.shape[0])

    else:
        raise ValueError("Invalid measure requested within function: tmle_calculator. Input measure is " +
                         str(measure) + " but only 'ate', 'risk_difference', 'risk_ratio', and "
                                        "'odds_ratio' are accepted.")


def _sample_split_(data, n_splits, random_state=None):
    """Background function to split data into three non-overlapping pieces
    """
    # Break into approx even splits
    n = int(data.shape[0] / n_splits)

    splits = []
    data_to_sample = data.copy()
    # Procedures is done n_splits - 1 times
    for i in range(n_splits-1):  # Loops through splits and takes random sample all remaining sets of the data
        s = data_to_sample.sample(n=n, random_state=RandomState(random_state))
        splits.append(s.copy())
        data_to_sample = data_to_sample.loc[data_to_sample.index.difference(s.index)].copy()

    # Remaining data becomes last split
    splits.append(data_to_sample)
    return splits


def _ml_predictor(xdata, fitted_algorithm):
    """Background function to generate predictions of treatments
    """
    if hasattr(fitted_algorithm, 'predict_proba'):
        return fitted_algorithm.predict_proba(xdata)[:, 1]
    elif hasattr(fitted_algorithm, 'predict'):
        return fitted_algorithm.predict(xdata)


def _treatment_nuisance_(treatment, estimator, samples, covariates):
    """Procedure to fit the treatment ML
    """
    treatment_fit_splits = []
    for s in samples:
        # Using patsy to pull out the covariates
        xdata = np.asarray(patsy.dmatrix(covariates + ' - 1', s))
        ydata = np.asarray(s[treatment])

        # Fitting machine learner / super learner to each split
        est = copy.deepcopy(estimator)
        try:
            fm = est.fit(X=xdata, y=ydata)
            # print("Treatment model")
            # print(fm.summary())
        except TypeError:
            raise TypeError("Currently custom_model must have the 'fit' function with arguments 'X', 'y'. This "
                            "covers both sklearn and supylearner")

        # Adding model to the list of models
        treatment_fit_splits.append(fm)

    return treatment_fit_splits


def _outcome_nuisance_(outcome, estimator, samples, covariates):
    """Background function to generate predictions of outcomes
    """
    outcome_fit_splits = []
    for s in samples:
        # Using patsy to pull out the covariates
        xdata = np.asarray(patsy.dmatrix(covariates + ' - 1', s))
        ydata = np.asarray(s[outcome])

        # Fitting machine learner / super learner to each
        est = copy.deepcopy(estimator)
        try:
            fm = est.fit(X=xdata, y=ydata)
            # print(est.summary())
        except TypeError:
            raise TypeError("Currently custom_model must have the 'fit' function with arguments 'X', 'y'. This "
                            "covers both sklearn and supylearner")

        # Adding model to the list of models
        outcome_fit_splits.append(fm)

    return outcome_fit_splits


def _estimate_density_plot_(estimates, bw_method='scott', fill=True, color='gray', variance=False):
    """Generates a density plot of the different estimates for each of the different sample splits. Helps to visualize
    the variability between different splits. If there is high variability, this indicates there is high sensitivity
    to the particular chosen split.

    Returns
    -------

    """
    if variance:
        x = np.linspace(0, np.max(estimates)+0.005, 10000)
    else:
        x = np.linspace(np.min(estimates)-0.02, np.max(estimates)+0.02, 10000)
    density_t = gaussian_kde(estimates, bw_method=bw_method)

    # Plot
    ax = plt.gca()
    if fill:
        ax.fill_between(x, density_t(x), color=color, alpha=0.2, label=None)
    ax.plot(x, density_t(x), color=color)
    ax.set_yticks([])
    return ax


def _run_diagnostic_(diff, diff_var, rratio=None, rratio_var=None, oratio=None, oratio_var=None, color="gray"):
    """Background function to run all diagnostics

    Returns
    -------
    Plot to console
    """
    # Continuous outcomes have less plots to generate
    if rratio is None:
        # Point estimates
        plt.subplot(121)
        _estimate_density_plot_(diff, bw_method='scott', fill=True, color=color)
        plt.title("ACE")
        # Variance estimates
        plt.subplot(122)
        _estimate_density_plot_(diff_var, bw_method='scott', fill=True, color=color)
        plt.title("Var(ACE)")

    # Binary outcomes have plots for all measures
    else:
        if oratio is None:
            # Risk Difference estimates
            plt.subplot(221)
            _estimate_density_plot_(diff, bw_method='scott', fill=True, color=color)
            plt.title("Risk Difference")
            # Var(RD) estimates
            plt.subplot(223)
            _estimate_density_plot_(diff_var, bw_method='scott',
                                    fill=True, color=color, variance=True)
            plt.title("Var(RD)")

            # Risk Ratio estimates
            plt.subplot(222)
            _estimate_density_plot_(rratio, bw_method='scott', fill=True, color=color)
            plt.title("Risk Ratio")
            # Var(RR) estimates
            plt.subplot(224)
            _estimate_density_plot_(rratio_var, bw_method='scott',
                                    fill=True, color=color, variance=True)
            plt.title("Var(ln(RR))")
        else:
            # Risk Difference estimates
            plt.subplot(231)
            _estimate_density_plot_(diff, bw_method='scott', fill=True, color=color)
            plt.title("Risk Difference")
            # Var(RD) estimates
            plt.subplot(234)
            _estimate_density_plot_(diff_var, bw_method='scott',
                                    fill=True, color=color, variance=True)
            plt.title("Var(RD)")

            # Risk Ratio estimates
            plt.subplot(232)
            _estimate_density_plot_(rratio, bw_method='scott', fill=True, color=color)
            plt.title("Risk Ratio")
            # Var(RR) estimates
            plt.subplot(235)
            _estimate_density_plot_(rratio_var, bw_method='scott',
                                    fill=True, color=color, variance=True)
            plt.title("Var(ln(RR))")
            # Odds Ratio estimates
            plt.subplot(233)
            _estimate_density_plot_(oratio, bw_method='scott', fill=True, color=color)
            plt.title("Odds Ratio")
            # Var(OR) estimates
            plt.subplot(236)
            _estimate_density_plot_(oratio_var, bw_method='scott',
                                    fill=True, color=color, variance=True)
            plt.title("Var(ln(OR))")

    plt.tight_layout()
    plt.show()
