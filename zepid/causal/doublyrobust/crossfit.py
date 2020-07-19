import copy
import warnings
import patsy
import numpy as np
import pandas as pd
from scipy.stats import logistic, norm
import statsmodels.api as sm
import statsmodels.formula.api as smf

from zepid.causal.utils import _bounding_


class SingleCrossfitAIPTW:
    """Implementation of the augmented inverse probability weighted estimator with a cross-fit procedure. The purpose
    of the cross-fit procedure is to all for non-Donsker nuisance function estimators. Some of machine learning
    algorithms are non-Donsker. In practice this means that confidence interval coverage can be incorrect when certain
    nuisance function estimators are used. Additionally, bias may persist as well.

    SingleCrossfitAIPTW allows for both single cross-fit, where the data set is paritioned into at least two
    non-overlapping splits. The nuisance function estimators are then estimated in each split. The estimated nuisance
    functions are then used to predict values in the opposing split. This decouple the nuisance function estimation
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

    >>> scf_aipw = SingleCrossfitAIPTW(df, exposure='art', outcome='dead')
    >>> scf_aipw.exposure_model("male + age0 + cd40 + dvl0", estimator=LogisticRegression(solver='lbfgs'))
    >>> scf_aipw.outcome_model("art + male + age0 + cd40 + dvl0", estimator=LogisticRegression(solver='lbfgs'))
    >>> scf_aipw.fit(n_splits=5, n_partitions=100)
    >>> scf_aipw.summary()

    References
    ----------
    Chernozhukov V, Chetverikov D, Demirer M, Duflo E, Hansen C, Newey W, & Robins J. (2018). "Double/debiased machine
    learning for treatment and structural parameters". The Econometrics Journal 21:1; pC1–C6
    """
    def __init__(self, df, exposure, outcome, alpha=0.05):
        if df.dropna().shape[0] != df.shape[0]:
            warnings.warn("There is missing data in the dataset. By default, SingleCrossfitAIPTW will drop all missing "
                          "data. SingleCrossfitAIPTW will fit " + str(df.dropna().shape[0]) + ' of ' +
                          str(df.shape[0]) + ' observations', UserWarning)
        self.df = df.copy().dropna().reset_index()
        self.exposure = exposure
        self.outcome = outcome
        self.alpha = alpha

        # Checking whether binary or continuous outcome
        if self.df[self.outcome].value_counts().index.isin([0, 1]).all():
            self._continuous_outcome_ = False
        else:
            self._continuous_outcome_ = True

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

        self.ace = None
        self.ace_ci = None
        self.ace_se = None

        self.risk_difference = None
        self.risk_difference_ci = None
        self.risk_difference_se = None

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

    def fit(self, n_splits=2, n_partitions=100, method='median'):
        """Runs the crossfit estimation procedure with augmented inverse probability weighted estimator. The
        estimation process is completed for multiple different splits during the procedure. The final estimate is
        defined as either the median or mean of the average causal effect from each of the different splits. Median is
        used as the default since it is more stable.

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
        for j in range(self._n_partitions):
            # Estimating for a particular split (lots of functions happening in the background)
            result = self._single_crossfit_()

            # Appending results of this particular split combination
            # TODO Risk Ratios are not currently calculated. They would need to be added throughout
            diff_est.append(result[0])
            diff_var.append(result[1])

        # Obtaining overall estimate and (1-alpha)% CL from all splits
        zalpha = norm.ppf(1 - self.alpha / 2, loc=0, scale=1)

        if self._continuous_outcome_:
            self.ace, self.ace_se = calculate_joint_estimate(diff_est, diff_var, method=method)
            self.ace_ci = (self.ace - zalpha*self.ace_se,
                                       self.ace + zalpha*self.ace_se)
        else:
            # Risk Difference
            self.risk_difference, self.risk_difference_se = calculate_joint_estimate(diff_est, diff_var, method=method)
            self.risk_difference_ci = (self.risk_difference - zalpha*self.risk_difference_se,
                                       self.risk_difference + zalpha*self.risk_difference_se)
            # TODO Risk Ratio
            # self.calculate_estimate(point_est, var_est, method=method)
            # self.risk_ratio_ci = (self.risk_ratio - zalpha*self.risk_ratio_se,
            #                           self.risk_ratio + zalpha*self.risk_ratio_se)

    def summary(self, decimal=3):
        """Prints summary of model results

        Parameters
        ----------
        decimal : int, optional
            Number of decimal places to display. Default is 3
        """
        if (self._fit_outcome_ is False) or (self._fit_treatment_ is False):
            raise ValueError('The treatment and outcome models must be specified before the double robust estimate can '
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
            print('Average Treatment Effect: ', round(float(self.ace), decimal))
            print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: (' +
                  str(round(self.ace_ci[0], decimal)), ',',
                  str(round(self.ace_ci[1], decimal)) + ')')
        else:
            print('Risk Difference:    ', round(float(self.risk_difference), decimal))
            print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: (' +
                  str(round(self.risk_difference_ci[0], decimal)), ',',
                  str(round(self.risk_difference_ci[1], decimal)) + ')')
            # print('----------------------------------------------------------------------')
            # print('Risk Ratio:         ', round(float(self.risk_ratio), decimal))
            # print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: (' +
            #      str(round(self.risk_ratio_ci[0], decimal)), ',',
            #      str(round(self.risk_ratio_ci[1], decimal)) + ')')

        print('======================================================================')

    def _single_crossfit_(self):
        """Background function that runs a single crossfit of the split samples
        """
        # Dividing into s different splits
        sample_split = _sample_split_(self.df, n_splits=self._n_splits_)

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
            pred_a_array = _bounding_(pred_a_array, bounds=self._gbounds)

        # Calculating point estimates
        riskdifference, var_rd = _aipw_calculator_(y=y_obs, a=a_obs,
                                                   py_a=pred_y1_array, py_n=pred_y0_array,
                                                   pa=pred_a_array, splits=np.asarray(split_index))
        return riskdifference, var_rd

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
    nuisance function estimators are used.

    DoubleCrossfitAIPTW allows for double-crossfit, where the data set is partitioned into at least three
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

    >>> scf_aipw = DoubleCrossfitAIPTW(df, exposure='art', outcome='dead')
    >>> scf_aipw.exposure_model("male + age0 + cd40 + dvl0", estimator=LogisticRegression(solver='lbfgs'))
    >>> scf_aipw.outcome_model("art + male + age0 + cd40 + dvl0", estimator=LogisticRegression(solver='lbfgs'))
    >>> scf_aipw.fit(n_splits=5, n_partitions=100)
    >>> scf_aipw.summary()

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
        if df.dropna().shape[0] != df.shape[0]:
            warnings.warn("There is missing data in the dataset. By default, DoubleCrossfitAIPTW will drop all missing "
                          "data. DoubleCrossfitAIPTW will fit " + str(df.dropna().shape[0]) + ' of ' +
                          str(df.shape[0]) + ' observations', UserWarning)
        self.df = df.copy().dropna().reset_index()
        self.exposure = exposure
        self.outcome = outcome
        self.alpha = alpha

        # Checking whether binary or continuous outcome
        if self.df[self.outcome].value_counts().index.isin([0, 1]).all():
            self._continuous_outcome_ = False
        else:
            self._continuous_outcome_ = True

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

        self.ace = None
        self.ace_ci = None
        self.ace_se = None

        self.risk_difference = None
        self.risk_difference_ci = None
        self.risk_difference_se = None

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

    def fit(self, n_splits=3, n_partitions=100, method='median'):
        """Runs the crossfit estimation procedure with augmented inverse probability weighted estimator. The
        estimation process is completed for multiple different splits during the procedure. The final estimate is
        defined as either the median or mean of the average causal effect from each of the different splits. Median is
        used as the default since it is more stable.

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
        """
        # Checking for various issues
        if not self._fit_treatment_:
            raise ValueError("exposure_model() must be called before fit()")
        if not self._fit_outcome_:
            raise ValueError("outcome_model() must be called before fit()")
        if n_splits < 3:
            raise ValueError("DoubleCrossfitAIPTW requires that n_splits > 2")

        # Storing some information
        self._n_splits_ = n_splits
        self._n_partitions = n_partitions
        self._combine_method_ = method

        # Creating blank lists
        diff_est, diff_var, ratio_est, ratio_var = [], [], [], []

        # Conducts the re-sampling procedure
        for j in range(self._n_partitions):
            # Estimating for a particular split (lots of functions happening in the background)
            result = self._single_crossfit_()

            # Appending results of this particular split combination
            # TODO Risk Ratios are not currently calculated. They would need to be added throughout
            diff_est.append(result[0])
            diff_var.append(result[1])

        # Obtaining overall estimate and (1-alpha)% CL from all splits
        zalpha = norm.ppf(1 - self.alpha / 2, loc=0, scale=1)

        if self._continuous_outcome_:
            self.ace, self.ace_se = calculate_joint_estimate(diff_est, diff_var, method=method)
            self.ace_ci = (self.ace - zalpha*self.ace_se,
                           self.ace + zalpha*self.ace_se)
        else:
            # Risk Difference
            self.risk_difference, self.risk_difference_se = calculate_joint_estimate(diff_est, diff_var, method=method)
            self.risk_difference_ci = (self.risk_difference - zalpha*self.risk_difference_se,
                                       self.risk_difference + zalpha*self.risk_difference_se)
            # TODO Risk Ratio
            # self.calculate_estimate(point_est, var_est, method=method)
            # self.risk_ratio_ci = (self.risk_ratio - zalpha*self.risk_ratio_se,
            #                           self.risk_ratio + zalpha*self.risk_ratio_se)

    def summary(self, decimal=3):
        """Prints summary of model results

        Parameters
        ----------
        decimal : int, optional
            Number of decimal places to display. Default is 3
        """
        if (self._fit_outcome_ is False) or (self._fit_treatment_ is False):
            raise ValueError('The treatment and outcome models must be specified before the double robust estimate can '
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
            print('Average Treatment Effect: ', round(float(self.ace), decimal))
            print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: (' +
                  str(round(self.ace_ci[0], decimal)), ',',
                  str(round(self.ace_ci[1], decimal)) + ')')
        else:
            print('Risk Difference:    ', round(float(self.risk_difference), decimal))
            print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: (' +
                  str(round(self.risk_difference_ci[0], decimal)), ',',
                  str(round(self.risk_difference_ci[1], decimal)) + ')')
            # print('----------------------------------------------------------------------')
            # print('Risk Ratio:         ', round(float(self.risk_ratio), decimal))
            # print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: (' +
            #      str(round(self.risk_ratio_ci[0], decimal)), ',',
            #      str(round(self.risk_ratio_ci[1], decimal)) + ')')

        print('======================================================================')

    def _single_crossfit_(self):
        """Background function that runs a single crossfit of the split samples
        """
        # Dividing into s different splits
        sample_split = _sample_split_(self.df, n_splits=self._n_splits_)

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
            pred_a_array = _bounding_(pred_a_array, bounds=self._gbounds)

        # Calculating point estimates
        riskdifference, var_rd = _aipw_calculator_(y=y_obs, a=a_obs,
                                                   py_a=pred_y1_array, py_n=pred_y0_array,
                                                   pa=pred_a_array, splits=np.asarray(split_index))
        return riskdifference, var_rd

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
        single_point_se = np.sqrt(np.median(var_est + (point_est - single_point)**2))

    # Using the Mean Method
    elif method == 'mean':
        single_point = np.mean(point_est)
        single_point_se = np.sqrt(np.mean(var_est + (point_est - single_point)**2))

    # Error if neither exists
    else:
        raise ValueError("Either 'mean' or 'median' must be selected for the pooling of repeated sample splits")

    return single_point, single_point_se


def _sample_split_(data, n_splits):
    """Background function to split data into three non-overlapping pieces
    """
    # Break into approx even splits
    n = int(data.shape[0] / n_splits)

    splits = []
    data_to_sample = data.copy()
    # Procedures is done n_splits - 1 times
    for i in range(n_splits-1):  # Loops through splits and takes random sample all remaining sets of the data
        s = data_to_sample.sample(n=n)
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
            # print(fm.summarize())
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
            # print("Outcome model")
            # print(fm.summarize())
        except TypeError:
            raise TypeError("Currently custom_model must have the 'fit' function with arguments 'X', 'y'. This "
                            "covers both sklearn and supylearner")

        # Adding model to the list of models
        outcome_fit_splits.append(fm)

    return outcome_fit_splits


def _aipw_calculator_(y, a, py_a, py_n, pa, splits):
    """Background calculator for AIPW and AIPW standard error
    """
    y1 = np.where(a == 1, y/pa - py_a*((1 - pa) / pa), py_a)
    y0 = np.where(a == 0, y/(1 - pa) - py_n*(pa / (1 - pa)), py_n)
    rd = np.mean(y1 - y0)
    # Variance calculations
    var_rd = []
    for i in set(splits):
        y1s = y1[splits == i]
        y0s = y0[splits == i]
        var_rd.append(np.var((y1s - y0s) - rd, ddof=1))

    return rd, (np.mean(var_rd) / y.shape[0])
