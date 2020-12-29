import copy
import warnings
import patsy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import DescrStatsW
from scipy.stats import norm

from zepid.calc import probability_bounds
from zepid.causal.utils import (check_input_data, propensity_score, plot_kde, plot_love, iptw_calculator,
                                standardized_mean_differences, positivity,
                                plot_kde_accuracy, outcome_accuracy, aipw_calculator,
                                exposure_machine_learner, outcome_machine_learner, missing_machine_learner)


class AIPTW:
    r"""Augmented inverse probability of treatment weight estimator. This implementation calculates AIPTW for a
    time-fixed exposure and a single time-point outcome. `AIPTW` supports correcting for informative censoring (missing
    outcome data) through inverse probability of censoring/missingness weights.

    AIPTW is a doubly robust estimator, with a desirable property. Both of the the g-formula and IPTW require that
    our parametric regression models are correctly specified. Instead, AIPTW allows us to have two 'chances' at getting
    the model correct. If either our outcome-model or treatment-model is correctly specified, then our estimate
    will be unbiased. This property does not hold for the variance (i.e. the variance will not be doubly robust)

    The augment-inverse probability weight estimator is calculated from the following formula

    .. math::

        \widehat{DR}(a) = \frac{YA}{\widehat{\Pr}(A=a|L)} - \frac{\hat{Y}^a*(A-\widehat{\Pr}(A=a|L)}{
        \widehat{\Pr}(A=a|L)}

    The risk difference and risk ratio are calculated using the following formulas, respectively

    .. math::

        \widehat{RD} = \widehat{DR}(a=1) - \widehat{DR}(a=0)

    .. math::

        \widehat{RR} = \frac{\widehat{DR}(a=1)}{\widehat{DR}(a=0)}

    Confidence intervals for the risk difference come from the influence curve. Confidence intervals for the risk ratio
    are less straight-forward. To get confidence intervals for the risk ratio, a bootstrap procedure should be used.

    Parameters
    ----------
    df : DataFrame
        Pandas DataFrame object containing all variables of interest
    exposure : str
        Column name of the exposure variable. Currently only binary is supported
    outcome : str
        Column name of the outcome variable. Currently only binary is supported
    weights : str, optional
        Column name of weights. Weights allow for items like sampling weights to be used to estimate effects
    alpha : float, optional
        Alpha for confidence interval level. Default is 0.05, returning the 95% CL

    Examples
    --------
    Set up the environment and the data set

    >>> from zepid import load_sample_data, spline
    >>> from zepid.causal.doublyrobust import AIPTW
    >>> df = load_sample_data(timevary=False).drop(columns=['cd4_wk45'])
    >>> df[['cd4_rs1','cd4_rs2']] = spline(df,'cd40',n_knots=3,term=2,restricted=True)
    >>> df[['age_rs1','age_rs2']] = spline(df,'age0',n_knots=3,term=2,restricted=True)

    Estimate the base AIPTW model

    >>> aipw = AIPTW(df, exposure='art', outcome='dead')
    >>> aipw.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>> aipw.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>> aipw.fit()
    >>> aipw.summary()

    Estimate AIPTW accounting for missing outcome data

    >>> aipw = AIPTW(df, exposure='art', outcome='dead')
    >>> aipw.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>> aipw.missing_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>> aipw.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>> aipw.fit()
    >>> aipw.summary()

    AIPTW for continuous outcomes

    >>> df = load_sample_data(timevary=False).drop(columns=['dead'])
    >>> df[['cd4_rs1','cd4_rs2']] = spline(df,'cd40',n_knots=3,term=2,restricted=True)
    >>> df[['age_rs1','age_rs2']] = spline(df,'age0',n_knots=3,term=2,restricted=True)

    >>> aipw = AIPTW(df, exposure='art', outcome='cd4_wk45')
    >>> aipw.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>> aipw.missing_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>> aipw.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>> aipw.fit()
    >>> aipw.summary()

    >>> aipw = AIPTW(df, exposure='art', outcome='cd4_wk45')
    >>> ymodel = 'art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0'
    >>> aipw.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>> aipw.missing_model(ymodel)
    >>> aipw.outcome_model(ymodel, continuous_distribution='poisson')
    >>> aipw.fit()
    >>> aipw.summary()

    References
    ----------
    Funk MJ, Westreich D, Wiesen C, Stürmer T, Brookhart MA, & Davidian M. (2011). Doubly robust
    estimation of causal effects. American Journal of Epidemiology, 173(7), 761-767.

    Lunceford JK, Davidian M. (2004). Stratification and weighting via the propensity score in estimation of causal
    treatment effects: a comparative study. Statistics in medicine, 23(19), 2937-2960.
    """

    def __init__(self, df, exposure, outcome, weights=None, alpha=0.05):
        self.exposure = exposure
        self.outcome = outcome
        self._missing_indicator = '__missing_indicator__'
        self.df, self._miss_flag, self._continuous_outcome = check_input_data(data=df,
                                                                              exposure=exposure,
                                                                              outcome=outcome,
                                                                              estimator="AIPTW",
                                                                              drop_censoring=False,
                                                                              drop_missing=True,
                                                                              binary_exposure_only=True)

        self._weight_ = weights
        self.alpha = alpha

        self.risk_difference = None
        self.risk_ratio = None
        self.risk_difference_ci = None
        self.risk_ratio_ci = None
        self.risk_difference_se = None
        self.risk_ratio_se = None

        self.average_treatment_effect = None
        self.average_treatment_effect_ci = None
        self.average_treatment_effect_se = None

        self._continuous_type = None
        self._fit_exposure_ = False
        self._exp_model_custom = False
        self._fit_outcome_ = False
        self._out_model_custom = False
        self._fit_missing_ = False
        self._miss_model_custom = False
        self._exp_model = None
        self._out_model = None
        self._predicted_y_ = None

    def exposure_model(self, model, custom_model=None, bound=False, print_results=True):
        r"""Specify the propensity score / inverse probability weight model. Model used to predict the exposure via a
        logistic regression model. This model estimates

        .. math::

            \widehat{\Pr}(A=1|L) = logit^{-1}(\widehat{\beta_0} + \widehat{\beta} L)

        Parameters
        ----------
        model : str
            Independent variables to predict the exposure. For example, 'var1 + var2 + var3'
        custom_model : optional
            Input for a custom model that is used in place of the logit model (default). The model must have the
            "fit()" and  "predict()" attributes. SciKit-Learn style models supported as custom models. In the
            background, AIPTW will fit the custom model and generate the predicted probablities
        bound : float, list, optional
            Value between 0,1 to truncate predicted probabilities. Helps to avoid near positivity violations.
            Specifying this argument can improve finite sample performance for random positivity violations. However,
            truncating weights leads to additional confounding. Default is False, meaning no truncation of
            predicted probabilities occurs. Providing a single float assumes symmetric trunctation, where values below
            or above the threshold are set to the threshold value. Alternatively a list of floats can be provided for
            asymmetric trunctation, with the first value being the lower bound and the second being the upper bound
        print_results : bool, optional
            Whether to print the fitted model results. Default is True (prints results)
        """
        self.__mweight = model
        self._exp_model = self.exposure + ' ~ ' + model

        if custom_model is None:
            d, n, iptw = iptw_calculator(df=self.df, treatment=self.exposure, model_denom=model, model_numer='1',
                                         weight=self._weight_, stabilized=False, standardize='population',
                                         bound=None, print_results=print_results)
        else:
            self._exp_model_custom = True
            data = patsy.dmatrix(model + ' - 1', self.df)
            d = exposure_machine_learner(xdata=np.asarray(data),
                                         ydata=np.asarray(self.df[self.exposure]),
                                         ml_model=copy.deepcopy(custom_model),
                                         print_results=print_results)

        g1w = d
        g0w = 1 - d

        # Applying bounds AFTER extracting g1 and g0
        if bound:
            g1w = probability_bounds(g1w, bounds=bound)
            g0w = probability_bounds(g0w, bounds=bound)

        self.df['_g1_'] = g1w
        self.df['_g0_'] = g0w
        self._fit_exposure_ = True

    def missing_model(self, model, custom_model=None, bound=False, print_results=True):
        r"""Estimation of Pr(M=0|A,L), which is the missing data mechanism for the outcome. Predicted probabilities are
        used to create inverse probability of censoring weights to account for informative missing data on the outcome.

        Missing weights take the following form

        .. math::
            \frac{1}{\Pr(C=0|A=a, L)}

        Weights are calculated for both A=1 and A=0

        Note
        ----
        The treatment variable should be included in the model

        Parameters
        ----------
        model : str
            Independent variables to predict the exposure. Example) 'var1 + var2 + var3'. The treatment must be
            included for the missing data model
        custom_model : optional
            Input for a custom model that is used in place of the logit model (default). The model must have the
            "fit()" and  "predict()" attributes. SciKit-Learn style models are supported as custom models. In the
            background, AIPTW will fit the custom model and generate the predicted probablities
        bound : float, list, optional
            Value between 0,1 to truncate predicted probabilities. Helps to avoid near positivity violations.
            Specifying this argument can improve finite sample performance for random positivity violations. However,
            truncating weights leads to additional confounding. Default is False, meaning no truncation of
            predicted probabilities occurs. Providing a single float assumes symmetric trunctation, where values below
            or above the threshold are set to the threshold value. Alternatively a list of floats can be provided for
            asymmetric trunctation, with the first value being the lower bound and the second being the upper bound
        print_results : bool, optional
            Whether to print the fitted model results. Default is True (prints results)
        """
        # Error if no missing outcome data
        if not self._miss_flag:
            raise ValueError("No missing outcome data is present in the data set")

        # Warning if exposure is not included in the missingness of outcome model
        if self.exposure not in model:
            warnings.warn("For the specified missing outcome model, the exposure variable should be included in the "
                          "model", UserWarning)

        # Warning if exposure is not included in the missingness of outcome model
        if self.exposure not in model:
            warnings.warn("For the specified missing outcome model, the exposure variable should be included in the "
                          "model", UserWarning)

        self._miss_model = self._missing_indicator + ' ~ ' + model
        fitmodel = propensity_score(self.df, self._miss_model, print_results=print_results)

        if custom_model is None:  # Logistic Regression model for predictions
            dfx = self.df.copy()
            dfx[self.exposure] = 1
            m1w = np.where(self.df[self._missing_indicator] == 1, fitmodel.predict(dfx), np.nan)
            dfx = self.df.copy()
            dfx[self.exposure] = 0
            m0w = np.where(self.df[self._missing_indicator] == 1, fitmodel.predict(dfx), np.nan)
        else:  # User-Specified model
            self._miss_model_custom = True
            data = patsy.dmatrix(model + ' - 1', self.df)
            dfx = self.df.copy()
            dfx[self.exposure] = 1
            adata = patsy.dmatrix(model + ' - 1', dfx)
            dfx = self.df.copy()
            dfx[self.exposure] = 0
            ndata = patsy.dmatrix(model + ' - 1', dfx)

            m1w, m0w = missing_machine_learner(xdata=np.array(data),
                                               mdata=self.df[self._missing_indicator],
                                               all_a=adata, none_a=ndata,
                                               ml_model=copy.deepcopy(custom_model),
                                               print_results=print_results)

        # If bounds are requested
        if bound:
            m1w = probability_bounds(m1w, bounds=bound)
            m0w = probability_bounds(m0w, bounds=bound)

        self.df['_ipmw_a1_'] = m1w
        self.df['_ipmw_a0_'] = m0w

        self._fit_missing_ = True

    def outcome_model(self, model, custom_model=None, continuous_distribution='gaussian', print_results=True):
        r"""Specify the outcome model. Model used to predict the outcome via a regression model. For binary outcome
        data, a logistic regression model is used. For continuous outcomes, either linear or Poisson regression are
        available.

        .. math::

            \widehat{\Pr}(Y|A,L) = logit^{-1}(\widehat{\beta_0} + \widehat{\beta_1} A + \widehat{\beta} L)

        Parameters
        ----------
        model : str
            Independent variables to predict the outcome. For example, 'var1 + var2 + var3 + var4'
        custom_model : optional
            Input for a custom model that is used in place of the logit model (default). The model must have the
            "fit()" and  "predict()" attributes. SciKit-Learn style models are supported as custom models. In the
            background, TMLE will fit the custom model and generate the predicted values
        continuous_distribution : str, optional
            Distribution to use for continuous outcomes. Options are 'gaussian' for normal distributions and 'poisson'
            for Poisson distributions
        print_results : bool, optional
            Whether to print the fitted model results. Default is True (prints results)
        """
        if self.exposure not in model:
            warnings.warn("It looks like '" + self.exposure + "' is not included in the outcome model.")

        if self._miss_flag:
            cc = self.df.copy().dropna()
        else:
            cc = self.df.copy()

        self._out_model = self.outcome + ' ~ ' + model

        if custom_model is None:
            if self._continuous_outcome:
                self._continuous_type = continuous_distribution
                if (continuous_distribution == 'gaussian') or (continuous_distribution == 'normal'):
                    f = sm.families.family.Gaussian()
                elif continuous_distribution == 'poisson':
                    f = sm.families.family.Poisson()
                else:
                    raise ValueError("Only 'gaussian' and 'poisson' distributions are supported")
            else:
                f = sm.families.family.Binomial()

            if self._weight_ is None:
                log = smf.glm(self._out_model, cc, family=f).fit()
            else:
                log = smf.glm(self._out_model, cc, freq_weights=cc[self._weight_], family=f).fit()

            if print_results:
                print('==============================================================================')
                print('Outcome Model')
                print(log.summary())
                print('==============================================================================')

            # Predicting under treatment strategies
            dfx = self.df.copy()
            dfx[self.exposure] = 1
            qa1w = log.predict(dfx)
            dfx = self.df.copy()
            dfx[self.exposure] = 0
            qa0w = log.predict(dfx)

        else:
            self._out_model_custom = True
            data = patsy.dmatrix(model + ' - 1', cc)

            dfx = self.df.copy()
            dfx[self.exposure] = 1
            adata = patsy.dmatrix(model + ' - 1', dfx)
            dfx = self.df.copy()
            dfx[self.exposure] = 0
            ndata = patsy.dmatrix(model + ' - 1', dfx)

            qa1w, qa0w = outcome_machine_learner(xdata=np.asarray(data),
                                                 ydata=np.asarray(cc[self.outcome]),
                                                 all_a=adata, none_a=ndata,
                                                 ml_model=copy.deepcopy(custom_model),
                                                 continuous=self._continuous_outcome,
                                                 print_results=print_results)

        self.df['_pY1_'] = qa1w
        self.df['_pY0_'] = qa0w
        self._predicted_y_ = qa1w * self.df[self.exposure] + qa0w * (1 - self.df[self.exposure])
        self._fit_outcome_ = True

    def fit(self):
        """Calculate the augmented inverse probability weights and effect measures from the predicted exposure
        probabilities and predicted outcome values.

        Note
        ----
        Exposure and outcome models must be specified prior to `fit()`

        Returns
        -------
        For binary outcomes, gains `risk_difference`, `risk_difference_ci`, and `risk_ratio` attributes. For continuous
        outcomes, gains `average_treatment_effect` and `average_treatment_effect_ci` attributes
        """
        if (self._fit_exposure_ is False) or (self._fit_outcome_ is False):
            raise ValueError('The exposure and outcome models must be specified before the doubly robust estimate can '
                             'be generated')

        if self._miss_flag and not self._fit_missing_:
            warnings.warn("All missing outcome data is assumed to be missing completely at random. To relax this "
                          "assumption to outcome data is missing at random please use the `missing_model()` "
                          "function", UserWarning)

        # Doubly robust estimator under all treated
        a_obs = np.asarray(self.df[self.exposure])
        y_obs = np.asarray(self.df[self.outcome])
        py_a1 = np.asarray(self.df['_pY1_'])
        py_a0 = np.asarray(self.df['_pY0_'])

        if self._fit_missing_:
            ps_g1 = np.asarray(self.df['_g1_'] * self.df['_ipmw_a1_'])
            ps_g0 = np.asarray(self.df['_g0_'] * self.df['_ipmw_a0_'])
        else:
            ps_g1 = np.asarray(self.df['_g1_'])
            ps_g0 = np.asarray(self.df['_g0_'])

        if self._weight_ is None:
            w = None
        else:
            w = self.df[self._weight_]
        diff_est, diff_var = aipw_calculator(y=y_obs, a=a_obs,
                                             py_a=py_a1, py_n=py_a0,
                                             pa1=ps_g1, pa0=ps_g0,
                                             difference=True, weights=w,
                                             splits=None, continuous=self._continuous_outcome)

        # Generating estimates for the risk difference and risk ratio
        zalpha = norm.ppf(1 - self.alpha / 2, loc=0, scale=1)

        if self._continuous_outcome:
            self.average_treatment_effect = diff_est
            self.average_treatment_effect_se = np.sqrt(diff_var)
            self.average_treatment_effect_ci = [self.average_treatment_effect - zalpha * np.sqrt(diff_var),
                                                self.average_treatment_effect + zalpha * np.sqrt(diff_var)]

        else:
            self.risk_difference = diff_est
            self.risk_difference_se = np.sqrt(diff_var)
            self.risk_difference_ci = [self.risk_difference - zalpha * np.sqrt(diff_var),
                                       self.risk_difference + zalpha * np.sqrt(diff_var)]

            rr, ln_rr_var = aipw_calculator(y=y_obs, a=a_obs,
                                            py_a=py_a1, py_n=py_a0,
                                            pa1=ps_g1, pa0=ps_g0,
                                            difference=False, weights=w,
                                            splits=None, continuous=False)
            self.risk_ratio = rr
            self.risk_ratio_se = np.sqrt(ln_rr_var)
            self.risk_ratio_ci = (np.exp(np.log(rr) - zalpha * self.risk_ratio_se),
                                  np.exp(np.log(rr) + zalpha * self.risk_ratio_se))

    def summary(self, decimal=3):
        """Prints a summary of the results for the doubly robust estimator. Confidence intervals are only available for
        the risk difference currently. For risk ratio confidence intervals, the user will need to manually conduct a
        boostrapping procedure.

        Parameters
        ----------
        decimal : int, optional
            Number of decimal places to display in the result
        """
        if (self._fit_exposure_ is False) or (self._fit_exposure_ is False):
            raise ValueError('The exposure and outcome models must be specified before the double robust estimate can '
                             'be generated')

        print('======================================================================')
        print('          Augmented Inverse Probability of Treatment Weights          ')
        print('======================================================================')
        fmt = 'Treatment:        {:<15} No. Observations:     {:<20}'
        print(fmt.format(self.exposure, self.df.shape[0]))

        fmt = 'Outcome:          {:<15} No. Missing Outcome:  {:<20}'
        print(fmt.format(self.outcome, np.sum(self.df[self.outcome].isnull())))

        fmt = 'g-Model:          {:<15} Missing Model:        {:<20}'
        if self._exp_model_custom:
            e = 'User-specified'
        else:
            e = 'Logistic'
        if self._miss_model_custom and self._fit_missing_:
            m = 'User-specified'
        elif self._fit_missing_:
            m = 'Logistic'
        else:
            m = 'None'

        print(fmt.format(e, m))

        fmt = 'Q-Model:          {:<15}'
        if self._out_model_custom:
            y = 'User-specified'
        elif self._continuous_outcome:
            y = self._continuous_type
        else:
            y = 'Logistic'

        print(fmt.format(y))

        print('======================================================================')

        if self._continuous_outcome:
            print('Average Treatment Effect:   ', round(float(self.average_treatment_effect), decimal))
            if self._weight_ is None:
                print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: (' +
                      str(round(self.average_treatment_effect_ci[0], decimal)), ',',
                      str(round(self.average_treatment_effect_ci[1], decimal)) + ')')
            else:
                print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: -')
        else:
            print('Risk Difference:    ', round(float(self.risk_difference), decimal))
            if self._weight_ is None:
                print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: (' +
                      str(round(self.risk_difference_ci[0], decimal)), ',',
                      str(round(self.risk_difference_ci[1], decimal)) + ')')
            else:
                print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: -')
            print('----------------------------------------------------------------------')
            print('Risk Ratio:        ', round(float(self.risk_ratio), decimal))
            if self._weight_ is None:
                print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: (' +
                      str(round(self.risk_ratio_ci[0], decimal)), ',',
                      str(round(self.risk_ratio_ci[1], decimal)) + ')')
            else:
                print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: -')

        print('======================================================================')

    def run_diagnostics(self, decimal=3):
        """Run all currently implemented diagnostics for the exposure and outcome models. Each
        `run_diagnostics` provides results for all implemented diagnostics for ease of the user. For publication
        quality presentations, I recommend calling each diagnostic function individually and utilizing the optional
        parameters

        Note
        ----
        The plot presented cannot be edited. To edit the plots, call `plot_kde` or `plot_love` directly. Those
        functions return an axes object

        Parameters
        ----------
        decimal : int, optional
            Number of decimal places to display. Default is 3

        Returns
        -------
        None
        """
        if not self._fit_outcome_ or not self._fit_exposure_:
            raise ValueError("The exposure_model and outcome_model function must be ran before any diagnostics")

        # Weight diagnostics
        print('\tExposure Model Diagnostics')
        self.positivity(decimal=decimal)

        print('\n======================================================================')
        print('                  Standardized Mean Differences')
        print('======================================================================')
        print(self.standardized_mean_differences().set_index(keys='labels'))
        print('======================================================================\n')

        # Outcome accuracy diagnostics
        print('\tOutcome Model Diagnostics')
        v = self._predicted_y_ - self.df[self.outcome]
        outcome_accuracy(true=self.df[self.outcome], predicted=self._predicted_y_, decimal=decimal)

        df = self.df.copy()
        df['_ipw_'] = np.where(df[self.exposure] == 1, 1 / df['_g1_'], 1 / (1 - df['_g1_']))

        plt.figure(figsize=[8, 6])
        plt.subplot(221)
        plot_love(df=df, treatment=self.exposure, weight='_ipw_', formula=self.__mweight)
        plt.title("Love Plot")

        plt.subplot(223)
        plot_kde(df=df, treatment=self.exposure, probability='_g1_')
        plt.title("Kernel Density of Propensity Scores")

        plt.subplot(222)
        plot_kde_accuracy(values=v.dropna(), color='green')
        plt.title("Kernel Density of Accuracy")
        plt.tight_layout()
        plt.show()

    def positivity(self, decimal=3):
        """Use this to assess whether positivity is a valid assumption for the exposure model / calculated IPTW. If
        there are extreme outliers, this may indicate problems with the calculated weights. To reduce extreme weights,
        the `bound` argument can be specified in `exposure_model()`

        Parameters
        ----------
        decimal : int, optional
            Number of decimal places to display. Default is three

        Returns
        -------
        None
            Prints the positivity results to the console but does not return any objects
        """
        df = self.df.copy()
        df['_ipw_'] = np.where(df[self.exposure] == 1, 1 / df['_g1_'], 1 / (1 - df['_g1_']))

        pos = positivity(df=df, weights='_ipw_')
        print('======================================================================')
        print('                      Weight Positivity Diagnostics')
        print('======================================================================')
        print('If the mean of the weights is far from either the min or max, this may\n '
              'indicate the model is incorrect or positivity is violated')
        print('Average weight should be 2')
        print('----------------------------------------------------------------------')
        print('Mean weight:           ', round(pos[0], decimal))
        print('Standard Deviation:    ', round(pos[1], decimal))
        print('Minimum weight:        ', round(pos[2], decimal))
        print('Maximum weight:        ', round(pos[3], decimal))
        print('======================================================================\n')

    def standardized_mean_differences(self):
        """Calculates the standardized mean differences for all variables based on the inverse probability weights.

        Returns
        -------
        DataFrame
            Returns pandas DataFrame of calculated standardized mean differences. Columns are labels (variables labels),
            smd_u (unweighted standardized difference), and smd_w (weighted standardized difference)
        """
        df = self.df.copy()
        df['_ipw_'] = np.where(df[self.exposure] == 1, 1 / df['_g1_'], 1 / (1 - df['_g1_']))
        s = standardized_mean_differences(df=df, treatment=self.exposure,
                                          weight='_ipw_', formula=self.__mweight)
        return s

    def plot_kde(self, to_plot, bw_method='scott', fill=True, color='g', color_e='b', color_u='r'):
        """Generates density plots that can be used to check predictions qualitatively. Density plots can be generated
        for assess either positivity violations of the exposure model or the accuracy in predicting the outcome for
        the outcome model. The kernel density used is SciPy's Gaussian kernel. Either Scott's Rule or
        Silverman's Rule can be implemented.

        Parameters
        ------------
        to_plot : str, optional
            The plot to generate. Specifying 'exposure' returns only the density plot for treatment probabilities,
            and 'outcome' returns only the density plot for the outcome accuracy
        bw_method : str, optional
            Method used to estimate the bandwidth. Following SciPy, either 'scott' or 'silverman' are valid options
        fill : bool, optional
            Whether to color the area under the density curves. Default is true
        color : str, optional
            Color of the line/area for predicted outcomes minus observed outcomes. Default is Green
        color_e : str, optional
            Color of the line/area for the treated group. Default is Blue
        color_u : str, optional
            Color of the line/area for the treated group. Default is Red

        Returns
        ---------------
        matplotlib axes
        """
        if to_plot == 'exposure':
            ax = plot_kde(df=self.df, treatment=self.exposure, probability='_g1_',
                          bw_method=bw_method, fill=fill, color_e=color_e, color_u=color_u)
            ax.set_title("Kernel Density of Propensity Scores")

        elif to_plot == 'outcome':
            v = self._predicted_y_ - self.df[self.outcome]
            ax = plot_kde_accuracy(values=v.dropna(), bw_method=bw_method, fill=fill, color=color)
            ax.set_title("Kernel Density of Accuracy")

        else:
            raise ValueError("Please use one of the following options for `to_plot`; 'treatment', 'outcome'")

        return ax

    def plot_love(self, color_unweighted='r', color_weighted='b', shape_unweighted='o', shape_weighted='o'):
        """Generates a Love-plot to detail covariate balance based on the IPTW weights. Further details on the usage of
        this plot are available in Austin PC & Stuart EA 2015 https://onlinelibrary.wiley.com/doi/full/10.1002/sim.6607

        The Love plot generates a dashed line at standardized mean difference of 0.10. Ideally, weighted SMD are below
        this level. Below 0.20 may also be sufficient. Variables above this level may be unbalanced despite the
        weighting procedure. Different functional forms (or approaches like machine learning) may be worth considering

        Parameters
        ----------
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
        df = self.df.copy()
        df['_ipw_'] = np.where(df[self.exposure] == 1, 1 / df['_g1_'], 1 / (1 - df['_g1_']))

        ax = plot_love(df=df, treatment=self.exposure, weight='_ipw_', formula=self.__mweight,
                       color_unweighted=color_unweighted, color_weighted=color_weighted,
                       shape_unweighted=shape_unweighted, shape_weighted=shape_weighted)
        return ax
