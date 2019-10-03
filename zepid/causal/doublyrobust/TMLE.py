import warnings
import patsy
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.stats import logistic, norm

from zepid.causal.utils import propensity_score
from zepid.calc import probability_to_odds
from zepid.causal.utils import (exposure_machine_learner, outcome_machine_learner, missing_machine_learner, _bounding_,
                                plot_kde, plot_love, standardized_mean_differences, positivity,
                                plot_kde_accuracy, outcome_accuracy)


class TMLE:
    r"""Implementation of target maximum likelihood estimator. This implementation calculates TMLE for a
    time-fixed exposure and a single time-point outcome.It uses standard regression models to calculate the estimate
    of interest. The TMLE estimator allows users to instead use machine learning algorithms from sklearn.

    Note
    ----
    Support for machine learning will be dropped in v0.9.0 due to poor confidence interval coverage. There will be
    new cross-fitting procedures and classes for causal inference with machine learning.

    Parameters
    ----------
    df : DataFrame
        Pandas dataframe containing the variables of interest
    exposure : str
        Column label for the exposure of interest
    outcome : str
        Column label for the outcome of interest
    alpha : float, optional
        Alpha for confidence interval level. Default is 0.05
    continuous_bound : float, optional
        Optional argument to control the bounding feature for continuous outcomes. The bounding process may result
        in values of 0,1 which are undefined for logit(x). This parameter adds or substracts from the scenarios of
        0,1 respectively. Default value is 0.0005

    Note
    ----
    TMLE is a doubly-robust substitution estimator. TMLE obtains the target estimate in a single step. The
    single-step TMLE is described further by van der Laan. For further details, see the listed references.

    Continuous outcomes must be bounded between 0 and 1. TMLE does this automatically for the user. Additionally,
    the average treatment effect is estimate is back converted to the original scale of Y. When scaling Y as Y*,
    some values may take the value of 0 or 1, which breaks a logit(Y*) transformation. To avoid this issue, Y* is
    bounded by the `continuous_bound` argument. The default is 0.0005, the same as R's tmle

    The following is a general outline of the estimation process for TMLE

    1. Initial estimates for Y are predicted from a regression model. Expected values for each individual are
    generated under the scenarios of all treated vs all untreated

    .. math::

        E(Y|A, L)

    2. Predicted probabilities are generated from a regression model

    .. math::

        \pi_1 = \Pr(A=1|L)

    3. The 'clever covariate' is calculated by

    .. math::

        H_a(A=a,L) = \frac{I(A=1)}{\pi_1} - \frac{I(A=0)}{\pi_0}

    for each individual. Afterwards, the predicted Y is set as an offset in the following logit model and used to
    predict values under each treatment strategy after fitted

    .. math::

        \text{logit}(E(Y|A,L)) = \text{logit}(Y_a) + \sigma  H_a

    4. The targeted Psi is estimated, representing the causal effect of all treated vs. all untreated

    Confidence intervals are constructed using influence curves.

    Examples
    --------
    Setting up environment

    >>> from zepid import load_sample_data, spline
    >>> from zepid.causal.doublyrobust import TMLE
    >>> df = load_sample_data(False).dropna()
    >>> df[['cd4_rs1', 'cd4_rs2']] = spline(df, 'cd40', n_knots=3, term=2, restricted=True)

    Estimating TMLE using logistic regression

    >>> tmle = TMLE(df, exposure='art', outcome='dead')
    >>> # Specifying exposure/treatment model
    >>> tmle.exposure_model('male + age0 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>> # Specifying outcome model
    >>> tmle.outcome_model('art + male + age0 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>> # TMLE estimation procedure
    >>> tmle.fit()
    >>> # Printing main results
    >>> tmle.summary()
    >>> # Extracting risk difference and confidence intervals, respectively
    >>> tmle.risk_difference
    >>> tmle.risk_difference_ci

    Estimating TMLE with machine learning algorithm from sklearn

    >>> from sklearn.linear_model import LogisticRegression
    >>> log1 = LogisticRegression(penalty='l1', random_state=201)
    >>> tmle = TMLE(df, 'art', 'dead')
    >>> # custom_model allows specification of machine learning algorithms
    >>> tmle.exposure_model('male + age0 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', custom_model=log1)
    >>> tmle.outcome_model('male + age0 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', custom_model=log1)
    >>> tmle.fit()

    Demonstration of estimating g-model with symmetric bounds

    >>> tmle.exposure_model('male + age0 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', bound=0.05)

    Demonstration of estimating g-model with asymmetric bounds

    >>> tmle.exposure_model('male + age0 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', bound=[0.05, 0.9])

    References
    ----------
    Schuler MS, and Sherri R. "Targeted maximum likelihood estimation for causal inference in
    observational studies." American journal of epidemiology 185.1 (2017): 65-73.

    Van der Laan, MJ, and Sherri R. Targeted learning: causal inference for observational and experimental
    data. Springer Science & Business Media, 2011.

    Van Der Laan, MJ, Rubin D. "Targeted maximum likelihood learning." The International Journal of
    Biostatistics 2.1 (2006).

    Gruber S, van der Laan, MJ. (2011). tmle: An R package for targeted maximum likelihood estimation.
    """
    def __init__(self, df, exposure, outcome, alpha=0.05, continuous_bound=0.0005):
        # Going through missing data (that is not the outcome)
        if df.dropna(subset=[d for d in df.columns if d != outcome]).shape[0] != df.shape[0]:
            warnings.warn("There is missing data that is not the outcome in the data set. TMLE will drop "
                          "all missing data that is not missing outcome data. TMLE will fit "
                          + str(df.dropna(subset=[d for d in df.columns if d != outcome]).shape[0]) +
                          ' of ' + str(df.shape[0]) + ' observations', UserWarning)
            self.df = df.copy().dropna(subset=[d for d in df.columns if d != outcome]).reset_index()
        else:
            self.df = df.copy().reset_index()

        # Checking to see if missing outcome data occurs
        self._missing_indicator = '__missing_indicator__'
        if self.df.dropna(subset=[outcome]).shape[0] != self.df.shape[0]:
            self._miss_flag = True
            self.df[self._missing_indicator] = np.where(self.df[outcome].isna(), 0, 1)
        else:
            self._miss_flag = False
            self.df[self._missing_indicator] = 1

        # Detailed steps follow "Targeted Learning" chapter 4, figure 4.2 by van der Laan, Rose
        self.exposure = exposure
        self.outcome = outcome

        if df[outcome].dropna().value_counts().index.isin([0, 1]).all():
            self._continuous_outcome = False
            self._cb = 0.0
        else:
            self._continuous_outcome = True
            self._continuous_min = np.min(df[outcome])
            self._continuous_max = np.max(df[outcome])
            self._cb = continuous_bound
            self.df[outcome] = self._unit_bounds(y=df[outcome], mini=self._continuous_min,
                                                 maxi=self._continuous_max, bound=self._cb)

        self._out_model = None
        self._exp_model = None
        self._miss_model = None
        self._out_model_custom = False
        self._exp_model_custom = False
        self._miss_model_custom = False
        self._fit_exposure_model = False
        self._fit_outcome_model = False
        self._fit_missing_model = False
        self.alpha = alpha

        self.QA0W = None
        self.QA1W = None
        self.QAW = None
        self.g1W = None
        self.g0W = None
        self.m1W = None
        self.m0W = None
        self._epsilon = None

        self.risk_difference = None
        self.risk_difference_ci = None
        self.risk_difference_se = None
        self.risk_ratio = None
        self.risk_ratio_ci = None
        self.risk_ratio_se = None
        self.odds_ratio = None
        self.odds_ratio_ci = None
        self.odds_ratio_se = None
        self.average_treatment_effect = None
        self.average_treatment_effect_ci = None
        self.average_treatment_effect_se = None

    def exposure_model(self, model, custom_model=None, bound=False, print_results=True):
        """Estimation of Pr(A=1|L), which is termed as g(A=1|L) in the literature

        Parameters
        ----------
        model : str
            Independent variables to predict the exposure. Example) 'var1 + var2 + var3'
        custom_model : optional
            Input for a custom model that is used in place of the logit model (default). The model must have the
            "fit()" and  "predict()" attributes. Both sklearn and supylearner are supported as custom models. In the
            background, TMLE will fit the custom model and generate the predicted probablities
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
        self._exp_model = self.exposure + ' ~ ' + model
        self.__mweight = model

        # Step 3) Estimation of g-model (exposure model)
        if custom_model is None:
            fitmodel = propensity_score(self.df, self._exp_model, print_results=print_results)
            self.g1W = fitmodel.predict(self.df)

        # User-specified prediction model
        else:
            # TODO need to create smart warning system
            # warnings.warn("TMLE can result in confidence intervals below nominal coverage when used with "
            #              "certain machine learning algorithms")
            self._exp_model_custom = True
            data = patsy.dmatrix(model + ' - 1', self.df)
            self.g1W = exposure_machine_learner(xdata=np.asarray(data), ydata=np.asarray(self.df[self.exposure]),
                                                ml_model=custom_model, print_results=print_results)

        self.g0W = 1 - self.g1W
        if bound:  # Bounding predicted probabilities if requested
            self.g1W = _bounding_(self.g1W, bounds=bound)
            self.g0W = _bounding_(self.g0W, bounds=bound)

        self._fit_exposure_model = True

    def missing_model(self, model, custom_model=None, bound=False, print_results=True):
        """Estimation of Pr(M=1|A,L), which is the missing data mechanism for the outcome. The corresponding observation
        probabilities are used to update the clever covariates for estimation of Qn.

        The initial estimate of Q is still based on complete observations only

        Parameters
        ----------
        model : str
            Independent variables to predict the exposure. Example) 'var1 + var2 + var3'. The treatment must be
            included for the missing data model
        custom_model : optional
            Input for a custom model that is used in place of the logit model (default). The model must have the
            "fit()" and  "predict()" attributes. Both sklearn and supylearner are supported as custom models. In the
            background, TMLE will fit the custom model and generate the predicted probablities
        bound: float, list, optional
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

        self._miss_model = self._missing_indicator + ' ~ ' + model

        # Step 3b) Prediction for M if missing outcome data exists
        if custom_model is None:  # Logistic Regression model for predictions
            fitmodel = propensity_score(self.df, self._miss_model, print_results=print_results)
            dfx = self.df.copy()
            dfx[self.exposure] = 1
            self.m1W = fitmodel.predict(dfx)
            dfx = self.df.copy()
            dfx[self.exposure] = 0
            self.m0W = fitmodel.predict(dfx)

        # User-specified model
        else:
            # TODO need to create smart warning system
            # warnings.warn("TMLE can result in confidence intervals below nominal coverage when used with "
            #              "certain machine learning algorithms")

            self._miss_model_custom = True
            data = patsy.dmatrix(model + ' - 1', self.df)

            dfx = self.df.copy()
            dfx[self.exposure] = 1
            adata = patsy.dmatrix(model + ' - 1', dfx)
            dfx = self.df.copy()
            dfx[self.exposure] = 0
            ndata = patsy.dmatrix(model + ' - 1', dfx)

            self.m1W, self.m0W = missing_machine_learner(xdata=np.array(data),
                                                         mdata=self.df[self._missing_indicator],
                                                         all_a=adata, none_a=ndata,
                                                         ml_model=custom_model, print_results=print_results)

        if bound:  # Bounding predicted probabilities if requested
            self.m1W = _bounding_(self.m1W, bounds=bound)
            self.m0W = _bounding_(self.m0W, bounds=bound)

        self._fit_missing_model = True

    def outcome_model(self, model, custom_model=None, bound=False, print_results=True,
                      continuous_distribution='gaussian'):
        """Estimation of E(Y|A,L,M=1), which is also written sometimes as Q(A,W,M=1) or Pr(Y=1|A,W,M=1). Estimation
        of this model is based on complete observations of Y only

        Parameters
        ----------
        model : str
            Independent variables to predict the exposure. Example) 'var1 + var2 + var3'
        custom_model : optional
            Input for a custom model that is used in place of the logit model (default). The model must have the
            "fit()" and  "predict()" attributes. Both sklearn and supylearner are supported as custom models. In the
            background, TMLE will fit the custom model and generate the predicted probablities
        bound : bool, optional
            This argument should ONLY be used if the outcome is continuous. Value between 0,1 to truncate the bounded
            predicted outcomes. Default is `False`, meaning no truncation of predicted outcomes occurs (unless a
            predicted outcome is outside the bounded continuous outcome). Providing a single float assumes symmetric
            trunctation. A list of floats can be provided for asymmetric trunctation.
        print_results : bool, optional
            Whether to print the fitted model results. Default is True (prints results)
        continuous_distribution : str, optional
            Distribution to use for continuous outcomes. Options are 'gaussian' for normal distributions and 'poisson'
            for Poisson distributions
        """
        self._out_model = self.outcome + ' ~ ' + model

        if self._miss_flag:
            cc = self.df.copy().dropna()
        else:
            cc = self.df.copy()

        # Step 1) Prediction for Q (estimation of Q-model)
        if custom_model is None:  # Logistic Regression model for predictions
            self._continuous_type = continuous_distribution
            if self._continuous_outcome:
                if (continuous_distribution == 'gaussian') or (continuous_distribution == 'normal'):
                    f = sm.families.family.Gaussian()
                elif continuous_distribution == 'poisson':
                    f = sm.families.family.Poisson()
                else:
                    raise ValueError("Only 'gaussian' and 'poisson' distributions are supported")
                log = smf.glm(self._out_model, cc, family=f).fit()
            else:
                f = sm.families.family.Binomial()
                log = smf.glm(self._out_model, cc, family=f).fit()

            if print_results:
                print('\n----------------------------------------------------------------')
                print('MODEL: ' + self._out_model)
                print('-----------------------------------------------------------------')
                print(log.summary())

            # Step 2) Estimation under the scenarios
            dfx = self.df.copy()
            dfx[self.exposure] = 1
            self.QA1W = log.predict(dfx)
            dfx = self.df.copy()
            dfx[self.exposure] = 0
            self.QA0W = log.predict(dfx)

        # User-specified model
        else:
            # TODO need to create smart warning system
            # warnings.warn("TMLE can result in confidence intervals below nominal coverage when used with "
            #              "certain machine learning algorithms")
            self._out_model_custom = True
            data = patsy.dmatrix(model + ' - 1', cc)

            dfx = self.df.copy()
            dfx[self.exposure] = 1
            adata = patsy.dmatrix(model + ' - 1', dfx)
            dfx = self.df.copy()
            dfx[self.exposure] = 0
            ndata = patsy.dmatrix(model + ' - 1', dfx)

            self.QA1W, self.QA0W = outcome_machine_learner(xdata=np.asarray(data),
                                                           ydata=np.asarray(cc[self.outcome]),
                                                           all_a=adata, none_a=ndata,
                                                           ml_model=custom_model,
                                                           continuous=self._continuous_outcome,
                                                           print_results=print_results)

        if not bound:  # Bounding predicted probabilities if requested
            bound = self._cb

        # This bounding step prevents continuous outcomes from being outside the range
        self.QA1W = _bounding_(self.QA1W, bounds=bound)
        self.QA0W = _bounding_(self.QA0W, bounds=bound)
        self.QAW = self.QA1W * self.df[self.exposure] + self.QA0W * (1 - self.df[self.exposure])
        self._fit_outcome_model = True

    def fit(self):
        """Calculate the effect measures from the predicted exposure probabilities and predicted outcome values using
        the TMLE procedure. Confidence intervals are calculated using influence curves.

        Note
        ----
        Exposure and outcome models must be specified prior to `fit()`

        Returns
        -------
        TMLE gains `risk_difference`, `risk_ratio`, and `odds_ratio` for binary outcomes and
        `average _treatment_effect` for continuous outcomes
        """
        if (self._fit_exposure_model is False) or (self._fit_outcome_model is False):
            raise ValueError('The exposure and outcome models must be specified before the psi estimate can '
                             'be generated')
        if self._miss_flag and not self._fit_missing_model:
            warnings.warn("No missing data model has been specified. All missing outcome data is assumed to be "
                          "missing completely at random. To relax this assumption to outcome data is missing at random"
                          "please use the `missing_model()` function", UserWarning)

        # Step 4) Calculating clever covariate (HAW)
        if self._miss_flag and self._fit_missing_model:
            self.g1W_total = self.g1W * self.m1W
            self.g0W_total = self.g0W * self.m0W
        else:
            self.g1W_total = self.g1W
            self.g0W_total = self.g0W
        H1W = self.df[self.exposure] / self.g1W_total
        H0W = -(1 - self.df[self.exposure]) / self.g0W_total
        HAW = H1W + H0W

        # Step 5) Estimating TMLE
        f = sm.families.family.Binomial()
        y = self.df[self.outcome]
        log = sm.GLM(y, np.column_stack((H1W, H0W)), offset=np.log(probability_to_odds(self.QAW)),
                     family=f, missing='drop').fit()
        self._epsilon = log.params
        Qstar1 = logistic.cdf(np.log(probability_to_odds(self.QA1W)) + self._epsilon[0] / self.g1W_total)
        Qstar0 = logistic.cdf(np.log(probability_to_odds(self.QA0W)) - self._epsilon[1] / self.g0W_total)
        Qstar = log.predict(np.column_stack((H1W, H0W)), offset=np.log(probability_to_odds(self.QAW)))

        # Step 6) Calculating Psi
        if self.alpha == 0.05:  # Without this, won't match R exactly. R relies on 1.96, while I use SciPy
            zalpha = 1.96
        else:
            zalpha = norm.ppf(1 - self.alpha / 2, loc=0, scale=1)

        # p-values are not implemented (doing my part to enforce CL over p-values)
        delta = np.where(self.df[self._missing_indicator] == 1, 1, 0)
        if self._continuous_outcome:
            # Calculating Average Treatment Effect
            Qstar = self._unit_unbound(Qstar, mini=self._continuous_min, maxi=self._continuous_max)
            Qstar1 = self._unit_unbound(Qstar1, mini=self._continuous_min, maxi=self._continuous_max)
            Qstar0 = self._unit_unbound(Qstar0, mini=self._continuous_min, maxi=self._continuous_max)

            self.average_treatment_effect = np.nanmean(Qstar1 - Qstar0)
            # Influence Curve for CL
            y_unbound = self._unit_unbound(self.df[self.outcome], mini=self._continuous_min, maxi=self._continuous_max)
            ic = np.where(delta == 1,
                          HAW * (y_unbound - Qstar) + (Qstar1 - Qstar0) - self.average_treatment_effect,
                          Qstar1 - Qstar0 - self.average_treatment_effect)
            seIC = np.sqrt(np.nanvar(ic, ddof=1) / self.df.shape[0])
            self.average_treatment_effect_se = seIC
            self.average_treatment_effect_ci = [self.average_treatment_effect - zalpha * seIC,
                                                self.average_treatment_effect + zalpha * seIC]
        else:
            # Calculating Risk Difference
            self.risk_difference = np.nanmean(Qstar1 - Qstar0)
            # Influence Curve for CL
            ic = np.where(delta == 1,
                          HAW * (self.df[self.outcome] - Qstar) + (Qstar1 - Qstar0) - self.risk_difference,
                          (Qstar1 - Qstar0) - self.risk_difference)
            seIC = np.sqrt(np.nanvar(ic, ddof=1) / self.df.shape[0])
            self.risk_difference_se = seIC
            self.risk_difference_ci = [self.risk_difference - zalpha * seIC,
                                       self.risk_difference + zalpha * seIC]

            # Calculating Risk Ratio
            self.risk_ratio = np.nanmean(Qstar1) / np.nanmean(Qstar0)
            # Influence Curve for CL
            ic = np.where(delta == 1,
                          (1 / np.mean(Qstar1) * (H1W * (self.df[self.outcome] - Qstar) + Qstar1 - np.mean(Qstar1)) -
                           (1/np.mean(Qstar0)) * (-1 * H0W * (self.df[self.outcome] - Qstar) + Qstar0 - np.mean(Qstar0))),
                          (Qstar1 - np.mean(Qstar1)) + Qstar0 - np.mean(Qstar0))

            seIC = np.sqrt(np.nanvar(ic, ddof=1) / self.df.shape[0])
            self.risk_ratio_se = seIC
            self.risk_ratio_ci = [np.exp(np.log(self.risk_ratio) - zalpha * seIC),
                                  np.exp(np.log(self.risk_ratio) + zalpha * seIC)]

            # Calculating Odds Ratio
            self.odds_ratio = (np.nanmean(Qstar1) / (1 - np.nanmean(Qstar1)
                                                     )) / (np.nanmean(Qstar0) / (1 - np.nanmean(Qstar0)))
            # Influence Curve for CL
            ic = np.where(delta == 1,
                          ((1 / (np.nanmean(Qstar1)*(1 - np.nanmean(Qstar1))) *
                            (H1W * (self.df[self.outcome] - Qstar) + Qstar1)) -
                           (1 / (np.nanmean(Qstar0)*(1 - np.nanmean(Qstar0))) *
                            (-1 * H0W * (self.df[self.outcome] - Qstar) + Qstar0))),

                          ((1 / (np.nanmean(Qstar1) * (1 - np.nanmean(Qstar1))) * Qstar1 -
                           (1 / (np.nanmean(Qstar0) * (1 - np.nanmean(Qstar0))) * Qstar0))))
            seIC = np.sqrt(np.nanvar(ic, ddof=1) / self.df.shape[0])
            self.odds_ratio_se = seIC
            self.odds_ratio_ci = [np.exp(np.log(self.odds_ratio) - zalpha * seIC),
                                  np.exp(np.log(self.odds_ratio) + zalpha * seIC)]

    def summary(self, decimal=3):
        """Prints summary of the estimated average causal effects

        Parameters
        ----------
        decimal : int, optional
            Number of decimal places to display. Default is 3
        """
        if (self._fit_exposure_model is False) or (self._fit_exposure_model is False):
            raise ValueError('The exposure and outcome models must be specified before the psi estimate can '
                             'be generated')

        print('======================================================================')
        print('                Targeted Maximum Likelihood Estimator                 ')
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
        if self._miss_model_custom and self._miss_model is not None:
            m = 'User-specified'
        elif self._miss_model is None:
            m = 'None'
        else:
            m = 'Logistic'

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
            print('Average Treatment Effect: ', round(float(self.average_treatment_effect), decimal))
            print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: (' +
                  str(round(self.average_treatment_effect_ci[0], decimal)), ',',
                  str(round(self.average_treatment_effect_ci[1], decimal)) + ')')
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
        if not self._fit_outcome_model or not self._fit_exposure_model:
            raise ValueError("The exposure_model and outcome_model function must be ran before any diagnostics")

        if self._fit_missing_model:
            ps = self.g1W * np.where(self.df[self.exposure] == 1, self.m1W, self.m0W)
        else:
            ps = self.g1W

        # Weight diagnostics
        print('\tExposure Model Diagnostics')
        self.positivity(decimal=decimal)

        print('======================================================================')
        print('                  Standardized Mean Differences')
        print('======================================================================')
        print(self.standardized_mean_differences().set_index(keys='labels'))
        print('======================================================================\n')

        # Outcome accuracy diagnostics
        print('\tOutcome Model Diagnostics')
        v = self.QAW - self.df[self.outcome]
        outcome_accuracy(true=self.df[self.outcome], predicted=self.QAW, decimal=decimal)

        df = self.df.copy()
        df['_ipw_'] = np.where(df[self.exposure] == 1, 1 / ps, 1 / (1 - ps))
        df['_g1_'] = ps

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
        --------------
        decimal : int, optional
            Number of decimal places to display. Default is three

        Returns
        --------------
        None
            Prints the positivity results to the console but does not return any objects
        """
        if self._fit_missing_model:
            ps = self.g1W * np.where(self.df[self.exposure] == 1, self.m1W, self.m0W)
        else:
            ps = self.g1W

        df = self.df.copy()
        df['_ipw_'] = np.where(df[self.exposure] == 1, 1 / ps, 1 / (1 - ps))
        pos = positivity(df=df, weights='_ipw_')
        print('======================================================================')
        print('                   Weight Positivity Diagnostics')
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
        if self._fit_missing_model:
            ps = self.g1W * np.where(self.df[self.exposure] == 1, self.m1W, self.m0W)
        else:
            ps = self.g1W

        df = self.df.copy()
        df['_ipw_'] = np.where(df[self.exposure] == 1, 1 / ps, 1 / (1 - ps))
        s = standardized_mean_differences(df=df, treatment=self.exposure,
                                          weight='_ipw_', formula=self.__mweight)
        return s

    def plot_kde(self, to_plot, bw_method='scott', fill=True,
                 color='g', color_e='b', color_u='r'):
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
        if not self._fit_outcome_model or not self._fit_exposure_model:
            raise ValueError("The exposure_model and outcome_model function must be ran before any diagnostics")

        if to_plot == 'exposure':
            if self._fit_missing_model:
                ps = self.g1W * np.where(self.df[self.exposure] == 1, self.m1W, self.m0W)
            else:
                ps = self.g1W
            df = self.df.copy()
            df['_g1_'] = ps
            ax = plot_kde(df=df, treatment=self.exposure, probability='_g1_',
                          bw_method=bw_method, fill=fill, color_e=color_e, color_u=color_u)
            ax.set_title("Kernel Density of Propensity Scores")

        elif to_plot == 'outcome':
            v = self.QAW - self.df[self.outcome]
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
        if not self._fit_outcome_model or not self._fit_exposure_model:
            raise ValueError("The exposure_model and outcome_model function must be ran before any diagnostics")

        if self._fit_missing_model:
            ps = self.g1W * np.where(self.df[self.exposure] == 1, self.m1W, self.m0W)
        else:
            ps = self.g1W
        df = self.df.copy()
        df['_g1_'] = ps
        df['_ipw_'] = np.where(df[self.exposure] == 1, 1 / df['_g1_'], 1 / (1 - df['_g1_']))
        ax = plot_love(df=df, treatment=self.exposure, weight='_ipw_', formula=self.__mweight,
                       color_unweighted=color_unweighted, color_weighted=color_weighted,
                       shape_unweighted=shape_unweighted, shape_weighted=shape_weighted)
        return ax

    @staticmethod
    def _unit_bounds(y, mini, maxi, bound):
        # bounding for continuous outcomes
        v = (y - mini) / (maxi - mini)
        v = np.where(np.less(v, bound), bound, v)
        v = np.where(np.greater(v, 1-bound), 1-bound, v)
        return v

    @staticmethod
    def _unit_unbound(ystar, mini, maxi):
        # unbounding of bounded continuous outcomes
        return ystar*(maxi - mini) + mini
