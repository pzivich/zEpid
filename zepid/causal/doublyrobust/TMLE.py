import math
import warnings
import patsy
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import logistic, norm

from zepid.causal.ipw import propensity_score
from zepid.calc import probability_to_odds


def _exposure_machine_learner(xdata, ydata, ml_model, print_results=True):
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


def _outcome_machine_learner(xdata, ydata, all_a, none_a, ml_model, continuous, print_results=True):
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


def _missing_machine_learner(xdata, mdata, all_a, none_a, ml_model, print_results=True):
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


class TMLE:
    def __init__(self, df, exposure, outcome, measure=None, alpha=0.05, continuous_bound=0.0005):
        """Implementation of a single time-point target maximum likelihood estimator. It uses standard
        regression models to calculate the estimate of interest. The TMLE estimator allows users are able to directly
        input predicted outcomes from other methods (like machine learning algorithms from sklearn).

        Parameters
        ----------
        df : DataFrame
            Pandas dataframe containing the variables of interest
        exposure : str
            Column label for the exposure of interest
        outcome : str
            Column label for the outcome of interest
        measure : str, optional
            Depreciated. By default risk difference, risk ratio, and odds ratio are all calculated
        alpha : int, optional
            Alpha for confidence interval level. Default is 0.05
        continuous_bound : float, optional
            Optional argument to control the bounding feature for continuous outcomes. The bounding process may result
            in values of 0,1 which are undefined for logit(x). This parameter adds or substracts from the scenarios of
            0,1 respectively. Default value is 0.0005

        Notes
        -----
        TMLE is a double robust substitution estimator. TMLE obtains the target estimate in a single step. The
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

            \Pr(A=1|L)

        3. The predicted Y is merged together with the IPW using

        .. math::

            H_a(A=a,L) = \frac{I(A=1)}{\pi_1} - \frac{I(A=0)}{\pi_0}

        for each individual. Afterwards, the predicted Y is set as an offset in the following logit model and used to
        predict values under each treatment strategy after fitted

        .. math::

            logit(E(Y|A,L)) = logit(Y_a) + \sigma * H_a

        4. The targeted Psi is estimated, representing the causal effect of all treated vs. all untreated

        Confidence intervals are constructed using influence curve theory.

        Examples
        --------
        Setting up environment
        >>>from zepid import load_sample_data, spline
        >>>from zepid.causal.doublyrobust import TMLE
        >>>df = load_sample_data(False).dropna()
        >>>df[['cd4_rs1', 'cd4_rs2']] = spline(df, 'cd40', n_knots=3, term=2, restricted=True)

        Estimating TMLE using logistic regression
        >>>tmle = TMLE(df, exposure='art', outcome='dead')
        >>># Specifying exposure/treatment model
        >>>tmle.exposure_model('male + age0 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
        >>># Specifying outcome model
        >>>tmle.outcome_model('art + male + age0 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
        >>># TMLE estimation procedure
        >>>tmle.fit()
        >>># Printing main results
        >>>tmle.summary()
        >>># Extracting risk difference and confidence intervals, respectively
        >>>tmle.risk_difference
        >>>tmle.risk_difference_ci

        Estimating TMLE with machine learning algorithm from sklearn
        >>>from sklearn.linear_model import LogisticRegression
        >>>log1 = LogisticRegression(penalty='l1', random_state=201)
        >>>tmle = TMLE(df, 'art', 'dead')
        >>># custom_model allows specification of machine learning algorithms
        >>>tmle.exposure_model('male + age0 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', custom_model=log1)
        >>>tmle.outcome_model('male + age0 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', custom_model=log1)
        >>>tmle.fit()

        Demonstration of estimating g-model with symmetric bounds
        >>>tmle.exposure_model('male + age0 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', bound=0.05)

        Demonstration of estimating g-model with asymmetric bounds
        >>>tmle.exposure_model('male + age0 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', bound=[0.05, 0.9])

        References
        ----------
        Schuler, Megan S., and Sherri Rose. "Targeted maximum likelihood estimation for causal inference in
        observational studies." American journal of epidemiology 185.1 (2017): 65-73.

        Van der Laan, Mark J., and Sherri Rose. Targeted learning: causal inference for observational and experimental
        data. Springer Science & Business Media, 2011.

        Van Der Laan, Mark J., and Daniel Rubin. "Targeted maximum likelihood learning." The International Journal of
        Biostatistics 2.1 (2006).

        Gruber, S., & van der Laan, M. J. (2011). tmle: An R package for targeted maximum likelihood estimation.
        """
        if measure is not None:
            warnings.warn('As of v0.4.2, TMLE defaults to calculate all implemented measures (RD, RR, OR)', UserWarning)

        # Going through missing data (that is not the outcome)
        if df.dropna(subset=[d for d in df.columns if d != outcome]).shape[0] != df.shape[0]:
            warnings.warn("There is missing data that is not the outcome in the data set. TMLE will drop "
                          "all missing data that is not missing outcome data. TMLE will fit "
                          + str(df.dropna(subset=[d for d in df.columns if d != outcome]).shape[0]) +
                          ' of ' + str(df.shape[0]) + ' observations', UserWarning)
            self.df = df.copy().dropna(subset=[d for d in df.columns if d != outcome]).reset_index()
            print(self.df.shape[0])
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
        self._exposure = exposure
        self._outcome = outcome

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
        self.risk_ratio = None
        self.risk_ratio_ci = None
        self.odds_ratio = None
        self.odds_ratio_ci = None
        self.average_treatment_effect = None
        self.average_treatment_effect_ic = None

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
            inference becomes limited to the restricted population. Default is False, meaning no truncation of
            predicted probabilities occurs. Providing a single float assumes symmetric trunctation. A collection of
            floats can be provided for asymmetric trunctation
        print_results : bool, optional
            Whether to print the fitted model results. Default is True (prints results)
        """
        self._exp_model = self._exposure + ' ~ ' + model

        # Step 3) Estimation of g-model (exposure model)
        if custom_model is None:
            fitmodel = propensity_score(self.df, self._exp_model, print_results=print_results)
            self.g1W = fitmodel.predict(self.df)

        # User-specified prediction model
        else:
            data = patsy.dmatrix(model + ' - 1', self.df)
            self.g1W = _exposure_machine_learner(xdata=np.asarray(data), ydata=np.asarray(self.df[self._exposure]),
                                                 ml_model=custom_model, print_results=print_results)

        self.g0W = 1 - self.g1W
        if bound:  # Bounding predicted probabilities if requested
            self.g1W = self._bounding(self.g1W, bounds=bound)
            self.g0W = self._bounding(self.g0W, bounds=bound)

        self._fit_exposure_model = True

    def missing_model(self, model, custom_model=None, print_results=True):
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
        print_results : bool, optional
            Whether to print the fitted model results. Default is True (prints results)
        """
        # Error if no missing outcome data
        if not self._miss_flag:
            raise ValueError("No missing outcome data is present in the data set")

        # Warning if exposure is not included in the missingness of outcome model
        if self._exposure not in model:
            warnings.warn("For the specified missing outcome model, the exposure variable should be included in the "
                          "model", UserWarning)

        self._miss_model = self._missing_indicator + ' ~ ' + model

        # Step 3b) Prediction for M if missing outcome data exists
        if custom_model is None:  # Logistic Regression model for predictions
            fitmodel = propensity_score(self.df, self._miss_model, print_results=print_results)
            dfx = self.df.copy()
            dfx[self._exposure] = 1
            self.m1W = fitmodel.predict(dfx)
            dfx = self.df.copy()
            dfx[self._exposure] = 0
            self.m0W = fitmodel.predict(dfx)

        # User-specified model
        else:
            data = patsy.dmatrix(model + ' - 1', self.df)

            dfx = self.df.copy()
            dfx[self._exposure] = 1
            adata = patsy.dmatrix(model + ' - 1', dfx)
            dfx = self.df.copy()
            dfx[self._exposure] = 0
            ndata = patsy.dmatrix(model + ' - 1', dfx)

            self.m1W, self.m0W = _missing_machine_learner(xdata=np.array(data),
                                                          mdata=self.df[self._missing_indicator],
                                                          all_a=adata, none_a=ndata,
                                                          ml_model=custom_model, print_results=print_results)

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
            Value between 0,1 to truncate predicted outcomes. Helps to avoid near positivity violations. Default is
            `False`, meaning no truncation of predicted outcomes occurs (unless a predicted outcome is outside the
            bounded continuous outcome). Providing a single float assumes symmetric trunctation. A collection of
            floats can be provided for asymmetric trunctation
        print_results : bool, optional
            Whether to print the fitted model results. Default is True (prints results)
        continuous_distribution : str, optional
            Distribution to use for continuous outcomes. Options are 'gaussian' for normal distributions and 'poisson'
            for Poisson distributions
        """
        self._out_model = self._outcome + ' ~ ' + model

        if self._miss_flag:
            cc = self.df.copy().dropna()
        else:
            cc = self.df.copy()

        # Step 1) Prediction for Q (estimation of Q-model)
        if custom_model is None:  # Logistic Regression model for predictions
            if self._continuous_outcome:
                if continuous_distribution == 'gaussian':
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
            dfx[self._exposure] = 1
            self.QA1W = log.predict(dfx)
            dfx = self.df.copy()
            dfx[self._exposure] = 0
            self.QA0W = log.predict(dfx)

        # User-specified model
        else:
            data = patsy.dmatrix(model + ' - 1', cc)

            dfx = self.df.copy()
            dfx[self._exposure] = 1
            adata = patsy.dmatrix(model + ' - 1', dfx)
            dfx = self.df.copy()
            dfx[self._exposure] = 0
            ndata = patsy.dmatrix(model + ' - 1', dfx)

            self.QA1W, self.QA0W = _outcome_machine_learner(xdata=np.asarray(data),
                                                            ydata=np.asarray(cc[self._outcome]),
                                                            all_a=adata, none_a=ndata,
                                                            ml_model=custom_model,
                                                            continuous=self._continuous_outcome,
                                                            print_results=print_results)

        if not bound:  # Bounding predicted probabilities if requested
            bound = self._cb

        # This bounding step prevents continuous outcomes from being outside the range
        self.QA1W = self._bounding(self.QA1W, bounds=bound)
        self.QA0W = self._bounding(self.QA0W, bounds=bound)
        self.QAW = self.QA1W * self.df[self._exposure] + self.QA0W * (1 - self.df[self._exposure])
        self._fit_outcome_model = True

    def fit(self):
        """Estimates risk difference, risk ratio, and odds ratio based on the gAW and QAW. Confidence intervals come
        from the influence curve

        Returns
        -------
        TMLE gains Psi and confint attributes
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
        H1W = self.df[self._exposure] / self.g1W_total
        H0W = -(1 - self.df[self._exposure]) / self.g0W_total
        HAW = H1W + H0W

        # Step 5) Estimating TMLE
        f = sm.families.family.Binomial()
        y = self.df[self._outcome]
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
            y_unbound = self._unit_unbound(self.df[self._outcome], mini=self._continuous_min, maxi=self._continuous_max)
            ic = np.where(delta == 1,
                          HAW * (y_unbound - Qstar) + (Qstar1 - Qstar0) - self.average_treatment_effect,
                          Qstar1 - Qstar0 - self.average_treatment_effect)
            varIC = np.nanvar(ic, ddof=1) / self.df.shape[0]
            self.average_treatment_effect_ic = [self.average_treatment_effect - zalpha * math.sqrt(varIC),
                                                self.average_treatment_effect + zalpha * math.sqrt(varIC)]
        else:
            # Calculating Risk Difference
            self.risk_difference = np.nanmean(Qstar1 - Qstar0)
            # Influence Curve for CL
            ic = np.where(delta == 1,
                          HAW * (self.df[self._outcome] - Qstar) + (Qstar1 - Qstar0) - self.risk_difference,
                          (Qstar1 - Qstar0) - self.risk_difference)
            varIC = np.nanvar(ic, ddof=1) / self.df.shape[0]
            self.risk_difference_ci = [self.risk_difference - zalpha * np.sqrt(varIC),
                                       self.risk_difference + zalpha * np.sqrt(varIC)]

            # Calculating Risk Ratio
            self.risk_ratio = np.nanmean(Qstar1) / np.nanmean(Qstar0)
            # Influence Curve for CL
            ic = np.where(delta == 1,
                          (1/np.mean(Qstar1) * (H1W * (self.df[self._outcome] - Qstar) + Qstar1 - np.mean(Qstar1)) -
                           (1/np.mean(Qstar0)) * (-1*H0W*(self.df[self._outcome] - Qstar) + Qstar0 - np.mean(Qstar0))),
                          (Qstar1 - np.mean(Qstar1)) + Qstar0 - np.mean(Qstar0))

            varIC = np.nanvar(ic, ddof=1) / self.df.shape[0]
            self.risk_ratio_ci = [np.exp(np.log(self.risk_ratio) - zalpha * np.sqrt(varIC)),
                                  np.exp(np.log(self.risk_ratio) + zalpha * np.sqrt(varIC))]

            # Calculating Odds Ratio
            self.odds_ratio = (np.nanmean(Qstar1) / (1 - np.nanmean(Qstar1)
                                                     )) / (np.nanmean(Qstar0) / (1 - np.nanmean(Qstar0)))
            # Influence Curve for CL
            ic = np.where(delta == 1,
                          ((1/(np.nanmean(Qstar1)*(1 - np.nanmean(Qstar1))) *
                            (H1W*(self.df[self._outcome] - Qstar) + Qstar1)) -
                           (1/(np.nanmean(Qstar0)*(1 - np.nanmean(Qstar0))) *
                            (-1*H0W*(self.df[self._outcome] - Qstar) + Qstar0))),

                          ((1 / (np.nanmean(Qstar1) * (1 - np.nanmean(Qstar1))) * Qstar1 -
                           (1 / (np.nanmean(Qstar0) * (1 - np.nanmean(Qstar0))) * Qstar0))))
            varIC = np.nanvar(ic, ddof=1) / self.df.shape[0]
            self.odds_ratio_ci = [np.exp(np.log(self.odds_ratio) - zalpha * np.sqrt(varIC)),
                                  np.exp(np.log(self.odds_ratio) + zalpha * np.sqrt(varIC))]

    def summary(self, decimal=3):
        """Prints summary of model results

        Parameters
        ----------
        decimal : int, optional
            Number of decimal places to display. Default is 3
        """
        if (self._fit_exposure_model is False) or (self._fit_exposure_model is False):
            raise ValueError('The exposure and outcome models must be specified before the psi estimate can '
                             'be generated')

        if self._continuous_outcome:
            print('----------------------------------------------------------------------')
            print('Average Treatment Effect: ', round(float(self.average_treatment_effect), decimal))
            print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: (' +
                  str(round(self.average_treatment_effect_ic[0], decimal)), ',',
                  str(round(self.average_treatment_effect_ic[1], decimal)) + ')')
            print('----------------------------------------------------------------------')
        else:
            print('----------------------------------------------------------------------')
            print('Risk Difference: ', round(float(self.risk_difference), decimal))
            print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: (' +
                  str(round(self.risk_difference_ci[0], decimal)), ',',
                  str(round(self.risk_difference_ci[1], decimal)) + ')')
            print('----------------------------------------------------------------------')
            print('Risk Ratio: ', round(float(self.risk_ratio), decimal))
            print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: (' +
                  str(round(self.risk_ratio_ci[0], decimal)), ',',
                  str(round(self.risk_ratio_ci[1], decimal)) + ')')
            print('----------------------------------------------------------------------')
            print('Odds Ratio: ', round(float(self.odds_ratio), decimal))
            print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: (' +
                  str(round(self.odds_ratio_ci[0], decimal)), ',',
                  str(round(self.odds_ratio_ci[1], decimal)) + ')')
            print('----------------------------------------------------------------------')

    @staticmethod
    def _bounding(v, bounds):
        """Background function to perform bounding feature. Not intended for users to access

        v:
            -Values to be bounded
        bounds:
            -Percentile thresholds for bounds
        """
        if type(bounds) is float:  # Symmetric bounding
            if bounds < 0 or bounds > 1:
                raise ValueError('Bound value must be between (0, 1)')
            v = np.where(v < bounds, bounds, v)
            v = np.where(v > 1-bounds, 1-bounds, v)
        elif type(bounds) is str:  # Catching string inputs
            raise ValueError('Bounds must either be a float between (0, 1), or a collection of floats between (0, 1)')
        else:  # Asymmetric bounds
            if bounds[0] > bounds[1]:
                raise ValueError('Bound thresholds must be listed in ascending order')
            if len(bounds) > 2:
                warnings.warn('It looks like your specified bounds is more than two floats. Only the first two '
                              'specified bounds are used by the bound statement. So only ' +
                              str(bounds[0:2]) + ' will be used', UserWarning)
            if type(bounds[0]) is str or type(bounds[1]) is str:
                raise ValueError('Bounds must be floats between (0, 1)')
            if (bounds[0] < 0 or bounds[1] > 1) or (bounds[0] < 0 or bounds[1] > 1):
                raise ValueError('Both bound values must be between (0, 1)')
            v = np.where(v < bounds[0], bounds[0], v)
            v = np.where(v > bounds[1], bounds[1], v)
        return v

    @staticmethod
    def _unit_bounds(y, mini, maxi, bound):
        v = (y - mini) / (maxi - mini)
        v = np.where(np.less(v, bound), bound, v)
        v = np.where(np.greater(v, 1-bound), 1-bound, v)
        return v

    @staticmethod
    def _unit_unbound(ystar, mini, maxi):
        return ystar*(maxi - mini) + mini
