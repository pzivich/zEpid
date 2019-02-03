import math
import warnings
import patsy
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import logistic, norm

from zepid.causal.ipw import propensity_score
from zepid.calc import probability_to_odds


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
            No longer supported. By default risk difference, risk ratio, and odds ratio are all calculated
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
        if df.dropna().shape[0] != df.shape[0]:
            warnings.warn("There is missing data in the dataset. By default, TMLE will drop all missing data. TMLE will"
                          "fit "+str(df.dropna().shape[0])+' of '+str(df.shape[0])+' observations', UserWarning)

        # Detailed steps follow "Targeted Learning" chapter 4, figure 4.2 by van der Laan, Rose
        if measure is not None:
            warnings.warn('As of v0.4.2, TMLE defaults to calculate all implemented measures (RD, RR, OR)', UserWarning)
        self.df = df.copy().dropna().reset_index()
        self._exposure = exposure
        self._outcome = outcome

        if df[outcome].dropna().value_counts().index.isin([0, 1]).all():
            self._continuous_outcome = False
        else:
            self._continuous_outcome = True
            self._continuous_min = np.min(df[outcome])
            self._continuous_max = np.max(df[outcome])
            self._cb = continuous_bound
            self.df[outcome] = self._unit_bounds(y=df[outcome], mini=self._continuous_min,
                                                 maxi=self._continuous_max, bound=self._cb)

        self._out_model = None
        self._exp_model = None
        self._fit_exposure_model = False
        self._fit_outcome_model = False
        self.alpha = alpha

        self.QA0W = None
        self.QA1W = None
        self.QAW = None
        self.g1W = None
        self.g0W = None
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
            try:
                fm = custom_model.fit(X=data, y=self.df[self._exposure])
            except TypeError:
                raise TypeError("Currently custom_model must have the 'fit' function with arguments 'X', 'y'. This "
                                "covers both sklearn and supylearner. If there is a predictive model you would "
                                "like to use, please open an issue at https://github.com/pzivich/zepid and I "
                                "can work on adding support")
            if print_results and hasattr(fm, 'summarize'):
                fm.summarize()
            if hasattr(fm, 'predict_proba'):
                self.g1W = fm.predict_proba(data)[:, 1]
            elif hasattr(fm, 'predict'):
                self.g1W = fm.predict(data)
            else:
                raise ValueError("Currently custom_model must have 'predict' or 'predict_proba' attribute")

        self.g0W = 1 - self.g1W
        if bound:  # Bounding predicted probabilities if requested
            self.g1W = self._bounding(self.g1W, bounds=bound)
        if bound:  # Bounding predicted probabilities if requested
            self.g0W = self._bounding(self.g0W, bounds=bound)

        self._fit_exposure_model = True

    def outcome_model(self, model, custom_model=None, print_results=True, continous_distribution='gaussian'):
        """Estimation of E(Y|A,L), which is also written sometimes as Q(A,W) or Pr(Y=1|A,W)

        Parameters
        ----------
        model : str
            Independent variables to predict the exposure. Example) 'var1 + var2 + var3'
        custom_model : optional
            Input for a custom model that is used in place of the logit model (default). The model must have the
            "fit()" and  "predict()" attributes. Both sklearn and supylearner are supported as custom models. In the
            background, TMLE will fit the custom model and generate the predicted probablities
        print_results : bool, optional
            Whether to print the fitted model results. Default is True (prints results)
        """
        self._out_model = self._outcome + ' ~ ' + model

        # Step 1) Prediction for Q (estimation of Q-model)
        if custom_model is None:  # Logistic Regression model for predictions
            if self._continuous_outcome:
                if continous_distribution == 'gaussian':
                    f = sm.families.family.Gaussian()
                elif continous_distribution == 'poisson':
                    f = sm.families.family.Poisson()
                else:
                    raise ValueError("Only 'gaussian' and 'poisson' distributions are supported")
                log = smf.glm(self._out_model, self.df, family=f).fit()
            else:
                f = sm.families.family.Binomial()
                log = smf.glm(self._out_model, self.df, family=f).fit()

            if print_results:
                print('\n----------------------------------------------------------------')
                print('MODEL: ' + self._out_model)
                print('-----------------------------------------------------------------')
                print(log.summary())

            # Step 2) Estimation under the scenarios
            self.QAW = log.predict(self.df)
            dfx = self.df.copy()
            dfx[self._exposure] = 1
            self.QA1W = log.predict(dfx)
            dfx = self.df.copy()
            dfx[self._exposure] = 0
            self.QA0W = log.predict(dfx)

        # User-specified model
        else:
            data = patsy.dmatrix(model + ' - 1', self.df)
            try:
                fm = custom_model.fit(X=data, y=self.df[self._outcome])
            except TypeError:
                raise TypeError("Currently custom_model must have the 'fit' function with arguments 'X', 'y'. This "
                                "covers both sklearn and supylearner. If there is a predictive model you would "
                                "like to use, please open an issue at https://github.com/pzivich/zepid and I "
                                "can work on adding support")
            if print_results and hasattr(fm, 'summarize'):
                fm.summarize()

            # Continuous outcomes should only support `predict` function
            if self._continuous_outcome:
                if hasattr(fm, 'predict'):
                    self.QAW = fm.predict(data)

                    dfx = self.df.copy()
                    dfx[self._exposure] = 1
                    data = patsy.dmatrix(model + ' - 1', dfx)
                    self.QA1W = fm.predict(data)

                    dfx = self.df.copy()
                    dfx[self._exposure] = 0
                    data = patsy.dmatrix(model + ' - 1', dfx)
                    self.QA0W = fm.predict(data)
                else:
                    raise ValueError("Currently custom_model must have 'predict' or 'predict_proba' attribute")

            # Binary outcomes can support either the `predict` or `predict_proba`
            else:
                if hasattr(fm, 'predict_proba'):
                    self.QAW = fm.predict_proba(data)[:, 1]

                    dfx = self.df.copy()
                    dfx[self._exposure] = 1
                    data = patsy.dmatrix(model + ' - 1', dfx)
                    self.QA1W = fm.predict_proba(data)[:, 1]

                    dfx = self.df.copy()
                    dfx[self._exposure] = 0
                    data = patsy.dmatrix(model + ' - 1', dfx)
                    self.QA0W = fm.predict_proba(data)[:, 1]

                elif hasattr(fm, 'predict'):
                    self.QAW = fm.predict(data)

                    dfx = self.df.copy()
                    dfx[self._exposure] = 1
                    data = patsy.dmatrix(model + ' - 1', dfx)
                    self.QA1W = fm.predict(data)

                    dfx = self.df.copy()
                    dfx[self._exposure] = 0
                    data = patsy.dmatrix(model + ' - 1', dfx)
                    self.QA0W = fm.predict(data)

                else:
                    raise ValueError("Currently custom_model must have 'predict' or 'predict_proba' attribute")

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

        # Step 4) Calculating clever covariate (HAW)
        H1W = self.df[self._exposure] / self.g1W
        H0W = -(1 - self.df[self._exposure]) / self.g0W
        HAW = H1W + H0W

        # Step 5) Estimating TMLE
        f = sm.families.family.Binomial()
        y = self.df[self._outcome]
        log = sm.GLM(y, np.column_stack((H1W, H0W)), offset=np.log(probability_to_odds(self.QAW)),
                     family=f).fit()
        self._epsilon = log.params
        Qstar1 = logistic.cdf(np.log(probability_to_odds(self.QA1W)) + self._epsilon[0] * 1/self.g1W)
        Qstar0 = logistic.cdf(np.log(probability_to_odds(self.QA0W)) + self._epsilon[1] * -1/self.g0W)
        Qstar = log.predict(np.column_stack((H1W, H0W)), offset=np.log(probability_to_odds(self.QAW)))

        # Step 6) Calculating Psi
        if self.alpha == 0.05:  # Without this, won't match R exactly. R relies on 1.96, while I use SciPy
            zalpha = 1.96
        else:
            zalpha = norm.ppf(1 - self.alpha / 2, loc=0, scale=1)

        # p-values are not implemented (doing my part to enforce CL over p-values)
        if self._continuous_outcome:
            # Calculating Average Treatement Effect
            Qstar = self._unit_unbound(Qstar, mini=self._continuous_min, maxi=self._continuous_max)
            Qstar1 = self._unit_unbound(Qstar1, mini=self._continuous_min, maxi=self._continuous_max)
            Qstar0 = self._unit_unbound(Qstar0, mini=self._continuous_min, maxi=self._continuous_max)

            self.average_treatment_effect = np.mean(Qstar1 - Qstar0)
            # Influence Curve for CL
            y_unbound = self._unit_unbound(self.df[self._outcome], mini=self._continuous_min, maxi=self._continuous_max)
            ic = HAW * (y_unbound - Qstar) + (Qstar1 - Qstar0) - self.average_treatment_effect
            varIC = np.var(ic, ddof=1) / self.df.shape[0]
            self.average_treatment_effect_ic = [self.average_treatment_effect - zalpha * math.sqrt(varIC),
                                                self.average_treatment_effect + zalpha * math.sqrt(varIC)]
        else:
            # Calculating Risk Difference
            self.risk_difference = np.mean(Qstar1 - Qstar0)
            # Influence Curve for CL
            ic = HAW * (self.df[self._outcome] - Qstar) + (Qstar1 - Qstar0) - self.risk_difference
            varIC = np.var(ic, ddof=1) / self.df.shape[0]
            self.risk_difference_ci = [self.risk_difference - zalpha * math.sqrt(varIC),
                                       self.risk_difference + zalpha * math.sqrt(varIC)]

            # Calculating Risk Ratio
            self.risk_ratio = np.mean(Qstar1) / np.mean(Qstar0)
            # Influence Curve for CL
            ic = (1/np.mean(Qstar1)*(H1W * (self.df[self._outcome] - Qstar) + Qstar1 - np.mean(Qstar1)) -
                  (1/np.mean(Qstar0))*(-1*H0W * (self.df[self._outcome] - Qstar) + Qstar0 - np.mean(Qstar0)))
            varIC = np.var(ic, ddof=1) / self.df.shape[0]
            self.risk_ratio_ci = [np.exp(np.log(self.risk_ratio) - zalpha * math.sqrt(varIC)),
                                  np.exp(np.log(self.risk_ratio) + zalpha * math.sqrt(varIC))]

            # Calculating Odds Ratio
            self.odds_ratio = (np.mean(Qstar1) / (1 - np.mean(Qstar1))) / (np.mean(Qstar0) / (1 - np.mean(Qstar0)))
            # Influence Curve for CL
            ic = ((1/(np.mean(Qstar1)*(1-np.mean(Qstar1))) * (H1W*(self.df[self._outcome] - Qstar) + Qstar1)) -
                  (1/(np.mean(Qstar0)*(1 - np.mean(Qstar0))) * (-1*H0W*(self.df[self._outcome] - Qstar) + Qstar0)))
            seIC = math.sqrt(np.var(ic, ddof=1) / self.df.shape[0])
            self.odds_ratio_ci = [np.exp(np.log(self.odds_ratio) - zalpha * seIC),
                                  np.exp(np.log(self.odds_ratio) + zalpha * seIC)]

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
        v = np.where(v < bound, bound, v)
        v = np.where(v > 1-bound, 1-bound, v)
        return v

    @staticmethod
    def _unit_unbound(ystar, mini, maxi):
        return ystar*(maxi - mini) + mini
