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
    def __init__(self, df, exposure, outcome, measure='risk_difference', alpha = 0.05):
        """Implementation of a single time-point TMLE model. It uses standard logistic regression models to calculate
        Psi. The TMLE estimator allows a standard logistic regression to be used. Alternatively, users are able to
        directly input predicted outcomes from other methods (like machine learning algorithms).

        df:
            -pandas dataframe containing the variables of interest
        exposure:
            -column label for the exposure of interest
        outcome:
            -column label for the outcome of interest
        measure:
            -What the TMLE psi estimates. Current options include; risk difference comparing treated to untreated
             (F(A=1) - F(A=0)), risk ratio (F(A=1) / F(A=0)), and odds ratio.
             The following keywords are used
             'risk_difference'  :   F(A=1) - F(A=0)
             'risk_ratio'       :   F(A=1) / F(A=0)
             'odds_ratio'       :   (F(A=1) / (1 - F(A=1)) / (F(A=0) / (1 - F(A=0))
        alpha:
            -alpha for confidence interval level. Default is 0.05
        """
        if df.dropna().shape[0] != df.shape[0]:
            warnings.warn("There is missing data in the dataset. By default, TMLE will drop all missing data. TMLE will"
                          "fit "+str(df.dropna().shape[0])+' of '+str(df.shape[0])+' observations')
        # Detailed steps follow "Targeted Learning" chapter 4, figure 4.2 by van der Laan, Rose
        self._psi_correspond = measure
        self.df = df.copy().dropna().reset_index()
        self.alpha = alpha
        self._exposure = exposure
        self._outcome = outcome
        self._out_model = None
        self._exp_model = None
        self._fit_exposure_model = False
        self._fit_outcome_model = False
        self.QA0W = None
        self.QA1W = None
        self.QAW = None
        self.g1W = None
        self.g0W = None
        self._epsilon = None
        self.psi = None
        self.confint = None

    def exposure_model(self, model, custom_model=None, bound=False, print_results=True):
        """Estimation of g(A=1,W), which is Pr(A=1|W)

        model:
            -Independent variables to predict the exposure. Example) 'var1 + var2 + var3'
        custom_model:
            -Input for a custom model. The model must already be estimated and have the "predict()" attribute to work.
             This allows the user to use any outside model they want and bring it into TMLE. For example, you can use
             any sklearn model, ensemble model (SuPyLearner), or just different statsmodels regression models than
             logistic regression. Please see online for an example
             NOTE: if a custom model is used, patsy in the background does the data filtering from the equation above.
             The equation order of variables MUST match that of the custom_model when it was fit. If not, this can lead
             to unexpected estimates
        bound:
            -Value between 0,1 to truncate predicted probabilities. Helps to avoid near positivity violations. Default
             is False, meaning no truncation of predicted probabilities occurs. Providing a single float assumes
             symmetric trunctation. A collection of floats can be provided for asymmetric trunctation
        print_results:
            -Whether to print the fitted model results. Default is True (prints results)
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
                fm = custom_model.fit(X=data, y=self.df[self._outcome])
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

    def outcome_model(self, model, custom_model=None, print_results=True):
        """Estimation of Q(A,W), which is Pr(Y=1|A,W)

        model:
            -Independent variables to predict the exposure. Example) 'var1 + var2 + var3'
        custom_model:
            -Input for a custom model. The model must already be estimated and have the "predict()" attribute to work.
             This allows the user to use any outside model they want and bring it into TMLE. For example, you can use
             any sklearn model, ensemble model (SuPyLearner), or just different statsmodels regression models than
             logistic regression. Please see online for an example
             NOTE: if a custom model is used, patsy in the background does the data filtering from the equation above.
             The equation order of variables MUST match that of the custom_model when it was fit. If not, this can lead
             to unexpected estimates
        print_results:
            -Whether to print the fitted model results. Default is True (prints results)
        """
        self._out_model = self._outcome + ' ~ ' + model

        # Step 1) Prediction for Q (estimation of Q-model)
        if custom_model is None:  # Logistic Regression model for predictions
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
        """Estimates psi based on the gAW and QAW. Confidence intervals come from the influence curve
        """
        if (self._fit_exposure_model is False) or (self._fit_outcome_model is False):
            raise ValueError('The exposure and outcome models must be specified before the psi estimate can '
                             'be generated')

        # Step 4) Calculating clever covariate (HAW)
        H1W = self.df[self._exposure] / self.g1W
        H0W = -(1 - self.df[self._exposure]) / (self.g0W)
        HAW = H1W + H0W

        # Step 5) Estimating TMLE
        f = sm.families.family.Binomial()
        log = sm.GLM(self.df[self._outcome], np.column_stack((H1W, H0W)), offset=np.log(probability_to_odds(self.QAW)),
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
        if self._psi_correspond == 'risk_difference':
            self.psi = np.mean(Qstar1 - Qstar0)
            # Influence Curve for CL
            ic = HAW * (self.df[self._outcome] - Qstar) + (Qstar1 - Qstar0) - self.psi
            varIC = np.var(ic, ddof=1) / self.df.shape[0]
            self.confint = [self.psi - zalpha * math.sqrt(varIC),
                            self.psi + zalpha * math.sqrt(varIC)]
            # TODO add p-values? Against my bias to add though

        elif self._psi_correspond == 'risk_ratio':
            self.psi = np.mean(Qstar1) / np.mean(Qstar0)
            # Influence Curve for CL
            ic = (1/np.mean(Qstar1)*(H1W * (self.df[self._outcome] - Qstar) + Qstar1 - np.mean(Qstar1)) -
                  (1/np.mean(Qstar0))*(-1*H0W * (self.df[self._outcome] - Qstar) + Qstar0 - np.mean(Qstar0)))
            varIC = np.var(ic, ddof=1) / self.df.shape[0]
            self.confint = [np.exp(np.log(self.psi) - zalpha * math.sqrt(varIC)),
                            np.exp(np.log(self.psi) + zalpha * math.sqrt(varIC))]

        elif self._psi_correspond == 'odds_ratio':
            self.psi = (np.mean(Qstar1) / (1 - np.mean(Qstar1))) / (np.mean(Qstar0) / (1 - np.mean(Qstar0)))
            # Influence Curve for CL
            ic = ((1/(np.mean(Qstar1)*(1-np.mean(Qstar1))) * (H1W*(self.df[self._outcome] - Qstar) + Qstar1)) -
                  (1/(np.mean(Qstar0)*(1 - np.mean(Qstar0))) * (-1*H0W*(self.df[self._outcome] - Qstar) + Qstar0)))
            seIC = math.sqrt(np.var(ic, ddof=1) / self.df.shape[0])
            self.confint = [np.exp(np.log(self.psi) - zalpha * seIC),
                            np.exp(np.log(self.psi) + zalpha * seIC)]

        else:
            raise ValueError('Specified parameter is not implemented. Please use one of the available options')

    def summary(self, decimal=3):
        """Prints summary of model results

        decimal:
            -number of decimal places to display. Default is 3
        """
        if (self._fit_exposure_model is False) or (self._fit_exposure_model is False):
            raise ValueError('The exposure and outcome models must be specified before the psi estimate can '
                             'be generated')

        print('----------------------------------------------------------------------')
        print('Psi: ', round(float(self.psi), decimal))
        print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: (' + str(round(self.confint[0], decimal)), ',',
              str(round(self.confint[1], decimal)) + ')')
        print('----------------------------------------------------------------------')
        print('Psi corresponds to '+self._psi_correspond)
        print('----------------------------------------------------------------------')

    @staticmethod
    def _bounding(v, bounds):
        """Background function to perform bounding feature.

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
                              str(bounds[0:2]) + ' will be used')
            if type(bounds[0]) is str or type(bounds[1]) is str:
                raise ValueError('Bounds must be floats between (0, 1)')
            if (bounds[0] < 0 or bounds[1] > 1) or (bounds[0] < 0 or bounds[1] > 1):
                raise ValueError('Both bound values must be between (0, 1)')
            v = np.where(v < bounds[0], bounds[0], v)
            v = np.where(v > bounds[1], bounds[1], v)
        print(np.max(v))
        return v


# TODO longitudinal TMLE; estimated by E[...E[Y_n|A=abar]...] working from center to outside
