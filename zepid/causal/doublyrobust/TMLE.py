# Only be very general (no SuPyLearner or any ML yet)

import math
import warnings
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import logistic, norm
from statsmodels.genmod.families import links

from zepid.causal.ipw import propensity_score

# TODO add variable selection algorithm
# TODO add option of ML algorithms to generate predictions


class TMLE:
    """
    Implementation of a simple TMLE model. It uses standard logistic regression models to calculate Psi. In the future,
    the addition of automatic variable/model selection and addition of machine learning models will be added.

    This is only the base TMLE currently
    """
    def __init__(self, df, exposure, outcome, psi='risk difference', alpha = 0.05):
        """

        df:
            -pandas dataframe containing the variables of interest
        exposure:
            -column label for the exposure of interest
        outcome:
            -column label for the outcome of interest
        psi:
            -What the TMLE psi estimates. Currently only Risk Difference is supported
        alpha:
            -alpha for confidence interval level. Default is 0.05
        """
        if df.dropna().shape[0] != df.shape[0]:
            warnings.warn("There is missing data in the dataset. By default, TMLE will drop all missing data. TMLE will"
                          "fit "+str(df.dropna().shape[0])+' of '+str(df.shape[0])+' observations')
        if psi != "risk difference":
            raise ValueError('Only the additive estimate of the risk difference is currently implemented')
        self._psi_correspond = psi
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
        self.gA1 = None
        self._epsilon = None
        self.psi = None
        self.confint = None

    def exposure_model(self, model, print_model_results=True):
        """Estimation of g(A=1,W), which is Pr(A=1|W)

        model:
            -Independent variables to predict the exposure. Example) 'var1 + var2 + var3'
        print_model_results:
            -Whether to print the fitted model results. Default is True (prints results)
        """
        self._exp_model = self._exposure + ' ~ ' + model
        fitmodel = propensity_score(self.df, self._exp_model, mresult=print_model_results)
        self.gA1 = fitmodel.predict(self.df)
        self._fit_exposure_model = True

    def outcome_model(self, model, print_model_results=True):
        """Estimation of Q(A,W), which is Pr(Y=1|A,W)

        model:
            -Independent variables to predict the exposure. Example) 'var1 + var2 + var3'
        print_model_results:
            -Whether to print the fitted model results. Default is True (prints results)
        """
        self._out_model = self._outcome + ' ~ ' + model
        f = sm.families.family.Binomial(sm.families.links.logit)
        log = smf.glm(self._out_model, self.df, family=f).fit()

        if print_model_results:
            print('\n----------------------------------------------------------------')
            print('MODEL: ' + self._out_model)
            print('-----------------------------------------------------------------')
            print(log.summary())

        self.QAW = log.predict(self.df)

        dfx = self.df.copy()
        dfx[self._exposure] = 1
        self.QA1W = log.predict(dfx)
        dfx = self.df.copy()
        dfx[self._exposure] = 0
        self.QA0W = log.predict(dfx)
        self._fit_outcome_model = True

    def fit(self):
        """Estimates psi based on the gAW and QAW. Confidence intervals come from the influence curve
        """
        if (self._fit_exposure_model is False) or (self._fit_exposure_model is False):
            raise ValueError('The exposure and outcome models must be specified before the psi estimate can '
                             'be generated')

        # Calculating items to through into regression
        gAW = self.df[self._exposure] / self.gA1 - ((1 - self.df[self._exposure]) / (1 - self.gA1))
        g1W = 1 / self.gA1
        g0W = -1 / (1 - self.gA1)

        # Fitting logistic model with QAW offset
        f = sm.families.family.Binomial(sm.families.links.logit)
        log = sm.GLM(self.df[self._outcome], gAW, offset=self.QAW, family=f).fit()
        self._epsilon = log.params['x1']

        # Getting Qn*
        # Qstar = logistic.cdf(self.QAW + self._epsilon*gAW) # I think this would allow natural course comparison
        Qstar1 = logistic.cdf(self.QA1W + self._epsilon*g1W)
        Qstar0 = logistic.cdf(self.QA0W + self._epsilon*g0W)
        self.psi = np.mean(Qstar1 - Qstar0)

        # Getting influence curve
        ic = (gAW * (self.df[self._outcome] - logistic.cdf(self.QAW)) +
              logistic.cdf(Qstar1 - Qstar0) - self.psi)
        varIC = np.var(ic) / self.df.shape[0]
        zalpha = norm.ppf(1 - self.alpha / 2, loc=0, scale=1)
        self.confint = [self.psi - zalpha * math.sqrt(varIC), self.psi + zalpha * math.sqrt(varIC)]

    def summary(self, decimal=3):
        """
        Prints summary of model results

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

