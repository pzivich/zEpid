import numpy as np
import pandas as pd
from zepid.causal.ipw.utils import propensity_score

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import links


class IPSW:
    def __init__(self, df, exposure, outcome, selection, generalize=True, stabilized=True, weights=None):
        """Calculate inverse probability of sampling weights through logistic regression.
        """
        self.df = df.copy()
        self.sample = df.loc[df[selection] == 1].copy()
        self.target = df.loc[df[selection] == 0].copy()

        self.generalize = generalize  # important for how weights constructed
        # IPSW for generalizability
        # IOSW for transportability
        self.stabilized = stabilized

        self.exposure = exposure
        self.outcome = outcome
        self.selection = selection
        self.weight = weights

        self.risk_difference = None
        self.risk_ratio = None
        self.Weight = None
        self._denominator_model = False

    def regression_models(self, model_denominator, model_numerator='1', print_results=True):
        """
        """
        if not self.stabilized:
            if model_numerator != '1':
                raise ValueError('Argument for model_numerator is only used for stabilized=True')

        dmodel = propensity_score(self.df, self.selection + ' ~ ' + model_denominator, print_results=print_results)

        self.sample['__denom__'] = dmodel.predict(self.sample)
        self._denominator_model = True

        # Stabilization factor if valid
        if self.stabilized:
            nmodel = propensity_score(self.df, 'Q('+self.selection + ') ~ ' + model_numerator, print_results=print_results)
            self.sample['__numer__'] = nmodel.predict(self.sample)
        else:
            self.sample['__numer__'] = 1

        # Calculate IPSW (generalizability)
        if self.generalize:
            self.sample['__ipsw__'] = self.sample['__numer__'] / self.sample['__denom__']

        # Calculate IOSW (transportability)
        else:
            if self.stabilized:
                self.sample['__ipsw__'] = (((1 - self.sample['__denom__']) / self.sample['__denom__']) *
                                           (self.sample['__numer__'] / (1 - self.sample['__numer__'])))
            else:
                self.sample['__ipsw__'] = (1 - self.sample['__denom__']) / self.sample['__denom__']

        self.Weight = self.sample['__ipsw__']

    def fit(self):
        """
        """
        if self.weight is not None:
            self.sample['__ipw__'] = self.Weight * self.sample[self.weight]
        else:
            self.sample['__ipw__'] = self.Weight

        exp = self.sample[self.sample[self.exposure] == 1].copy()
        uxp = self.sample[self.sample[self.exposure] == 0].copy()

        r1 = np.average(exp[self.outcome], weights=exp['__ipw__'])
        r0 = np.average(uxp[self.outcome], weights=uxp['__ipw__'])

        self.risk_difference = r1 - r0
        self.risk_ratio = r1 / r0

    def summary(self, decimal=3):
        """
        """
        print('----------------------------------------------------------------------')
        print('Risk Difference: ', round(float(self.risk_difference), decimal))
        print('Risk Ratio: ', round(float(self.risk_ratio), decimal))
        print('----------------------------------------------------------------------')
