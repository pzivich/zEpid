import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from zepid.causal.ipw.utils import propensity_score


class IPSW:
    r"""Calculate inverse probability of sampling weights through logistic regression. Inverse probability of
    sampling weights are an extension of inverse probability weights to allow for the generalizability or the
    transportability of results.

    For generalizability, inverse probability of sampling weights take the following form

    .. math::

        IPSW = \frac{1}{\Pr(S=1|W)}

    where `W` is all the factors related to the sample selection process

    For transportability, the inverse probability of sampling weights are actually inverse odds of sampling weights.
    They take the following form

    .. math::

        IPSW = \frac{\Pr(S=0|W}{\Pr(S=1|W)}

    Confidence intervals should be obtained by using a non-parametric bootstrapping procedure

    Parameters
    ----------
    df : DataFrame
        Pandas dataframe containing all variables required for generalization/transportation. Should include
        all features related to sample selection, indicator for selection into the sample, and treatment/outcome
        information for the sample (selection == 1)
    exposure : str
        Column label for exposure/treatment of interest. Can be nan for all those not in sample
    outcome : str
        Column label for outcome of interest. Can be nan for all those not in sample
    selection : str
        Column label for indicator of selection into the sample. Should be 1 if individual comes from the study
        sample and 0 if individual is from random sample of source population
    generalize : bool, optional
        Whether the problem is a generalizability (True) problem or a transportability (False) problem. See notes
        for further details on the difference between the two estimation methods
    stabilized : bool, optional
        Whether to return stabilized or unstabilized weights. Default is stabilized weights (True)
    weights : None, str, optional
        For conditionally randomized trials, or observational research, inverse probability of treatment weights
        can be used to adjust for confounding. Before estimating the effect measures, this weight vector and the
        IPSW are multiplied to calculate new weights
        When weights is None, the data is assumed to come from a randomized trial, and does not need to be adjusted

    Notes
    -----
    There are two related concepts; generalizability and transportability. Generalizability is when your study
    sample is part of your target population. For example, you want to generalize results from California to the
    entire United States. Transportability is when your study sample is not part of your target population. For
    example, we want to apply our results from California to Canada. Depending on the scenario, the IPSW have
    slightly different forms. `IPSW` allows for both of these problems

    Examples
    --------
    Setting up the environment

    >>>from zepid import load_generalize_data
    >>>from zepid.causal.generalize import IPSW
    >>>df = load_generalize_data(False)

    Generalizability for RCT results

    >>>ipsw = IPSW(df, exposure='A', outcome='Y', selection='S', generalize=True)
    >>>ipsw.regression_models('L + W + L:W', print_results=False)
    >>>ipsw.fit()
    >>>ipsw.summary()

    Transportability for RCT results

    >>>ipsw = IPSW(df, exposure='A', outcome='Y', selection='S', generalize=False)
    >>>ipsw.regression_models('L + W + L:W', print_results=False)
    >>>ipsw.fit()
    >>>ipsw.summary()

    For observational studies, IPTW can be used to account for confounders

    >>>from zepid.causal.ipw import IPTW
    >>>iptw = IPTW(df, treatment='A')
    >>>iptw.regression_models('L')
    >>>iptw.fit()
    >>>df['iptw'] = iptw.Weight

    >>>ipsw = IPSW(df, exposure='A', outcome='Y', selection='S', weights='iptw', generalize=False)
    >>>ipsw.regression_models('L + W + L:W', print_results=False)
    >>>ipsw.fit()
    >>>ipsw.summary()

    References
    ----------
    Lesko CR, Buchanan AL, Westreich D, Edwards JK, Hudgens MG, & Cole SR. (2017).
    Generalizing study results: a potential outcomes perspective. Epidemiology (Cambridge, Mass.), 28(4), 553.

    Westreich D, Edwards JK, Lesko CR, Stuart E, & Cole SR. (2017). Transportability of trial
    results using inverse odds of sampling weights. AJE, 186(8), 1010-1014.

    Dahabreh IJ, Robertson SE, Stuart EA, Hernan MA (2018). Transporting inferences from a
    randomized trial to a new target population. arXiv preprint arXiv:1805.00550.
    """

    def __init__(self, df, exposure, outcome, selection, generalize=True, stabilized=True, weights=None):
        self.df = df.copy()
        self.sample = df.loc[df[selection] == 1].copy()
        self.target = df.loc[df[selection] == 0].copy()

        self.generalize = generalize  # determines whether IPSW or IOSW are calculated
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
        """Logistic regression model(s) for estimating weights. The model denominator must be specified for both
        stabilized and unstabilized weights. The optional argument 'model_numerator' allows specification of the
        stabilization factor for the weight numerator. By default model results are returned

        Parameters
        ----------
        model_denominator : str
            String listing variables to predict the exposure, separated by +. For example, 'var1 + var2 + var3'. This
            is for the predicted probabilities of the denominator
        model_numerator : str, optional
            Optional string listing variables to predict the selection separated by +. Only used to calculate the
            numerator. Default ('1') calculates the overall probability of selection. In general, this is recommended.
            Adding in other variables means they are no longer accounted for in estimation of IPSW. Argument is also
            only used when calculating stabilized weights
        print_results : bool, optional
            Whether to print the model results from the regression models. Default is True
        """
        if not self.stabilized:
            if model_numerator != '1':
                raise ValueError('Argument for model_numerator is only used for stabilized=True')

        dmodel = propensity_score(self.df, self.selection + ' ~ ' + model_denominator, print_results=print_results)

        self.sample['__denom__'] = dmodel.predict(self.sample)
        self._denominator_model = True

        # Stabilization factor if valid
        if self.stabilized:
            nmodel = propensity_score(self.df, self.selection + ' ~ ' + model_numerator, print_results=print_results)
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
        """Uses the calculated IPSW to obtain the risk difference and risk ratio from the sample. If weights are
        provided in the initial step, those weights are multiplied with IPSW to obtain the overall weights.

        The risk for the exposed and unexposed are calculated by taking the weighted averages.

        Returns
        -------
        `IPSW` gains `risk_difference` and `risk_ratio` which are the generalized risk difference and risk ratios for
        the exposure-outcome relationship based on the data and IPSW model
        """
        if not self._denominator_model:
            raise ValueError('The regression_models() function must be specified before effect measure estimation')
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

    def summary(self, decimal=4):
        """Prints a summary of the results for the IPSW estimator

        Parameters
        ----------
        decimal : int, optional
            Number of decimal places to display in the result
        """
        print('======================================================================')
        if self.generalize:
            print('           Inverse Probability of Sampling Weights')
        else:
            print('               Inverse Odds of Sampling Weights')
        print('======================================================================')
        print('Risk Difference: ', round(float(self.risk_difference), decimal))
        print('----------------------------------------------------------------------')
        print('Risk Ratio: ', round(float(self.risk_ratio), decimal))
        print('======================================================================')


class GTransportFormula:
    r"""Calculate the g-transport-formula using a observed study sample and a sample from the target population.
    Broadly, the process for fitting the g-transport-formula is similar to the g-formula (as implemented in
    `TimeFixedGFormula`). Instead of predicting the potential outcomes of only the sample, the g-transport-formula
    predicts potential outcomes for the full target population

    For generalizability, we first fit a Q-model predicting the outcome as a function of the treatment and any
    modifiers (along with confounders if in observation data). Afterwards, we predict the potential outcomes for
    the entire population (S=1 and S=0). To obtain the marginal effect measure, we take the mean of the entire
    population (S=1 and S=0)

    For transportability, we similarly fit a Q-model in the observed sample and generate predictions for the entire
    sample. However, for transportability our sample is not part of the target population. Therefore, we only take
    the marginal of the S=0 group.

    Confidence intervals should be obtained by using a non-parametric bootstrapping procedure

    Parameters
    ----------
    df : DataFrame
        Pandas dataframe containing all variables required for generalization/transportation. Should include
        all features related to sample selection, indicator for selection into the sample, and treatment/outcome
        information for the sample (selection == 1)
    exposure : str
        Column label for exposure/treatment of interest. Can be nan for all those not in sample. Only binary
        exposures are currently supported
    outcome : str
        Column label for outcome of interest. Can be nan for all those not in sample
    selection : str
        Column label for indicator of selection into the sample. Should be 1 if individual comes from the study
        sample and 0 if individual is from random sample of source population
    outcome_type : str, optional
        Outcome variable type. Currently only 'binary', 'normal', and 'poisson variable types are supported
    generalize : bool, optional
        Whether the problem is a generalizability (True) problem or a transportability (False) problem. See notes
        for further details on the difference between the two estimation methods
    weights : None, str, optional
        If there are associated sampling (or other types of) weights, these can be included in this statement.
        Sampling weights may be used for a transportability problem? Either way, I would like to keep this as an
        option (to mirror TimeFixedGFormula)

    Notes
    -----
    There are two related concepts; generalizability and transportability. Generalizability is when your study
    sample is part of your target population. For example, you want to generalize results from California to the
    entire United States. Transportability is when your study sample is not part of your target population. For
    example, we want to apply our results from California to Canada. Depending on the scenario, how the marginal
    risk difference is calculated is slightly different. `GTransportFormula` allows for both of these problems

    Examples
    --------
    Setting up the environment

    >>>from zepid import load_generalize_data
    >>>from zepid.causal.generalize import GTransportFormula
    >>>df = load_generalize_data(False)

    Generalizability

    >>>gtf = GTransportFormula(df, exposure='A', outcome='Y', selection='S', generalize=True)
    >>>gtf.outcome_model('A + L + L:A + W + W:A + W:A:L')
    >>>gtf.fit()
    >>>gtf.summary()

    Transportability

    >>>gtf = GTransportFormula(df, exposure='A', outcome='Y', selection='S', generalize=False)
    >>>gtf.outcome_model('A + L + L:A + W + W:A + W:A:L')
    >>>gtf.fit()
    >>>gtf.summary()

    For observational studies, confounders should be included in the Q-model

    References
    ----------
    Lesko CR, Buchanan AL, Westreich D, Edwards JK, Hudgens MG, & Cole SR. (2017).
    Generalizing study results: a potential outcomes perspective. Epidemiology (Cambridge, Mass.), 28(4), 553.

    Dahabreh IJ, Robertson SE, Stuart EA, Hernan MA (2018). Transporting inferences from a
    randomized trial to a new target population. arXiv preprint arXiv:1805.00550.
    """

    def __init__(self, df, exposure, outcome, selection, outcome_type='binary', generalize=True, weights=None):
        self.df = df.copy()
        self.sample = df.loc[df[selection] == 1].copy()
        self.target = df.loc[df[selection] == 0].copy()

        self.generalize = generalize  # determines the marginal calculation

        self.exposure = exposure
        self.outcome = outcome
        self.outcome_type = outcome_type
        self.selection = selection
        self.weight = weights

        self.risk_difference = None
        self.risk_ratio = None
        self.Weight = None
        self._outcome_model = False

    def outcome_model(self, model, print_results=True):
        """Build the model for the outcome. This is also referred to at the Q-model. This must be specified
        before the fit function. If it is not, an error will be raised.

        Parameters
        ----------
        model : str
            Variables to include in the model for predicting the outcome. Must be contained within the input
            pandas dataframe when initialized. Model form should contain the exposure, i.e. 'art + age + male'
        print_results : bool, optional
            Whether to print the logistic regression results to the terminal. Default is True
        """
        if self.outcome_type == 'binary':
            linkdist = sm.families.family.Binomial()
        elif self.outcome_type == 'normal':
            linkdist = sm.families.family.Gaussian()
        elif self.outcome_type == 'poisson':
            linkdist = sm.families.family.Poisson()
        else:
            raise ValueError("Only 'binary', 'normal', and 'poisson' distributed outcomes are available")

        # Modeling the outcome
        if self.weight is None:
            m = smf.glm(self.outcome+' ~ '+model, self.sample, family=linkdist)
            self._outcome_model = m.fit()
        else:
            m = smf.glm(self.outcome+' ~ '+model, self.sample, family=linkdist,
                        freq_weights=self.sample[self.weight])
            self._outcome_model = m.fit()

        # Printing results of the model and if any observations were dropped
        if print_results:
            print(self._outcome_model.summary())

    def fit(self):
        """Uses the g-transport formula to obtain the risk difference and risk ratio from the sample.

        Returns
        -------
        `GTransportFormula` gains `risk_difference` and `risk_ratio` which are the generalized risk difference and
        risk ratios for the exposure-outcome relationship
        """
        if not self._outcome_model:
            raise ValueError('The outcome_model() function must be specified before effect measure estimation')

        # Generalizability problem
        if self.generalize:
            dfa = self.df.copy()
            dfn = self.df.copy()
            dfa[self.exposure] = 1
            dfn[self.exposure] = 0

            ya = self._outcome_model.predict(dfa)
            yn = self._outcome_model.predict(dfn)

            if self.weight is not None:
                r1 = np.average(ya, weights=self.df[self.weight])
                r0 = np.average(yn, weights=self.df[self.weight])
            else:
                r1 = np.mean(ya)
                r0 = np.mean(yn)

        # Tranportability problem
        else:
            dfa = self.target.copy()
            dfn = self.target.copy()
            dfa[self.exposure] = 1
            dfn[self.exposure] = 0

            ya = self._outcome_model.predict(dfa)
            yn = self._outcome_model.predict(dfn)

            if self.weight is not None:
                r1 = np.average(ya, weights=self.df[self.weight])
                r0 = np.average(yn, weights=self.df[self.weight])
            else:
                r1 = np.mean(ya)
                r0 = np.mean(yn)

        self.risk_difference = r1 - r0
        self.risk_ratio = r1 / r0

    def summary(self, decimal=4):
        """Prints a summary of the results for the g-transport estimator

        Parameters
        ----------
        decimal : int, optional
            Number of decimal places to display in the result
        """
        print('======================================================================')
        print('                       g-Transport formula')
        print('======================================================================')
        print('Risk Difference: ', round(float(self.risk_difference), decimal))
        print('----------------------------------------------------------------------')
        print('Risk Ratio: ', round(float(self.risk_ratio), decimal))
        print('======================================================================')


class AIPSW:
    r"""Doubly robust estimator for generalizability. I haven't found a good name for it in the literature yet, so I
    am naming it augmented-IPSW (in honor of other doubly robust estimators like AIPTW and AIPMW).

    The process of estimating AIPSW follows other doubly robust estimators. We need to specify both the IPSW model
    and the g-transport model. From this information, the AIPSW is calculated via the following

    .. math::

        \psi = \frac{1}{n} \sum \left(E[Y|A=a,L,S=1] + \frac{I(S=1, A=a)}{\Pr(S=1|L)} (Y - E[Y|A=a,L,S=1])\right)

    For transportability problems, AIPSW takes the following form

    .. math::

        \psi = \frac{\sum IPSW\times I(S=1, A=a)(Y - E[Y|A=a,L,S=1]) + (1-S)E[Y|A=a,L,S=1]}{\Pr(S=0)}

    For generalizability, we first fit a Q-model predicting the outcome as a function of the treatment and any
    modifiers (along with confounders if in observation data). Next we calculate IPSW (with IPTW if there is any
    confounders). Afterwards, we predict the potential outcomes for the entire population (S=1 and S=0). We then
    use the above formula to calculate the marginal effect

    A similar process is done for transportability. Instead we merge g-transport and inverse odds of sampling weights

    Confidence intervals should be obtained by using a non-parametric bootstrapping procedure

    Parameters
    ----------
    df : DataFrame
        Pandas dataframe containing all variables required for generalization/transportation. Should include
        all features related to sample selection, indicator for selection into the sample, and treatment/outcome
        information for the sample (selection == 1)
    exposure : str
        Column label for exposure/treatment of interest. Can be nan for all those not in sample. Only binary
        exposures are currently supported
    outcome : str
        Column label for outcome of interest. Can be nan for all those not in sample
    selection : str
        Column label for indicator of selection into the sample. Should be 1 if individual comes from the study
        sample and 0 if individual is from random sample of source population
    generalize : bool, optional
        Whether the problem is a generalizability (True) problem or a transportability (False) problem. See notes
        for further details on the difference between the two estimation methods
    weights : None, str, optional
        For conditionally randomized trials, or observational research, inverse probability of treatment weights
        can be used to adjust for confounding in IPSW. Before estimating the effect measures, this weight vector
        and the IPSW are multiplied to calculate new weights
        When weights is None, the data is assumed to come from a randomized trial, and does not need to be adjusted

    Notes
    -----
    There are two related concepts; generalizability and transportability. Generalizability is when your study
    sample is part of your target population. For example, you want to generalize results from California to the
    entire United States. Transportability is when your study sample is not part of your target population. For
    example, we want to apply our results from California to Canada. Depending on the scenario, how the marginal
    risk difference is calculated is slightly different. `GTransportFormula` allows for both of these problems

    Examples
    --------
    Setting up the environment

    >>>from zepid import load_generalize_data
    >>>from zepid.causal.generalize import AIPSW
    >>>df = load_generalize_data(False)

    Generalizability

    >>>aipw = AIPSW(df, exposure='A', outcome='Y', selection='S', generalize=True)
    >>>aipw.weight_model('L + W_sq', print_results=False)
    >>>aipw.outcome_model('A + L + L:A + W + W:A + W:A:L', print_results=False)
    >>>aipw.fit()
    >>>aipw.summary()

    Transportability

    >>>aipw = AIPSW(df, exposure='A', outcome='Y', selection='S', generalize=False)
    >>>aipw.weight_model('L + W_sq', print_results=False)
    >>>aipw.outcome_model('A + L + L:A + W + W:A + W:A:L', print_results=False)
    >>>aipw.fit()
    >>>aipw.summary()

    References
    ----------
    Dahabreh IJ, Robertson SE, Stuart EA, Hernan MA (2018). Transporting inferences from a
    randomized trial to a new target population. arXiv preprint arXiv:1805.00550.

    Dahabreh IJ, Hernan MA, Robertson SE, Buchanan A, Steingrimsson JA. (2019). Generalizing
    trial findings in nested trial designs with sub-sampling of non-randomized individuals. arXiv preprint
    arXiv:1902.06080.
    """

    def __init__(self, df, exposure, outcome, selection, generalize=True, weights=None):
        self.df = df.copy()
        self.sample = df[selection] == 1
        self.target = df[selection] == 0

        self.generalize = generalize  # determines whether IPSW or IOSW are calculated

        self.exposure = exposure
        self.outcome = outcome
        self.selection = selection
        self.weight = weights

        self.Weight = None
        self._denominator_model = False
        self._outcome_model = False
        self._YA1 = None
        self._YA0 = None

        self.risk_difference = None
        self.risk_ratio = None

    def weight_model(self, model_denominator, model_numerator='1', stabilized=True, print_results=True):
        """Logistic regression model(s) for estimating IPSW. The model denominator must be specified for both
        stabilized and unstabilized weights. The optional argument 'model_numerator' allows specification of the
        stabilization factor for the weight numerator. By default model results are returned

        Parameters
        ----------
        model_denominator : str
            String listing variables to predict the exposure, separated by +. For example, 'var1 + var2 + var3'. This
            is for the predicted probabilities of the denominator
        model_numerator : str, optional
            Optional string listing variables to predict the selection separated by +. Only used to calculate the
            numerator. Default ('1') calculates the overall probability of selection. In general, this is recommended.
            Adding in other variables means they are no longer accounted for in estimation of IPSW. Argument is also
            only used when calculating stabilized weights
        stabilized : bool, optional
            Whether to generated stabilized IPSW. Default is True, which returns the stabilized IPSW
        print_results : bool, optional
            Whether to print the model results from the regression models. Default is True
        """
        dmodel = propensity_score(self.df, self.selection + ' ~ ' + model_denominator, print_results=print_results)

        self.df['__denom__'] = dmodel.predict(self.df)
        self._denominator_model = True

        # Stabilization factor if valid
        if stabilized:
            nmodel = propensity_score(self.df, self.selection + ' ~ ' + model_numerator, print_results=print_results)
            self.df['__numer__'] = np.where(self.sample, nmodel.predict(self.df), 0)
        else:
            self.df['__numer__'] = np.where(self.sample, 1, 0)

        # Calculate IPSW (generalizability)
        if self.generalize:
            self.df['__ipsw__'] = self.df['__numer__'] / self.df['__denom__']

        # Calculate IOSW (transportability)
        else:
            if stabilized:
                self.df['__ipsw__'] = (((1 - self.df['__denom__']) / self.df['__denom__']) *
                                           (self.df['__numer__'] / (1 - self.df['__numer__'])))
            else:
                self.df['__ipsw__'] = (1 - self.df['__denom__']) / self.df['__denom__']

        if self.weight is not None:
            self.Weight = self.df['__ipsw__'] * self.df[self.weight]
        else:
            self.Weight = self.df['__ipsw__']

    def outcome_model(self, model, outcome_type='binary', print_results=True):
        """Build the g-transport model for the outcome. This is also referred to at the Q-model.

        Parameters
        ----------
        model : str
            Variables to include in the model for predicting the outcome. Must be contained within the input
            pandas dataframe when initialized. Model form should contain the exposure, i.e. 'art + age + male'
        outcome_type : str, optional
            Variable type for the outcome. Default is binary. Also supports 'normal' and 'poisson' distributed outcomes
        print_results : bool, optional
            Whether to print the logistic regression results to the terminal. Default is True
        """
        if outcome_type == 'binary':
            linkdist = sm.families.family.Binomial()
        elif outcome_type == 'normal':
            linkdist = sm.families.family.Gaussian()
        elif outcome_type == 'poisson':
            linkdist = sm.families.family.Poisson()
        else:
            raise ValueError("Only 'binary', 'normal', and 'poisson' distributed outcomes are available")

        # Modeling the outcome
        df = self.df[self.sample].copy()
        m = smf.glm(self.outcome+' ~ '+model, df, family=linkdist)
        self._outcome_model = m.fit()

        # Printing results of the model and if any observations were dropped
        if print_results:
            print(self._outcome_model.summary())

        dfa = self.df.copy()
        dfn = self.df.copy()
        dfa[self.exposure] = 1
        dfn[self.exposure] = 0

        self._YA1 = self._outcome_model.predict(dfa)
        self._YA0 = self._outcome_model.predict(dfn)

    def fit(self):
        """Uses AIPSW formula to obtain the risk difference and risk ratio from the sample.

        Returns
        -------
        `AIPSW` gains `risk_difference` and `risk_ratio` which are the generalized risk difference and risk ratios for
        the exposure-outcome relationship based on the data and IPSW model
        """
        if not self._denominator_model:
            raise ValueError('The weight_model() function must be specified before effect measure estimation')
        if not self._outcome_model:
            raise ValueError('The outcome_model() function must be specified before effect measure estimation')

        # Generalizability problem
        if self.generalize:
            part1 = self._YA1
            part2 = np.where(self.sample & (self.df[self.exposure] == 1),
                             self.Weight * (self.df[self.outcome] - self._YA1), 0)
            r1 = np.mean(part1 + part2)

            part1 = self._YA0
            part2 = np.where(self.sample & (self.df[self.exposure] == 0),
                             self.Weight * (self.df[self.outcome] - self._YA0), 0)
            r0 = np.mean(part1 + part2)

        # Transportability problem
        else:
            part1 = np.where(self.sample & (self.df[self.exposure] == 1),
                             self.Weight * (self.df[self.outcome] - self._YA1), 0)
            part2 = (1 - self.df[self.selection]) * self._YA1
            r1 = np.sum(part1 + part2) / np.sum(1 - self.df[self.selection])

            part1 = np.where(self.sample & (self.df[self.exposure] == 0),
                             self.Weight * (self.df[self.outcome] - self._YA0), 0)
            part2 = (1 - self.df[self.selection]) * self._YA0
            r0 = np.sum(part1 + part2) / np.sum(1 - self.df[self.selection])

        self.risk_difference = r1 - r0
        self.risk_ratio = r1 / r0

    def summary(self, decimal=4):
        """Prints a summary of the results for the AIPSW estimator

        Parameters
        ----------
        decimal : int, optional
            Number of decimal places to display in the result
        """
        print('======================================================================')
        print('           Augmented Inverse Probability of Sampling Weights')
        print('======================================================================')
        print('Risk Difference: ', round(float(self.risk_difference), decimal))
        print('----------------------------------------------------------------------')
        print('Risk Ratio: ', round(float(self.risk_ratio), decimal))
        print('======================================================================')
