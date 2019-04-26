import warnings
import numpy as np
import pandas as pd

from zepid.causal.utils import propensity_score


class IPMW:
    r"""Calculates inverse probability of missing weights. `IPMW` automatically codes a missingness indicator (based
    on np.nan), so data can be directly input, without creation of missingness indicator before inputting data

    The formula for stabilized IPMW is

    .. math::

        \pi_i = \frac{\Pr(M=0)}{\Pr(M=0|L=l)}

    where M=0 indicates observed data. For unstabilized IPMW

    .. math::

        \pi_i = \frac{1}{\Pr(M=0|L=l)}

    `IPMW` currently supports weights for a single missing variable, or a list of variables that are monotonically
    missing. For data to be missing monotonically, there is some ordering of the variables with missing data
    such that the previous variable must be observed for the later to be observed. A simple example is censoring in
    longitudinal data without late entry. To be observed at time *t*, the individual must be observed at time *t-1*

    For multiple variables with missing data, `IPMW` determines if the two variables are uniform missing. This is a
    special case of monotonic missing data. As a result, `IPMW` will only need to calculate IPMW for one of the
    variables. See the references for further details on this

    Parameters
    ----------
    df : DataFrame
        Pandas Dataframe object containing all variables of interest
    missing_variable : str, list
        Column name for missing data. numpy.nan values should indicate missing observations. For multiple missing
        variables, a list of strings (indicating column labels) can be added
    stabilized : bool, optional
        Whether to return the stabilized or unstabilized IPMW. Default is to return unstabilized weights
    monotone : bool, optional
        Whether missing data is monotonic or nonmonotonic. This option is only used for when multiple missing
        variables are provided. monotone=False will give an error (for now)

    Note
    ----
    Nonmonotonic missing data is arguably more common in practice. Sun and Tchetgen Tchetgen recently proposed a
    way to estimate IPMW under nonmonotonic missing data. I plan on implementing this in a future release. Until
    then `IPMW` only supports monotonic missing data

    Examples
    --------
    Setting up the environment

    >>> from zepid import load_sample_data, load_monotone_missing_data
    >>> from zepid.causal.ipw import IPMW
    >>> df = load_sample_data(timevary=False)

    Calculating unstabilized Inverse Probability of Missingness Weights

    >>> ipm = IPMW(df, missing='dead', stabilized=False)
    >>> ipm.regression_models(model_denominator='age0 + art + male')
    >>> ipm.fit()

    Extracting calculated weights

    >>> ipm.Weight

    Calculating IPMW for monotone missing variables

    >>> df = load_monotone_missing_data()
    >>> ipm = IPMW(df, missing_variable=['B', 'C'], monotone=True)
    >>> ipm.regression_models(model_denominator=['L + A', 'L + B'])
    >>> ipm.fit()
    >>> ipm.Weight

    References
    ----------
    Sun B, et al. (2017). Inverse-probability-weighted estimation for monotone and nonmonotone missing data. American
    Journal of Epidemiology, 187(3), 585-591.

    Perkins, NJ et al. (2017). Principled approaches to missing data in epidemiologic studies. American Journal of
    Epidemiology, 187(3), 568-575.

    Li L, Shen C, Li X, Robins JM. (2013). On weighting approaches for missing data. Statistical Methods
    in Medical Research, 22(1), 14-30.

    Greenland S, & Finkle WD. (1995). A critical look at methods for handling missing covariates in
    epidemiologic regression analyses. American journal of epidemiology, 142(12), 1255-1264.

    Seaman SR., White IR. (2013). Review of inverse probability weighting for dealing with missing data.
    Statistical Methods in Medical Research, 22(3), 278-295.
    """
    def __init__(self, df, missing_variable, stabilized=False, monotone=True):
        # Checking input data has missing labeled as np.nan
        if isinstance(missing_variable, str):
            if df.loc[df[missing_variable].isnull(), missing_variable].shape[0] == 0:
                raise ValueError('IPMW requires that missing data is coded as np.nan')
        else:
            for m in missing_variable:
                if df.loc[df[m].isnull(), m].shape[0] == 0:
                    raise ValueError('IPMW requires that missing data is coded as np.nan')

        self.df = df.copy()
        self.missing = missing_variable
        self.stabilized = stabilized
        self.Weight = None
        self._denominator_model = False
        self._monotone = monotone

    def regression_models(self, model_denominator, model_numerator='1', print_results=True):
        """Regression model to generate predicted probabilities of censoring, conditional on specified variables.
        Whether stabilized or unstabilized IPMW are generated depends on the specified model numerator.

        Parameters
        --------------
        model_denominator : str, list
            String of predictor variables for the denominator separated by +. Any variables included in the numerator,
            should be included in the denominator as well. Example 'var1 + var2 + var3 + t_start + t_squared'
        model_numerator : str, optional
            String of predictor variables for the numerator separated by +. In general, time is used as the
            stabilization factor. Example of argument 't_start + t_squared'
        print_results : bool, optional
            Whether to print the model results. Default is True
        """
        if not self.stabilized:
            if model_numerator != '1':
                raise ValueError('Argument for model_numerator is only used for stabilized=True')

        # IPMW for a single missing variable
        if not isinstance(self.missing, list):
            if isinstance(model_denominator, list):
                raise ValueError('For a single missing variable, the model denominator cannot be a list of models')
            self._single_variable(model_denominator, model_numerator=model_numerator, print_results=print_results)

        # IPMW for monotone missing variables
        elif self._monotone:
            # Check if monotone
            self._check_monotone()

            # Check if uniform
            self.missing, overall_uniform = self._check_overall_uniform(df=self.df, miss_vars=self.missing)

            # Uniform monotone missing can safely be treated as only a single variable
            if overall_uniform:
                warnings.warn("It looks like your data is monotone missing data is uniform. Uniform missing data is a "
                              "special case of monotone missing data, where weights need only depend on a single "
                              "variable. The corresponding weights will be generated using the first specified "
                              "regression model")
                self._single_variable(list(model_denominator)[0],
                                      model_numerator=list(model_numerator)[0], print_results=print_results)

            # When not the special case of uniform monotone, use this procedure
            else:
                self._monotone_variables(model_denominator, model_numerator=model_numerator, print_results=print_results)

        # IPMW for nonmonotone missing variables
        else:
            raise ValueError('Non-monotonic IPMW is to be implemented in the future...')

    def fit(self):
        """Calculates the IPMW based on the predicted probabilities from the fitted logistic regression models.
        Calculated weights can be accessed via the `IPMW.Weight` attribute
        """
        if self._denominator_model is False:
            raise ValueError('No model has been fit to generated predicted probabilities')

        w = self.df['__numer__'] / self.df['__denom__']
        self.Weight = w

    def _check_monotone(self):
        """Background function to check whether data is actually monotone. Raise ValueError if user requests
        monotone=True, but the missing data is nonmonotonic.
        """
        miss_order = list(reversed(self.missing))
        for m in range(len(miss_order)):
            if m == 0 or miss_order[m] == len(miss_order):
                continue

            # Checking adjacent variables
            post = np.where(self.df[miss_order[m]].isnull(), 1, 0)
            prior = np.where(self.df[miss_order[m-1]].isnull(), 1, 0)

            # Post must be zero where prior is zero
            check = np.where((post == 1) & (prior == 0))
            if np.any(check):
                raise ValueError('It looks like your data is non-monotonic. Please check that the missing variables'
                                 'are input in the correct order and that your data is missing monotonically')

    @staticmethod
    def _check_overall_uniform(df, miss_vars):
        """Checks whether all the provided variables with missing data are missing uniformly. If so, then the
        algorithm treats it as a single missing variable. This is a special case of monotone missing data
        """
        # Checks that missing always at same rows in data set
        uniform = pd.Series(np.prod(df[miss_vars].notnull(), axis=1)).equals(df[miss_vars[0]].notnull().astype(int))
        if uniform:
            return miss_vars[0], True
        else:
            return miss_vars, False

    @staticmethod
    def _check_uniform(df, miss1, miss2):
        """Checks whether two variables are missing uniformly. If so, this is a special case of monotone missing data
        that can be treated as a single missing variable. This is used for monotone missing data when all provided
        variables are not missing at random
        """
        # Check that two variables are not uniform missing
        uniform = pd.Series(np.prod(df[[miss1, miss2]].notnull(), axis=1)).equals(df[miss1].notnull().astype(int))
        return uniform

    def _single_variable(self, model_denominator, model_numerator, print_results):
        """Estimates probabilities when only a single variable is missing
        """
        self.df.loc[self.df[self.missing].isnull(), '_observed_indicator_'] = 0
        self.df.loc[self.df[self.missing].notnull(), '_observed_indicator_'] = 1

        dmodel = propensity_score(self.df, '_observed_indicator_ ~ ' + model_denominator, print_results=print_results)
        self.df['__denom__'] = np.where(self.df[self.missing].notnull(), dmodel.predict(self.df), np.nan)
        self._denominator_model = True

        if self.stabilized:
            nmodel = propensity_score(self.df, '_observed_indicator_ ~ ' + model_numerator, print_results=print_results)
            self.df['__numer__'] = np.where(self.df[self.missing].notnull(), nmodel.predict(self.df), np.nan)
        else:
            self.df['__numer__'] = np.where(self.df[self.missing].notnull(), 1, np.nan)

    def _monotone_variables(self, model_denominator, model_numerator, print_results):
        """Estimates probabilities under the monotone missing mechanism
        """
        model_denominator = list(model_denominator)
        model_numerator = list(model_numerator)

        # Check to make sure number of models is not more than number of missing variables
        if len(self.missing) < len(model_denominator) or len(self.missing) < len(model_numerator):
            raise ValueError('More models are specified than missing variables!')

        # If less models than missing variables are specified, repeat the last model for all variables
        while len(self.missing) > len(model_denominator):
            model_denominator.append(model_denominator[-1])
        while len(self.missing) > len(model_numerator):
            model_numerator.append(model_numerator[-1])

        # Looping through all missing variables and specified models
        probs_denom = pd.Series([1] * self.df.shape[0])
        probs_num = pd.Series([1] * self.df.shape[0])

        for mv, model_d, model_n in zip(self.missing, model_denominator, model_numerator):
            df = self.df.copy()

            # Restricting to all those observed by the "outer set" variable
            if mv == self.missing[0]:
                uniform = False
            else:
                # Checking to see if this variable and the previous are uniformly missing
                uniform = self._check_uniform(df, miss1=self.missing[self.missing.index(mv) - 1], miss2=mv)
                # Restricting to only observed
                df = df.loc[df[self.missing[self.missing.index(mv) - 1]].notnull()].copy()

            if uniform:
                continue
            else:
                df.loc[df[mv].isnull(), '_observed_indicator_'] = 0
                df.loc[df[mv].notnull(), '_observed_indicator_'] = 1
                dmodel = propensity_score(df, '_observed_indicator_ ~ ' + model_d, print_results=print_results)
                probs_denom = probs_denom * dmodel.predict(self.df)

                # Only for stabilized IPMW with monotone missing data
                if self.stabilized:
                    nmodel = propensity_score(df, '_observed_indicator_ ~ ' + model_n, print_results=print_results)
                    probs_num = probs_num * nmodel.predict(self.df)

        # Calculating Probabilities
        self.df['__denom__'] = np.where(self.df[self.missing[-1]].notnull(), probs_denom, np.nan)
        if self.stabilized:
            self.df['__numer__'] = np.where(self.df[self.missing[-1]].notnull(), probs_num, np.nan)
        else:
            self.df['__numer__'] = np.where(self.df[self.missing[-1]].notnull(), 1, np.nan)
        self._denominator_model = True

