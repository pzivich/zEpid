import numpy as np
import pandas as pd
from .utils import propensity_score


class IPMW:
    def __init__(self, df, missing_variable, stabilized=True, monotone=True):
        """Calculates the weight for inverse probability of missing weights using logistic regression. IPMW
        automatically codes a missingness indicator (based on np.nan), so data can be directly input, without creation
        of missingness indicator before inputting data

        Parameters
        ----------
        df : DataFrame
            Pandas dataframe object containing all variables of interest
        missing_variable :
            Column name for missing data. numpy.nan values should indicate missing observations
        stabilized : bool, optional
            Whether to return the stabilized or unstabilized IPMW. Default is to return stabilized weights

        Examples
        --------
        Setting up the environment
        >>>from zepid import load_sample_data
        >>>from zepid.causal.ipw import IPMW
        >>>df = load_sample_data(timevary=False)

        Calculating stabilized Inverse Probability of Missingness Weights
        >>>ipm = IPMW(df, missing='dead', stabilized=True)
        >>>ipm.regression_models(model_denominator='age0 + art + male')
        >>>ipm.fit()

        Extracting calculated weights
        >>>ipm.Weight
        """
        # Checking input data has missing labeled as np.nan
        if type(missing_variable) is str:
            if df.loc[df[missing_variable].isnull()][missing_variable].shape[0] == 0:
                raise ValueError('IPMW requires that missing data is coded as np.nan')
        else:
            for m in missing_variable:
                if df.loc[df[m].isnull()][m].shape[0] == 0:
                    raise ValueError('IPMW requires that missing data is coded as np.nan')

        self.df = df.copy()
        self.missing = missing_variable
        self.stabilized = stabilized
        self.Weight = None
        self._denominator_model = False
        self._monotone = monotone

    def regression_models(self, model_denominator, model_numerator='1', print_results=True):
        """Regression model to generate predicted probabilities of censoring, conditional on specified variables.
        Whether stabilized or unstabilized IPCW are generated depends on the specified model numerator.

        Parameters
        --------------
        model_denominator : str
            String of predictor variables for the denominator separated by +. Any variables included in the numerator,
            should be included in the denominator as well. Example 'var1 + var2 + var3 + t_start + t_squared'
        model_numerator : str, optional
            String of predictor variables for the numerator separated by +. In general, time is used as the
            stabilization factor. Example of argument 't_start + t_squared'
        print_results : bool, optional
            Whether to print the model results. Default is True
        """
        # TODO check in uniformly missing. If so, then throw into single variable
        if not self.stabilized:
            if model_numerator != '1':
                raise ValueError('Argument for model_numerator is only used for stabilized=True')

        # IPMW for a single missing variable
        if type(self.missing) is not list:
            if type(model_denominator) is list:
                raise ValueError('For a single missing variable, the model denominator cannot be a list of models')
            self._single_variable(model_denominator, model_numerator=model_numerator, print_results=True)

        # IPMW for monotone missing variables
        elif self._monotone:
            self._check_monotone()
            self._monotone_variables(model_denominator, model_numerator=model_numerator, print_results=True)

        # IPMW for nonmonotone missing variables
        else:
            raise ValueError('Nonmonotonic IPMW is to be implemented still...')

    def fit(self):
        """
        Provide the regression model to generate the inverse probability of missing weights. The fitted regression
        model will be used to generate the IPW. The weights can be accessed via the IMPW.Weight attribute

        model:
            -statsmodels glm format for modeling data. Independent variables should be predictive of missingness of
             variable of interest. Example) 'var1 + var2 + var3'
        print_results:
            -whether to print the model results. Default is True
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
            if m == 0:
                pass
            elif miss_order[m] == len(miss_order):
                pass
            else:
                post = np.where(self.df[miss_order[m]].isnull(), 1, 0)
                prior = np.where(self.df[miss_order[m-1]].isnull(), 1, 0)

                # Post must be zero where prior is zero
                check = np.where((post == 1) & (prior == 0))
                if np.any(check):
                    raise ValueError('It looks like your data is not monotonic. Please check that the missing variables'
                                     'are input in the right order and that your data is missing monotonically')

    def _single_variable(self, model_denominator, model_numerator, print_results):
        """Estimates probabilities when only a single variable is missing
        """
        self.df.loc[self.df[self.missing].isnull(), '_observed_indicator_'] = 0
        self.df.loc[self.df[self.missing].notnull(), '_observed_indicator_'] = 1

        dmodel = propensity_score(self.df, '_observed_indicator_ ~ ' + model_denominator, print_results=print_results)
        self.df['__denom__'] = dmodel.predict(self.df)
        self._denominator_model = True

        if self.stabilized:
            nmodel = propensity_score(self.df, '_observed_indicator_ ~ ' + model_numerator, print_results=print_results)
            self.df['__numer__'] = nmodel.predict(self.df)
        else:
            self.df['__numer__'] = 1

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
        probs_denom = pd.Series([1]*len(self.df))
        probs_num = pd.Series([1]*len(self.df))
        for mv, model_d, model_n in zip(self.missing, model_denominator, model_numerator):
            df = self.df.copy()
            # Restricting to all those observed by the "outer set" variable
            if mv == self.missing[0]:
                pass
            else:
                df = df.loc[df[self.missing[self.missing.index(mv) - 1]].notnull()].copy()
            df.loc[df[mv].isnull(), '_observed_indicator_'] = 0
            df.loc[df[mv].notnull(), '_observed_indicator_'] = 1
            dmodel = propensity_score(df, '_observed_indicator_ ~ ' + model_d, print_results=print_results)
            probs_denom = probs_denom * dmodel.predict(self.df)

            # Only for stabilized IPMW with monotone missing data
            if self.stabilized:
                nmodel = propensity_score(df, '_observed_indicator_ ~ ' + model_n, print_results=print_results)
                probs_num = probs_num * nmodel.predict(self.df)

        # Calculating Probabilities
        self.df['__denom__'] = probs_denom
        if self.stabilized:
            self.df['__numer__'] = probs_num
        else:
            self.df['__numer__'] = 1
        self._denominator_model = True

