import numpy as np
from .utils import propensity_score


class IPMW:
    def __init__(self, df, missing_variable, stabilized=True):
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

        Example
        -------
        >>>import zepid as ze
        >>>from zepid.causal.ipw import IPMW
        >>>df = ze.load_sample_data(timevary=False)
        >>>ipm = IPMW(df,missing='dead')
        >>>ipm.fit(model='age0 + art + male')
        """
        if df.loc[df[missing_variable].isnull()][missing_variable].shape[0] == 0:
            raise ValueError('IPMW requires that missing data is coded as np.nan')
        self.df = df.copy()
        self.missing = missing_variable
        self.df.loc[self.df[self.missing].isnull(), '_observed_indicator_'] = 0
        self.df.loc[self.df[self.missing].notnull(), '_observed_indicator_'] = 1
        self.stabilized = stabilized
        self.Weight = None
        self.denominator_model = False

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
        dmodel = propensity_score(self.df, '_observed_indicator_ ~ ' + model_denominator, print_results=print_results)
        self.df['__denom__'] = dmodel.predict(self.df)
        self.denominator_model = True

        if self.stabilized is True:
            nmodel = propensity_score(self.df, '_observed_indicator_ ~ ' + model_numerator, print_results=print_results)
            self.df['__numer__'] = nmodel.predict(self.df)
        else:
            if model_numerator != '1':
                raise ValueError('Argument for model_numerator is only used for stabilized=True')

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
        if self.denominator_model is False:
            raise ValueError('No model has been fit to generated predicted probabilities')

        if self.stabilized:
            w = self.df['__numer__'] / self.df['__denom__']
        else:
            w = 1 / self.df['__denom__']
        self.Weight = w
