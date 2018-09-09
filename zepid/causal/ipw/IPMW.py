import numpy as np
from .utils import propensity_score


class IPMW:
    """
    IPMW model

    Example)
    >>>import zepid as ze
    >>>from zepid.causal.ipw import IPMW
    >>>df = ze.load_sample_data(timevary=False)
    >>>ipm = IPMW(df,missing='dead')
    >>>ipm.fit(model='age0 + art + male')
    """

    def __init__(self, df, missing_variable, stabilized=True):
        """Calculates the weight for inverse probability of missing weights using logistic regression.
        Function automatically codes a missingness indicator (based on np.nan), so data can be directly
        input.

        df:
            -pandas dataframe object containing all variables of interest
        missing_variable:
            -column name for missing data. numpy.nan values should indicate missing observations
        stabilized:
            -whether to return the stabilized or unstabilized IPMW. Default is to return stabilized weights
        """
        if df.loc[df[missing_variable].isnull()][missing_variable].shape[0] == 0:
            raise ValueError('IPMW requires that missing data is coded as np.nan')
        self.df = df.copy()
        self.missing = missing_variable
        self.df.loc[self.df[self.missing].isnull(), 'observed_indicator'] = 0
        self.df.loc[self.df[self.missing].notnull(), 'observed_indicator'] = 1
        self.stabilized = stabilized
        self.Weight = None

    def fit(self, model, print_results=True):
        """
        Provide the regression model to generate the inverse probability of missing weights. The fitted regression
        model will be used to generate the IPW. The weights can be accessed via the IMPW.Weight attribute

        model:
            -statsmodels glm format for modeling data. Independent variables should be predictive of missingness of
             variable of interest. Example) 'var1 + var2 + var3'
        print_results:
            -whether to print the model results. Default is True
        """
        mod = propensity_score(self.df, 'observed_indicator ~ ' + model, print_results=print_results)
        p = mod.predict(self.df)
        if self.stabilized:
            p_ = np.mean(self.df['observed_indicator'])
            w = p_ / p
        else:
            w = p ** -1
        self.df['ipmw'] = w
        self.Weight = w
