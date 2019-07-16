import warnings
import numpy as np
import pandas as pd

from zepid.causal.utils import propensity_score


class IPCW:
    r"""Calculates inverse probability of censoring weights. Note that this function will accept either a flat
    file (one row per individual) or a long format (multiple rows per individual). If a flat file is provided, it
    must be converted to a long format. This will be done automatically if `flat_df=True`. Additionally, a
    warning and some comparison statistics are provided. Please verify that they match. In general, it is recommended
    to convert the data set yourself

    IPCW are calculated via logistic regression and weights are cumulative products per unique ID. IPCW can
    be used to correct for missing at random data by the generated model in weighted Kaplan-Meier curves. The
    formula used to generate the unstabilized IPCW is

    .. math::

        \pi_i(t) = \prod_{R_k \le t} \frac{1}{\Pr(C_i > R_k | \bar{L} = \bar{l}, C_i > R_{k-1})}

    The stabilized IPCW substitutes predicted probabilities under the specified numerator model into the numerator
    of the previous equation. In general, it is recommended to stabilize IPCW by the time.

    .. math::

        \pi_i(t) = \prod_{R_k \le t} \frac{\Pr(C_i > R_k)}{\Pr(C_i > R_k | \bar{L} = \bar{l}, C_i > R_{k-1})}

    Note
    ----
    IPCW no longer support late-entry. The reason is that the pooled logistic regression model approach does not
    correctly accumulate the weights. As such, either all occurrences of late-entries need to be dropped (called the
    new-user design) or rows need to be back-propagated (unobserved rows are filled in). The second approach requires
    filling in the missing observed covariates and for time-varying variables will require imputation. The new-user
    design is a safer bet and generally what I will currently recommend

    Parameters
    ---------------
    df : DataFrame
        Pandas DataFrame object containing all the variables of interest
    idvar : str
        String that indicates the column name for a unique identifier for each individual
    time : str
        Column name for the ending observation time
    event : str
        Column name for the event of interest
    flat_df : bool, optional
        Whether the input dataframe only contains a single row per participant. If so, the flat dataframe is
        converted to a long dataframe. Default is False (for multiple rows per person)
    enter : str, optional
        Time participant began being observed. Default is None. This option is only needed when flat_df=True.
        Late-entries are no longer supported and specifying this will lead to a ValueError

    Example
    ------------
    Setting up the environment

    >>> from zepid import load_sample_data
    >>> from zepid.causal.ipw import IPCW
    >>> df = load_sample_data(timevary=True)
    >>> df['enter_q'] = df['enter'] ** 2
    >>> df['enter_c'] = df['enter'] ** 3
    >>> df['age0_q'] = df['age0'] ** 2
    >>> df['age0_c'] = df['age0'] ** 3

    Calculating stabilized IPCW with a long data set

    >>> ipc = IPCW(df, idvar='id', time='enter', event='dead')
    >>> ipc.regression_models(model_denominator='enter + enter_q + enter_c + male + age0 + age0_q + age0_c',
    >>>                       model_numerator='enter + enter_q + enter_c')
    >>> ipc.fit()

    Extracting calculated stabilized IPCW

    >>> ipc.Weight

    Calculating stabilized IPCW with a wide data set

    >>> df = load_sample_data(False)
    >>> ipc = IPCW(df, idvar='id', time='t', event='dead', flat_df=True)
    >>> ipc.regression_models(model_denominator='enter + enter_q + enter_c + male + age0 + age0_q + age0_c',
    >>>                       model_numerator='enter + enter_q + enter_c')
    >>> ipc.fit()

    References
    ----------
    Howe CJ et al. (2016) Selection bias due to loss to follow up in cohort studies. Epidemiology, 27(1), 91-97.
    """
    def __init__(self, df, idvar, time, event, flat_df=False, enter=None):
        if np.sum(df[time].isnull()) > 0:
            raise ValueError('Time is missing for at least one individual in the dataset. They must be removed before '
                             'IPCW can be fit.')
        f = df.copy()
        f.sort_values(by=[idvar, time], inplace=True)
        if np.max(f[time]) == 1:
            raise ValueError('The maximum observation time is 1. For IPCW to function properly, it needs to be greater'
                             'than 1. The more categories, the better the weight estimation')

        if flat_df:
            self.df = self._dataprep(f, idvar, time, event, enter=enter)
        else:
            late_check = f.drop_duplicates(subset=idvar, keep='first')
            if not np.all(late_check[time] <= 1):
                raise ValueError("IPCW no longer supports late-entries. IPCW cannot be correctly estimated using "
                                 "pooled logistic regression with late-entries. A future update will allow for weights "
                                 "with both late-entry and censoring")

            self.df = f
            self.df['__uncensored__'] = np.where((self.df[idvar] != self.df[idvar].shift(-1)) &
                                                 (self.df[event] == 0),
                                                 0, 1)  # generating indicator for uncensored
            self.df['__uncensored__'] = np.where(self.df[time] == np.max(df[time]), 1, self.df['__uncensored__'])

        self.idvar = idvar
        self.time = time
        self.event = event
        self.Weight = None

    def regression_models(self, model_denominator, model_numerator, print_results=True):
        """Regression model to generate predicted probabilities of censoring, conditional on specified variables.
        Whether stabilized or unstabilized IPCW are generated depends on the specified model numerator.

        Parameters
        --------------
        model_denominator : str
            String of predictor variables for the denominator following `patsy` syntax. Any variables included in the
            numerator, should be included in the denominator as well. Example 'var1 + var2 + var3 + t_start + t_squared'
        model_numerator : str
            String of predictor variables for the numerator following `patsy` syntax. In general, time is used as the
            stabilization factor. Example of argument 't_start + t_squared'
        print_results : bool, optional
            Whether to print the model results. Default is True
        """
        nmodel = propensity_score(self.df, '__uncensored__ ~ ' + model_numerator, print_results=print_results)
        self.df['__numer__'] = nmodel.predict(self.df)
        dmodel = propensity_score(self.df, '__uncensored__ ~ ' + model_denominator, print_results=print_results)
        self.df['__denom__'] = dmodel.predict(self.df)
        self.df['__cnumer__'] = self.df.groupby(self.idvar)['__numer__'].cumprod()
        self.df['__cdenom__'] = self.df.groupby(self.idvar)['__denom__'].cumprod()

    def fit(self):
        """Calculates IPCW for each observation period for each observation. The calculated weights can be
        accessed through the `IPCW.Weights` attribute

        Returns
        -------------
        Fills in the `Weight` attribute
        """
        self.Weight = self.df['__cnumer__'] / self.df['__cdenom__']

    @staticmethod
    def _dataprep(cf, idvar, time, event, enter=None):
        """Function to prepare the data to an appropriate format for the `IPCW` class. It breaks the dataset into
        single observations for event one unit increase in time. This is a background process and not meant for users
        to access

        Parameters for my reference
        cf:
            -pandas dataframe to convert into a long format
        idvar:
            -ID variable to retain for observations
        time:
            -last follow-up visit for participant
        event:
            -indicator of whether participant had the event (1 is yes, 0 is no)
        enter:
            -entry time for the participant. Default is None, which means all participants are assumed
             to enter at time zero. Input should be column name of entrance time
        """
        if enter is not None:
            raise ValueError("IPCW no longer supports late-entries. IPCW cannot be correctly estimated using pooled "
                             "logistic regression with late-entries. A future update will allow for weights with "
                             "both late-entry and censoring")

        # Copying observations over times
        cf['t_int_zepid'] = cf[time].astype(int)
        lf = pd.DataFrame(np.repeat(cf.values, cf['t_int_zepid'] + 1, axis=0), columns=cf.columns)
        lf['tpoint_zepid'] = lf.groupby(idvar)['t_int_zepid'].cumcount()
        lf['tdiff_zepid'] = lf[time] - lf['tpoint_zepid']
        lf = lf.loc[
            lf['tdiff_zepid'] != 0].copy()  # gets rid of censored at absolute time point (ex. censored at time 10)
        lf.loc[lf['tdiff_zepid'] > 1, 'delta_indicator_zepid'] = 0
        lf.loc[((lf['tdiff_zepid'] <= 1) & (lf[event] == 0)), 'delta_indicator_zepid'] = 0
        lf.loc[((lf['tdiff_zepid'] <= 1) & (lf[event] == 1) & (lf[idvar] != lf[idvar].shift(-1))),
               'delta_indicator_zepid'] = 1
        lf['t_enter_zepid'] = lf['tpoint_zepid']
        lf['t_out_zepid'] = np.where(lf['tdiff_zepid'] < 1, lf[time], lf['t_enter_zepid'] + 1)
        lf['uncensored_zepid'] = np.where((lf[idvar] != lf[idvar].shift(-1)) & (lf['delta_indicator_zepid'] == 0), 0, 1)
        lf['uncensored_zepid'] = np.where(lf['t_out_zepid'] == np.max(lf['t_out_zepid']), 1, lf['uncensored_zepid'])

        # Cleaning up the edited dataframe to return to user
        lf.drop(['tdiff_zepid', 'tpoint_zepid', 't_int_zepid', time, event], axis=1, inplace=True)
        lf.rename(columns={"delta_indicator_zepid": event, 'uncensored_zepid': '__uncensored__',
                           't_enter_zepid': 't_enter', 't_out_zepid': 't_out'}, inplace=True)

        warnings.warn('Please verify the long dataframe was generated correctly', UserWarning)
        print('Check for dataframe')
        print('\tEvents in input:', np.sum(cf[event]))
        print('\tEvents in output:', np.sum(lf[event]))
        print('\tCensor in input:', cf.dropna(subset=[event]).shape[0] - np.sum(cf[event]))
        print('\tCensor in output:', lf.shape[0] - np.sum(lf['__uncensored__']))
        print('\tTotal t input:', np.sum(cf[time]))
        print('\tTotal t output:', np.sum(lf.loc[(lf[idvar] != lf[idvar].shift(-1))]['t_out']))
        return lf.reset_index(drop=True)

