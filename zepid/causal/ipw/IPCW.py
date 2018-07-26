import warnings
import numpy as np
import pandas as pd

from .utils import propensity_score

# TODO update docs
# TODO rerun for website

class IPCW:
    """
    Generate inverse probability of censoring weights
    """

    def __init__(self, df, idvar, time, event, flat_df=False, enter=None):
        """
        Calculate the IPCW. Note that this function will accept either a flat file (one row per individual) or a long
        format (multiple rows per individual). If a flat file is provided, it must be converted to a long format. This
        will be done automatically if flat_df is set to True.

        IPC weights are calculated via logistic regression and weights are cumulative products per unique ID. IPCW can
        be used to correct for missing at random data by the generated model in weighted Kaplan-Meier curves

        :param df: pandas DataFrame object containing all the variables of interest
        :param idvar: variable indicating a unique identifier for each individual followed over time
        :param time: column containing the observation time
        :param event:
        :param flat_df:
        :param enter:
        :param warn:
        """
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
            self.df = f
            self.df['uncensored'] = np.where((self.df[idvar] != self.df[idvar].shift(-1)) &
                                             (self.df[event] == 0),
                                             0, 1)  # generating indicator for uncensored
        self.idvar = idvar
        self.time = time
        self.event = event
        self.Weight = None

    def regression_models(self, model_denominator, model_numerator, print_model_results=True):
        """

        :param model_denominator: statsmodels glm format for modeling data. Only predictor variables for the
            denominator (variables determined to be related to censoring). All variables included in the numerator,
            should be included in the denominator. Ex) 'var1 + var2 + var3 + t_start + t_squared'
        :param model_numerator: statsmodels glm format for modeling data. Only includes predictor variables for the
            numerator. In general, time is included in the numerator. Example) 't_start + t_squared'
        :param print_model_results: whether to print the model results. Default is True
        """
        nmodel = propensity_score(self.df, 'uncensored ~ ' + model_numerator, mresult=print_model_results)
        self.df['numer'] = nmodel.predict(self.df)
        dmodel = propensity_score(self.df, 'uncensored ~ ' + model_denominator, mresult=print_model_results)
        self.df['denom'] = dmodel.predict(self.df)
        self.df['cnumer'] = self.df.groupby(self.idvar)['numer'].cumprod()
        self.df['cdenom'] = self.df.groupby(self.idvar)['denom'].cumprod()

    def fit(self):
        self.df['ipcw'] = self.df['cnumer'] / self.df['cdenom']
        self.Weight = self.df['ipcw']

    def _dataprep(self, cf, idvar, time, event, enter=None):
        """
        Function to prepare the data to an appropriate format for the IPCW class. It breaks the dataset into
        single observations for event one unit increase in time.

        :param cf: pandas dataframe to convert into a long format
        :param idvar: ID variable to retain for observations
        :param time: Last follow-up visit for participant
        :param event: indicator of whether participant had the event (1 is yes, 0 is no)
        :param enter: entry time for the participant. Default is None, which means all participants are assumed
            to enter at time zero. Input should be column name of entrance time
        :param warn: whether to generate warning for user
        :return:
        """
        # Copying observations over times
        cf['t_int_zepid'] = cf[time].astype(int)
        lf = pd.DataFrame(np.repeat(cf.values, cf['t_int_zepid'] + 1, axis=0), columns=cf.columns)
        lf['tpoint_zepid'] = lf.groupby(idvar)['t_int_zepid'].cumcount()
        lf['tdiff_zepid'] = lf[time] - lf['tpoint_zepid']
        lf = lf.loc[
            lf['tdiff_zepid'] != 0].copy()  # gets rid of censored at absolute time point (ex. censored at time 10)
        lf.loc[lf['tdiff_zepid'] > 1, 'delta_indicator_zepid'] = 0
        lf.loc[((lf['tdiff_zepid'] <= 1) & (lf[event] == 0)), 'delta_indicator_zepid'] = 0
        lf.loc[((lf['tdiff_zepid'] <= 1) & (lf[event] == 1)), 'delta_indicator_zepid'] = 1
        lf['t_enter_zepid'] = lf['tpoint_zepid']
        lf['t_out_zepid'] = np.where(lf['tdiff_zepid'] < 1, lf[time], lf['t_enter_zepid'] + 1)
        lf['uncensored_zepid'] = np.where((lf[idvar] != lf[idvar].shift(-1)) & (lf['delta_indicator_zepid'] == 0), 0, 1)

        # Removing blocks of observations that would have occurred before entrance into the sample
        if enter is not None:
            lf = lf.loc[lf['t_enter_zepid'] >= lf[enter]].copy()

        # Cleaning up the edited dataframe to return to user
        if enter is None:
            lf.drop(columns=['tdiff_zepid', 'tpoint_zepid', 't_int_zepid', time, event], inplace=True)
        else:
            lf.drop(columns=['tdiff_zepid', 'tpoint_zepid', 't_int_zepid', time, event, enter], inplace=True)
        lf.rename(columns={"delta_indicator_zepid": event, 'uncensored_zepid': 'uncensored', 't_enter_zepid': 't_enter',
                           't_out_zepid': 't_out'}, inplace=True)
        warnings.warn('Please verify the long dataframe was generated correctly')
        print('Check for dataframe')
        print('\tEvents in input:', np.sum(cf[event]))
        print('\tEvents in output:', np.sum(lf[event]))
        print('\tCensor in input:', cf.dropna(subset=[event]).shape[0] - np.sum(cf[event]))
        print('\tCensor in output:', lf.shape[0] - np.sum(lf.uncensored))
        if enter is None:
            print('\tTotal t input:', np.sum(cf[time]))
            print('\tTotal t output:', np.sum(lf.loc[(lf[idvar] != lf[idvar].shift(-1))]['t_out']))
        else:
            print('\tLate in input:', np.sum(cf[enter] != 0))
            print('\tLate in output:', np.sum(lf.loc[((lf['t_enter'] != 0) & (lf[idvar] != lf[idvar].shift(-1)))]))
            print('\tTotal t input:', np.sum(cf[time] - cf[enter]))
            print('\tTotal t output:', lf.shape[0])
        return lf

