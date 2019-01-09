import warnings
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from zepid.causal.ipw import propensity_score


class AIPW:
    def __init__(self, df, exposure, outcome):
        """Implementation of augmented inverse probablity weight estimator. This is a simple implementation of AIPW
        and does not account for missing data. It only handles time-fixed confounders and treatments at this point.

        The augment-inverse probability weights are calculated from the following formula

        .. math::

            RD = (\frac{Y_{A=1}*A}{\Pr(A=1)} - \frac{\hat{Y}_1*(A-\Pr(A=1)}{\Pr(A=1)}) - (\frac{Y_{A=0}*A}{\Pr(A=0)
            } - \frac{\hat{Y}_0*(A-\Pr(A=1)}{\Pr(A=0)})
            RR = (\frac{Y_{A=1}*A}{\Pr(A=1)} - \frac{\hat{Y}_1*(A-\Pr(A=1)}{\Pr(A=1)}) / (\frac{Y_{A=0}*A}{\Pr(A=0)
            } - \frac{\hat{Y}_0*(A-\Pr(A=1)}{\Pr(A=0)})

        Confidence intervals should be calculated using non-parametric bootstrapping

        Parameters
        ----------
        df : DataFrame
            Pandas DataFrame object containing all variables of interest
        exposure : str
            Column name of the exposure variable. Currently only binary is supported
        outcome : str
            Column name of the outcome variable. Currently only binary is supported

        Examples
        --------
        Set up the environment and the data set
        >>>from zepid import load_sample_data, spline
        >>>from zepid.causal.doublyrobust import AIPW
        >>>df = load_sample_data(timevary=False)
        >>>df[['cd4_rs1','cd4_rs2']] = spline(df,'cd40',n_knots=3,term=2,restricted=True)
        >>>df[['age_rs1','age_rs2']] = spline(df,'age0',n_knots=3,term=2,restricted=True)

        Initialize the AIPTW model
        >>>aipw = AIPW(df, exposure='art', outcome='dead')

        Specify the exposure/treatment model
        >>>aipw.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')

        Specify the outcome model
        >>>aipw.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')

        Estimate the risk difference and risk ratio
        >>>aipw.fit()

        Displaying the results
        >>>aipw.summary()

        Extracting risk difference and risk ratio respectively
        >>>aipw.risk_difference
        >>>aipw.risk_ratio

        References
        ----------
        Funk, M. J., Westreich, D., Wiesen, C., Stürmer, T., Brookhart, M. A., & Davidian, M. (2011). Doubly robust
        estimation of causal effects. American Journal of Epidemiology, 173(7), 761-767.
        """
        self.df = df.copy()
        if df.dropna().shape[0] != df.shape[0]:
            warnings.warn("There is missing data in the dataset. By default, AIPW will drop all missing data. AIPW will"
                          "fit "+str(df.dropna().shape[0])+' of '+str(df.shape[0])+' observations', UserWarning)
        self.df = df.copy().dropna().reset_index()
        self._exposure = exposure
        self._outcome = outcome
        self._fit_exposure_model = False
        self._fit_outcome_model = False
        self._generated_ci = False
        self.risk_difference = None
        self.risk_ratio = None
        self._exp_model = None
        self._out_model = None

    def exposure_model(self, model, print_results=True):
        """Specify the propensity score / inverse probability weight model. Model used to predict the exposure via a
        logistic regression model

        Parameters
        ----------
        model : str
            Independent variables to predict the exposure. For example, 'var1 + var2 + var3'
        print_results : bool, optional
            Whether to print the fitted model results. Default is True (prints results)
        """
        self._exp_model = self._exposure + ' ~ ' + model
        fitmodel = propensity_score(self.df, self._exp_model, print_results=print_results)
        self.df['ps'] = fitmodel.predict(self.df)
        self._fit_exposure_model = True

    def outcome_model(self, model, print_results=True):
        """Specify the outcome model. Model used to predict the outcome via a logistic regression model

        Parameters
        ----------
        model : str
            Independent variables to predict the outcome. For example, 'var1 + var2 + var3 + var4'
        print_results : bool, optional
            Whether to print the fitted model results. Default is True (prints results)
        """
        self._out_model = self._outcome + ' ~ ' + model
        f = sm.families.family.Binomial()
        log = smf.glm(self._out_model, self.df, family=f).fit()
        if print_results:
            print('\n----------------------------------------------------------------')
            print('MODEL: ' + self._out_model)
            print('-----------------------------------------------------------------')
            print(log.summary())

        dfx = self.df.copy()
        dfx[self._exposure] = 1
        self.df['pY1'] = log.predict(dfx)
        dfx = self.df.copy()
        dfx[self._exposure] = 0
        self.df['pY0'] = log.predict(dfx)
        self._fit_outcome_model = True

    def fit(self):
        """Once the exposure and outcome models are specified, we can estimate the Risk Ratio and Risk Difference.
        This function generates the estimated risk difference and risk ratio. For confidence intervals, a non-parametric
        bootstrap procedure should be used
        """
        if (self._fit_exposure_model is False) or (self._fit_outcome_model is False):
            raise ValueError('The exposure and outcome models must be specified before the doubly robust estimate can '
                             'be generated')

        # Doubly robust estimator under all treated
        self.df['dr1'] = np.where(self.df[self._exposure] == 1,
                                  ((self.df[self._outcome]) / self.df['ps']) - (((self.df['pY1'] * (1 - self.df['ps']))
                                                                                 / (self.df['ps']))),
                                  self.df['pY1'])

        # Doubly robust estimator under all untreated
        self.df['dr0'] = np.where(self.df[self._exposure] == 0,
                                  (self.df['pY0']),
                                  ((self.df[self._outcome]) / (1 - self.df['ps']) - (
                                              ((self.df['pY0']) * (self.df['ps']))
                                              / (1 - self.df['ps']))))

        # Generating estimates for the risk difference and risk ratio
        self.risk_difference = np.mean(self.df['dr1']) - np.mean(self.df['dr0'])
        self.risk_ratio = np.mean(self.df['dr1']) / np.mean(self.df['dr0'])

    def summary(self, decimal=4):
        """Prints a summary of the results for the doubly robust estimator.

        Parameters
        ----------
        decimal : int, optional
            Number of decimal places to display in the result
        """
        if (self._fit_exposure_model is False) or (self._fit_exposure_model is False):
            raise ValueError('The exposure and outcome models must be specified before the double robust estimate can '
                             'be generated')

        print('----------------------------------------------------------------------')
        print('Risk Difference: ', round(float(self.risk_difference), decimal))
        print('Risk Ratio: ', round(float(self.risk_ratio), decimal))
        print('----------------------------------------------------------------------')
