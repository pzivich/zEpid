import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import DescrStatsW
from scipy.stats import norm

from zepid.causal.utils import (propensity_score, plot_kde, plot_love,
                                standardized_mean_differences, positivity, _bounding_,
                                plot_kde_accuracy, outcome_accuracy)


class AIPTW:
    r"""Augmented inverse probablity weight estimator. This is a simple implementation of AIPW for a time-fixed exposure
    and a single time-point outcome. `AIPTW` does not account for missing data.

    AIPTW have a desirable property compared to either the g-formula or IPTW. Both of the prior estimators require that
    we specify our parametric regression models correctly. AIPTW allows us to have two 'chances' at getting the model
    correct. If either our outcome-model or treatment-model is correctly specified, then our estimate will be unbiased

    The augment-inverse probability weights are calculated from the following formula

    .. math::

        \widehat{DR}(a) = \frac{YA}{\widehat{\Pr}(A=a|L)} - \frac{\hat{Y}^a*(A-\widehat{\Pr}(A=a|L)}{
        \widehat{\Pr}(A=a|L)}

    The risk difference and risk ratio are calculated using the following formulas, respectively

    .. math::

        \widehat{RD} = \widehat{DR}(a=1) - \widehat{DR}(a=0)

    .. math::

        \widehat{RR} = \frac{\widehat{DR}(a=1)}{\widehat{DR}(a=0)}

    Confidence intervals for the risk difference come from the influence curve. Confidence intervals for the risk ratio
    are less straight-forward. To get confidence intervals for the risk ratio, a bootstrap procedure should be used

    Parameters
    ----------
    df : DataFrame
        Pandas DataFrame object containing all variables of interest
    exposure : str
        Column name of the exposure variable. Currently only binary is supported
    outcome : str
        Column name of the outcome variable. Currently only binary is supported
    weights : str, optional
        Column name of weights. Weights allow for items like sampling weights to be used to estimate effects
    alpha : float, optional
        Alpha for confidence interval level. Default is 0.05, returning the 95% CL

    Examples
    --------
    Set up the environment and the data set

    >>> from zepid import load_sample_data, spline
    >>> from zepid.causal.doublyrobust import AIPTW
    >>> df = load_sample_data(timevary=False).drop(columns=['cd4_wk45'])
    >>> df[['cd4_rs1','cd4_rs2']] = spline(df,'cd40',n_knots=3,term=2,restricted=True)
    >>> df[['age_rs1','age_rs2']] = spline(df,'age0',n_knots=3,term=2,restricted=True)

    Initialize the AIPTW model

    >>> aipw = AIPTW(df, exposure='art', outcome='dead')

    Specify the exposure/treatment model

    >>> aipw.exposure_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')

    Specify the outcome model

    >>> aipw.outcome_model('art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')

    Estimate the risk difference and risk ratio

    >>> aipw.fit()

    Displaying the results

    >>> aipw.summary()

    Extracting risk difference and risk ratio respectively

    >>> aipw.risk_difference
    >>> aipw.risk_ratio

    References
    ----------
    Funk, M. J., Westreich, D., Wiesen, C., Stürmer, T., Brookhart, M. A., & Davidian, M. (2011). Doubly robust
    estimation of causal effects. American Journal of Epidemiology, 173(7), 761-767.

    Lunceford JK, Davidian M. (2004). Stratification and weighting via the propensity score in estimation of causal
    treatment effects: a comparative study. Statistics in medicine, 23(19), 2937-2960.
    """
    def __init__(self, df, exposure, outcome, weights=None, alpha=0.05):
        if df.dropna(subset=[d for d in df.columns if d != outcome]).shape[0] != df.shape[0]:
            warnings.warn("There is missing data that is not the outcome in the data set. AIPTW will drop "
                          "all missing data that is not missing outcome data. AIPTW will fit "
                          + str(df.dropna(subset=[d for d in df.columns if d != outcome]).shape[0]) +
                          ' of ' + str(df.shape[0]) + ' observations', UserWarning)
            self.df = df.copy().dropna(subset=[d for d in df.columns if d != outcome]).reset_index()
        else:
            self.df = df.copy().reset_index()

        # Checking to see if missing outcome data occurs
        self._missing_indicator = '__missing_indicator__'
        if self.df.dropna(subset=[outcome]).shape[0] != self.df.shape[0]:
            self._miss_flag = True
            self.df[self._missing_indicator] = np.where(self.df[outcome].isna(), 0, 1)
        else:
            self._miss_flag = False
            self.df[self._missing_indicator] = 1

        self.exposure = exposure
        self.outcome = outcome

        if df[outcome].dropna().value_counts().index.isin([0, 1]).all():
            self._continuous_outcome = False
        else:
            self._continuous_outcome = True

        self._weight_ = weights
        self.alpha = alpha

        self.risk_difference = None
        self.risk_ratio = None
        self.risk_difference_ci = None
        self.risk_ratio_ci = None
        self.risk_difference_se = None

        self.average_treatment_effect = None
        self.average_treatment_effect_ci = None
        self.average_treatment_effect_se = None

        self._fit_exposure_ = False
        self._fit_outcome_ = False
        self._fit_missing_ = False
        self._exp_model = None
        self._out_model = None

    def exposure_model(self, model, bound=False, print_results=True):
        r"""Specify the propensity score / inverse probability weight model. Model used to predict the exposure via a
        logistic regression model. This model estimates

        .. math::

            \widehat{\Pr}(A=1|L) = logit^{-1}(\widehat{\beta_0} + \widehat{\beta} L)

        Parameters
        ----------
        model : str
            Independent variables to predict the exposure. For example, 'var1 + var2 + var3'
        bound : float, list, optional
            Value between 0,1 to truncate predicted probabilities. Helps to avoid near positivity violations.
            Specifying this argument can improve finite sample performance for random positivity violations. However,
            inference becomes limited to the restricted population. Default is False, meaning no truncation of
            predicted probabilities occurs. Providing a single float assumes symmetric trunctation. A collection of
            floats can be provided for asymmetric trunctation
        print_results : bool, optional
            Whether to print the fitted model results. Default is True (prints results)
        """
        self.__mweight = model
        self._exp_model = self.exposure + ' ~ ' + model
        fitmodel = propensity_score(self.df, self._exp_model, weights=self._weight_, print_results=print_results)
        ps = fitmodel.predict(self.df)
        self.df['_g1_'] = ps
        self.df['_g0_'] = 1 - ps

        # If bounds are requested
        if bound:
            self.df['_g1_'] = _bounding_(self.df['_g1_'], bounds=bound)
            self.df['_g0_'] = _bounding_(self.df['_g0_'], bounds=bound)

        self._fit_exposure_ = True

    def outcome_model(self, model, continuous_distribution='gaussian', print_results=True):
        r"""Specify the outcome model. Model used to predict the outcome via a logistic regression model

        .. math::

            \widehat{\Pr}(Y|A,L) = logit^{-1}(\widehat{\beta_0} + \widehat{\beta_1} A + \widehat{\beta} L)

        Parameters
        ----------
        model : str
            Independent variables to predict the outcome. For example, 'var1 + var2 + var3 + var4'
        continuous_distribution : str, optional
            Distribution to use for continuous outcomes. Options are 'gaussian' for normal distributions and 'poisson'
            for Poisson distributions
        print_results : bool, optional
            Whether to print the fitted model results. Default is True (prints results)
        """
        self._out_model = self.outcome + ' ~ ' + model

        if self._continuous_outcome:
            self._continuous_type = continuous_distribution
            if (continuous_distribution == 'gaussian') or (continuous_distribution == 'normal'):
                f = sm.families.family.Gaussian()
            elif continuous_distribution == 'poisson':
                f = sm.families.family.Poisson()
            else:
                raise ValueError("Only 'gaussian' and 'poisson' distributions are supported")
        else:
            f = sm.families.family.Binomial()

        if self._weight_ is None:
            log = smf.glm(self._out_model, self.df, family=f).fit()
        else:
            log = smf.glm(self._out_model, self.df, freq_weights=self.df[self._weight_], family=f).fit()

        if print_results:
            print('\n----------------------------------------------------------------')
            print('MODEL: ' + self._out_model)
            print('-----------------------------------------------------------------')
            print(log.summary())

        # Generating predictions for observed variables
        self._predicted_y_ = log.predict(self.df)

        # Predicting under treatment strategies
        dfx = self.df.copy()
        dfx[self.exposure] = 1
        self.df['_pY1_'] = log.predict(dfx)
        dfx = self.df.copy()
        dfx[self.exposure] = 0
        self.df['_pY0_'] = log.predict(dfx)
        self._fit_outcome_ = True

    def missing_model(self, model, custom_model=None, print_results=True):
        """Estimation of Pr(M=1|A,L), which is the missing data mechanism for the outcome. The corresponding observation
        probabilities are used to update the clever covariates for estimation of Qn.

        The initial estimate of Q is still based on complete observations only

        Parameters
        ----------
        model : str
            Independent variables to predict the exposure. Example) 'var1 + var2 + var3'. The treatment must be
            included for the missing data model
        custom_model : optional
            Input for a custom model that is used in place of the logit model (default). The model must have the
            "fit()" and  "predict()" attributes. Both sklearn and supylearner are supported as custom models. In the
            background, TMLE will fit the custom model and generate the predicted probablities
        print_results : bool, optional
            Whether to print the fitted model results. Default is True (prints results)
        """
        # Error if no missing outcome data
        if not self._miss_flag:
            raise ValueError("No missing outcome data is present in the data set")

        # Warning if exposure is not included in the missingness of outcome model
        if self.exposure not in model:
            warnings.warn("For the specified missing outcome model, the exposure variable should be included in the "
                          "model", UserWarning)

        # Warning if exposure is not included in the missingness of outcome model
        if self.exposure not in model:
            warnings.warn("For the specified missing outcome model, the exposure variable should be included in the "
                          "model", UserWarning)

        self._miss_model = self._missing_indicator + ' ~ ' + model
        fitmodel = propensity_score(self.df, self._miss_model, print_results=print_results)

        dfx = self.df.copy()
        dfx[self.exposure] = 1
        self.df['_ipmw_a1_'] = np.where(self.df[self._missing_indicator] == 1,
                                        fitmodel.predict(dfx), np.nan)
        dfx = self.df.copy()
        dfx[self.exposure] = 0
        self.df['_ipmw_a0_'] = np.where(self.df[self._missing_indicator] == 1,
                                        fitmodel.predict(dfx), np.nan)

        self._fit_missing_ = True

    def fit(self):
        """Once the exposure and outcome models are specified, we can estimate the risk ratio and risk difference.

        Returns
        -------
        Gains `risk_difference`, `risk_difference_ci`, and `risk_ratio` values
        """
        if (self._fit_exposure_ is False) or (self._fit_outcome_ is False):
            raise ValueError('The exposure and outcome models must be specified before the doubly robust estimate can '
                             'be generated')

        # Doubly robust estimator under all treated
        a_obs = self.df[self.exposure]
        y_obs = self.df[self.outcome]
        py_a1 = self.df['_pY1_']
        py_a0 = self.df['_pY0_']

        if self._fit_missing_:
            ps_g1 = self.df['_g1_'] * self.df['_ipmw_a1_']
            ps_g0 = self.df['_g0_'] * self.df['_ipmw_a0_']
        else:
            ps_g1 = self.df['_g1_']
            ps_g0 = self.df['_g0_']

        # Doubly robust estimator under all treated
        dr_a1 = np.where(a_obs == 1,
                         (y_obs / ps_g1) - ((py_a1 * ps_g0) / ps_g1),
                         py_a1)

        # Doubly robust estimator under all untreated
        dr_a0 = np.where(a_obs == 1,
                         py_a0,
                         (y_obs / ps_g0 - ((py_a0 * ps_g1) / ps_g0)))

        # Generating estimates for the risk difference and risk ratio
        zalpha = norm.ppf(1 - self.alpha / 2, loc=0, scale=1)

        if self._weight_ is None:
            if self._continuous_outcome:
                self.average_treatment_effect = np.nanmean(dr_a1) - np.nanmean(dr_a0)
                var_ic = np.nanvar((dr_a1 - dr_a0) - self.average_treatment_effect, ddof=1) / self.df.shape[0]
                self.average_treatment_effect_se = np.sqrt(var_ic)
                self.average_treatment_effect_ci = [self.average_treatment_effect - zalpha * np.sqrt(var_ic),
                                                    self.average_treatment_effect + zalpha * np.sqrt(var_ic)]

            else:
                self.risk_difference = np.nanmean(dr_a1) - np.nanmean(dr_a0)
                self.risk_ratio = np.nanmean(dr_a1) / np.nanmean(dr_a0)
                var_ic = np.nanvar((dr_a1 - dr_a0) - self.risk_difference, ddof=1) / self.df.shape[0]
                self.risk_difference_se = np.sqrt(var_ic)
                self.risk_difference_ci = [self.risk_difference - zalpha * np.sqrt(var_ic),
                                           self.risk_difference + zalpha * np.sqrt(var_ic)]
        else:
            dr_m1 = DescrStatsW(dr_a1, weights=self.df[self._weight_]).mean
            dr_m0 = DescrStatsW(dr_a0, weights=self.df[self._weight_]).mean

            if self._continuous_outcome:
                self.average_treatment_effect = dr_m1 - dr_m0
            else:
                self.risk_difference = dr_m1 - dr_m0
                self.risk_ratio = dr_m1 / dr_m0

    def summary(self, decimal=3):
        """Prints a summary of the results for the doubly robust estimator. Confidence intervals are only available for
        the risk difference currently

        Parameters
        ----------
        decimal : int, optional
            Number of decimal places to display in the result
        """
        if (self._fit_exposure_ is False) or (self._fit_exposure_ is False):
            raise ValueError('The exposure and outcome models must be specified before the double robust estimate can '
                             'be generated')

        print('======================================================================')
        print('          Augmented Inverse Probability of Treatment Weights          ')
        print('======================================================================')
        fmt = 'Treatment:        {:<15} No. Observations:     {:<20}'
        print(fmt.format(self.exposure, self.df.shape[0]))

        fmt = 'Outcome:          {:<15} No. Missing Outcome:  {:<20}'
        print(fmt.format(self.outcome, np.sum(self.df[self.outcome].isnull())))

        fmt = 'g-Model:          {:<15} Q-model:              {:<20}'
        e = 'Logistic'
        if self._continuous_outcome:
            y = self._continuous_type
        else:
            y = 'Logistic'

        print(fmt.format(e, y))
        print('======================================================================')

        if self._continuous_outcome:
            print('Average Treatment Effect:   ', round(float(self.average_treatment_effect), decimal))
            if self._weight_ is None:
                print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: (' +
                      str(round(self.average_treatment_effect_ci[0], decimal)), ',',
                      str(round(self.average_treatment_effect_ci[1], decimal)) + ')')
            else:
                print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: -')
        else:
            print('Risk Difference:    ', round(float(self.risk_difference), decimal))
            if self._weight_ is None:
                print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: (' +
                      str(round(self.risk_difference_ci[0], decimal)), ',',
                      str(round(self.risk_difference_ci[1], decimal)) + ')')
            else:
                print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: -')
            print('----------------------------------------------------------------------')
            print('Risk Ratio:        ', round(float(self.risk_ratio), decimal))
            print(str(round(100 * (1 - self.alpha), 1)) + '% two-sided CI: -')

        print('======================================================================')

    def run_diagnostics(self, decimal=3):
        """Run all currently implemented diagnostics for inverse probability of treatment weights available. Each
        diagnostic can be called individually for further optional specifications. `run_weight_diagnostics` will
        provide results for all implemented diagnostics for ease of the user. For presentation of results, I recommend
        calling each function individually and utilizing the optional parameters

        Note
        ----
        The plot presented cannot be edited. To edit the plots, call `plot_kde` or `plot_love` directly. Those
        functions return an axes object
        """
        if not self._fit_outcome_ or not self._fit_exposure_:
            raise ValueError("The exposure_model and outcome_model function must be ran before any diagnostics")

        # Weight diagnostics
        print('\tExposure Model Diagnostics')
        self.positivity(decimal=decimal)

        print('\n======================================================================')
        print('                  Standardized Mean Differences')
        print('======================================================================')
        print(self.standardized_mean_differences().set_index(keys='labels'))
        print('======================================================================\n')

        # Outcome accuracy diagnostics
        print('\tOutcome Model Diagnostics')
        v = self._predicted_y_ - self.df[self.outcome]
        outcome_accuracy(true=self.df[self.outcome], predicted=self._predicted_y_, decimal=decimal)

        df = self.df.copy()
        df['_ipw_'] = np.where(df[self.exposure] == 1, 1 / df['_g1_'], 1 / (1 - df['_g1_']))

        plt.figure(figsize=[8, 6])
        plt.subplot(221)
        plot_love(df=df, treatment=self.exposure, weight='_ipw_', formula=self.__mweight)
        plt.title("Love Plot")

        plt.subplot(223)
        plot_kde(df=df, treatment=self.exposure, probability='_g1_')
        plt.title("Kernel Density of Propensity Scores")

        plt.subplot(222)
        plot_kde_accuracy(values=v.dropna(), color='green')
        plt.title("Kernel Density of Accuracy")
        plt.tight_layout()
        plt.show()

    def positivity(self, decimal=3):
        """Use this to assess whether positivity is a valid assumption for inverse probability weights. If there are
        extreme outliers, this may indicate problems with the calculated weights

        Parameters
        --------------
        decimal : int, optional
            Number of decimal places to display. Default is three

        Returns
        --------------
        None
            Prints the positivity results to the console but does not return any objects
        """
        pos = positivity(df=self.df, weights='_g1_')
        print('======================================================================')
        print('                      Weight Positivity Diagnostics')
        print('======================================================================')
        print('If the mean of the weights is far from either the min or max, this may\n '
              'indicate the model is incorrect or positivity is violated')
        print('Average weight should be 2')
        print('----------------------------------------------------------------------')
        print('Mean weight:           ', round(pos[0], decimal))
        print('Standard Deviation:    ', round(pos[1], decimal))
        print('Minimum weight:        ', round(pos[2], decimal))
        print('Maximum weight:        ', round(pos[3], decimal))
        print('======================================================================\n')

    def standardized_mean_differences(self):
        """Calculates the standardized mean differences for all variables based on the inverse probability weights.

        Returns
        -------
        DataFrame
            Returns pandas DataFrame of calculated standardized mean differences. Columns are labels (variables labels),
            smd_u (unweighted standardized difference), and smd_w (weighted standardized difference)
        """
        df = self.df.copy()
        df['_ipw_'] = np.where(df[self.exposure] == 1, 1 / df['_g1_'], 1 / (1 - df['_g1_']))
        s = standardized_mean_differences(df=df, treatment=self.exposure,
                                          weight='_ipw_', formula=self.__mweight)
        return s

    def plot_kde(self, to_plot, bw_method='scott', fill=True,
                 color='g', color_e='b', color_u='r'):
        """Generates density plots that can be used to check whether positivity may be violated qualitatively, and
        accuracy of the outcome model

        The kernel density used is SciPy's Gaussian kernel. Either Scott's Rule or Silverman's Rule can be implemented.

        Parameters
        ------------
        to_plot : str, optional
            The plot to generate. Specifying 'exposure' returns only the density plot for treatment probabilities,
            and 'outcome' returns only the density plot for the outcome accuracy
        bw_method : str, optional
            Method used to estimate the bandwidth. Following SciPy, either 'scott' or 'silverman' are valid options
        fill : bool, optional
            Whether to color the area under the density curves. Default is true
        color : str, optional
            Color of the line/area for predicted outcomes minus observed outcomes. Default is Green
        color_e : str, optional
            Color of the line/area for the treated group. Default is Blue
        color_u : str, optional
            Color of the line/area for the treated group. Default is Red

        Returns
        ---------------
        matplotlib axes
        """
        if to_plot == 'exposure':
            ax = plot_kde(df=self.df, treatment=self.exposure, probability='_g1_',
                          bw_method=bw_method, fill=fill, color_e=color_e, color_u=color_u)
            ax.set_title("Kernel Density of Propensity Scores")

        elif to_plot == 'outcome':
            v = self._predicted_y_ - self.df[self.outcome]
            ax = plot_kde_accuracy(values=v.dropna(), bw_method=bw_method, fill=fill, color=color)
            ax.set_title("Kernel Density of Accuracy")

        else:
            raise ValueError("Please use one of the following options for `to_plot`; 'treatment', 'outcome'")

        return ax

    def plot_love(self, color_unweighted='r', color_weighted='b', shape_unweighted='o', shape_weighted='o'):
        """Generates a Love-plot to detail covariate balance based on the IPTW weights. Further details on the usage of
        this plot are available in Austin PC & Stuart EA 2015 https://onlinelibrary.wiley.com/doi/full/10.1002/sim.6607

        The Love plot generates a dashed line at standardized mean difference of 0.10. Ideally, weighted SMD are below
        this level. Below 0.20 may also be sufficient. Variables above this level may be unbalanced despite the
        weighting procedure. Different functional forms (or approaches like machine learning) may be worth considering

        Parameters
        ----------
        color_unweighted : str, optional
            Color for the unweighted standardized mean differences. Default is red
        color_weighted : str, optional
            Color for the weighted standardized mean differences. Default is blue
        shape_unweighted : str, optional
            Shape of points for the unweighted standardized mean differences. Default is circles
        shape_weighted:
            Shape of points for the weighted standardized mean differences. Default is circles

        Returns
        -------
        axes
            Matplotlib axes of the Love plot
        """
        df = self.df.copy()
        df['_ipw_'] = np.where(df[self.exposure] == 1, 1 / df['_g1_'], 1 / (1 - df['_g1_']))

        ax = plot_love(df=df, treatment=self.exposure, weight='_ipw_', formula=self.__mweight,
                       color_unweighted=color_unweighted, color_weighted=color_weighted,
                       shape_unweighted=shape_unweighted, shape_weighted=shape_weighted)
        return ax
