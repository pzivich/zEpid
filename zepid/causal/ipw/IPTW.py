import warnings
import patsy
import numpy as np
import pandas as pd
from scipy.stats.kde import gaussian_kde
from statsmodels.stats.weightstats import DescrStatsW
import matplotlib.pyplot as plt
from .utils import propensity_score

from zepid.calc import probability_to_odds


class IPTW:
    r"""Calculates inverse probability of treatment weights. Both stabilized or unstabilized weights are implemented.
    By default, stabilized weights are stabilized by the prevalence of the treatment in the population. `IPTW` will
    return an array of weights, which can be used to estimate a marginal structural model. For correct (but
    conservative) confidence interval coverage, generalized estimation equations should be used. This can be done
    by using `statsmodels` `GEE`

    The formula for stabilized IPTW is

    .. math::

        \pi_i = \frac{\Pr(A=a)}{\Pr(A=a|L=l)}

    For unstabilized IPTW

    .. math::

        \pi_i = \frac{1}{\Pr(A=a|L=l)}

    SMR unstabilized weights for weighting to exposed (A=1)

    .. math::

        \pi_i &= 1 \;\;\text{       if}\;\; A = 1 \\
              &= \frac{\Pr(A=1|L=l)}{\Pr(A=0|L=l)} \;\;\text{if}\;\; A = 0

    For SMR weighted to the unexposed (A=0) the equation becomes

    .. math::

        \pi_i &= \frac{\Pr(A=0|L=l)}{\Pr(A=1|L=l)} \;\;\text{if}\;\; A=1 \\
              &= 1 \;\;\text{       if} \;\;A = 0

    Diagnostics are also available for generated IPTW. For a full list of diagnostics, see specific function
    documentation below. Additionally, review the references listed for an in-depth explanation

    Parameters
    ----------
    df : DataFrame
        Pandas dataframe object containing all variables of interest
    treatment : str
        Variable name of treatment variable of interest. Must be coded as binary. 1 should indicate treatment,
        while 0 indicates no treatment
    stabilized : bool, optional
        Whether to return stabilized or unstabilized weights. Default is stabilized weights (True)
    standardize : str, optional
        Who to standardize the estimate to. Options are the entire population, the exposed, or the unexposed. See
        Sato & Matsuyama Epidemiology (2003) for details on weighting to exposed/unexposed. Weighting to the
        exposed or unexposed is also referred to as SMR weighting. Options for standardization are:
        * 'population'    :   weight to entire population
        * 'exposed'       :   weight to exposed individuals
        * 'unexposed'     :   weight to unexposed individuals
    weights: str, optional
        Optional column for weights. If specified, a weighted regression model is instead used to estimate the inverse
        probability of treatment weights. This optional is useful in the following scenario; some confounder
        information is missing and IPMW was used to correct for missing data. IPTW should be estimated with the IPMW
        to standardize to the correct pseudo-population.

    Examples
    ---------
    Setting up environment

    >>>import matplotlib.pyplot as plt
    >>>from zepid import load_sample_data
    >>>from zepid.causal.ipw import IPTW
    >>>df = load_sample_data(False)

    Calculate stabilized IPTW

    >>>ipt = IPTW(df, treatment='art', stabilized=True)
    >>>ipt.regression_models('male + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>>ipt.fit()

    Calculate unstabilized IPTW weights

    >>>ipt = IPTW(df, treatment='art', stabilized=False)
    >>>ipt.regression_models('male + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>>ipt.fit()

    SMR weight to the exposed population

    >>>ipt = IPTW(df, treatment='art', stabilized=False, standardize='exposed')
    >>>ipt.regression_models('male + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>>ipt.fit()

    Diagnostics:

    >>>ipt.positivity()
    >>>print(ipt.standardized_mean_differences())

    >>>ipt.plot_boxplot()
    >>>plt.show()

    >>>ipt.plot_kde()
    >>>plt.show()

    >>>ipt.plot_love()
    >>>plt.show()

    References
    ----------
    Robins JM, Hernan MA, Brumback B. (2000). Marginal structural models and causal inference in epidemiology.

    Hernán MÁ, Brumback B, Robins JM. (2000). Marginal structural models to estimate the causal effect of zidovudine
    on the survival of HIV-positive men. Epidemiology, 561-570.

    Bodnar LM, Davidian M, Siega-Riz AM, Tsiatis AA. (2004). Marginal structural models for analyzing causal effects
    of time-dependent treatments: an application in perinatal epidemiology. American Journal of
    Epidemiology, 159(10), 926-934.

    Cole SR, Hernán MA. (2008). Constructing inverse probability weights for marginal structural models.
    American journal of epidemiology, 168(6), 656-664.

    Austin PC, Stuart EA. (2015). Moving towards best practice when using inverse probability of treatment
    weighting (IPTW) using the propensity score to estimate causal treatment effects in observational studies.
    Statistics in medicine, 34(28), 3661-3679.

    Sato T, Matsuyama Y. (2003). Marginal structural models as a tool for standardization. Epidemiology, 14(6), 680-686.

    Love T. (2004). Graphical Display of Covariate Balance. Presentation,
    See http://chrp.org/love/JSM2004RoundTableHandout. pdf, 1364.
    """
    def __init__(self, df, treatment, weights=None, stabilized=True, standardize='population'):
        self.denominator_model = None
        self.numerator_model = None
        self.__mdenom = None

        self.Weight = None
        self.ProbabilityNumerator = None
        self.ProbabilityDenominator = None

        self.df = df.copy()
        self.ex = treatment
        self._weight_ = weights
        self.stabilized = stabilized
        if standardize in ['population', 'exposed', 'unexposed']:
            self.standardize = standardize
        else:
            raise ValueError('Please specify one of the currently supported weighting schemes: ' +
                             'population, exposed, unexposed')

        self._pos_avg = None
        self._pos_min = None
        self._pos_max = None
        self._pos_sd = None

    def regression_models(self, model_denominator, model_numerator='1', print_results=True,
                          custom_model_denominator=None, custom_model_numerator=None):
        """Logistic regression model(s) for propensity score models. The model denominator must be specified for both
        stabilized and unstabilized weights. The optional argument 'model_numerator' allows specification of the
        stabilization factor for the weight numerator. By default model results are returned

        Parameters
        ------------
        model_denominator : str
            String listing variables to predict the exposure via `patsy` syntax. For example, `'var1 + var2 + var3'`.
            This is for the predicted probabilities of the denominator
        model_numerator : str, optional
            Optional string listing variables to predict the exposure, separated by +. Only used to calculate the
            numerator. Default ('1') calculates the overall probability of exposure. In general this is recommended. If
            confounding variables are included in the numerator, they would later need to be adjusted for in the faux
            marginal structural argument. Additionally, used for assessment of effect measure modification. Argument is
            also only used when calculating stabilized weights
        print_results : bool, optional
            Whether to print the model results from the regression models. Default is True
        custom_model_denominator : optional
            Input for a custom model that is used in place of the logit model. The model must have the
            `fit()` and  `predict()` attributes. Both `sklearn` and `supylearner` are supported as custom models. In the
            background, `IPTW` will fit the custom model and generate the predicted probablities
        custom_model_numerator : optional
            Input for a custom model that is used in place of the logit model. The model must have the
            `fit()` and  `predict()` attributes. Both `sklearn` and `supylearner` are supported as custom models. In the
            background, `IPTW` will fit the custom model and generate the predicted probablities

        Note
        ----
        If custom models are used, it is important that GEE is used to obtain the variance. Bootstrapped confidence
        intervals are incorrect with the usage of some machine learning models
        """
        # Calculating denominator probabilities
        self.__mdenom = model_denominator
        if custom_model_denominator is None:
            self.denominator_model = propensity_score(self.df, self.ex + ' ~ ' + model_denominator,
                                                      weights=self._weight_,
                                                      print_results=print_results)
            d = self.denominator_model.predict(self.df)
        else:
            data = patsy.dmatrix(model_denominator + ' - 1', self.df)
            try:
                fm = custom_model_denominator.fit(X=data, y=self.df[self.ex])
            except TypeError:
                raise TypeError("Currently custom_model must have the 'fit' function with arguments 'X', 'y'. This "
                                "covers both sklearn and supylearner. If there is a predictive model you would "
                                "like to use, please open an issue at https://github.com/pzivich/zepid and I "
                                "can work on adding support")
            if print_results and hasattr(fm, 'summarize'):
                fm.summarize()
            if hasattr(fm, 'predict_proba'):
                d = fm.predict_proba(data)[:, 1]
            elif hasattr(fm, 'predict'):
                d = fm.predict(data)
            else:
                raise ValueError("Currently custom_model must have 'predict' or 'predict_proba' attribute")
            self.denominator_model = fm

        self.df['__denom__'] = d

        # Calculating numerator probabilities (if stabilized)
        if self.stabilized is True:
            if custom_model_numerator is None:
                self.numerator_model = propensity_score(self.df, self.ex + ' ~ ' + model_numerator,
                                                        weights=self._weight_,
                                                        print_results=print_results)
                n = self.numerator_model.predict(self.df)

            else:
                data = patsy.dmatrix(model_numerator + ' - 1', self.df)
                try:
                    fm = custom_model_numerator.fit(X=data, y=self.df[self.ex])
                except TypeError:
                    raise TypeError("Currently custom_model must have the 'fit' function with arguments 'X', 'y'. This "
                                    "covers both sklearn and supylearner. If there is a predictive model you would "
                                    "like to use, please open an issue at https://github.com/pzivich/zepid and I "
                                    "can work on adding support")
                if print_results and hasattr(fm, 'summarize'):
                    fm.summarize()
                if hasattr(fm, 'predict_proba'):
                    n = fm.predict_proba(data)[:, 1]
                elif hasattr(fm, 'predict'):
                    n = fm.predict(data)
                else:
                    raise ValueError("Currently custom_model must have 'predict' or 'predict_proba' attribute")

        # If unstabilized, numerator is always 1
        else:
            if model_numerator != '1':
                raise ValueError('Argument for model_numerator is only used for stabilized=True')
            n = 1
        self.df['__numer__'] = n

    def fit(self):
        """Calculates the inverse probability of treatment weights from the calculate probabilities from the specified
        models / prediction algorithms.

        Returns
        -------
        IPTW class gains the Weight, ProbabilityDenominator, and ProbabilityNumerator attributed. Weights is a pandas
        Series containing the calculated IPTW.
        """
        if self.denominator_model is None:
            raise ValueError('No model has been fit to generated predicted probabilities')

        self.ProbabilityDenominator = self.df['__denom__']
        self.ProbabilityNumerator = self.df['__numer__']
        self.Weight = self._weight_calculator(self.df, denominator='__denom__', numerator='__numer__')

        if self._weight_ is not None:  # Multiplying calculate IPTW and specified weights
            warnings.warn("You specified the optional `weights` argument. A regression model weighted by the `weights` "
                          "column was used to generate IPTW. Additionally, the calculated IPTW were multiplied by the "
                          "specified `weights` column. Diagnostics for the weights include IPTW multiplied by the "
                          "`weights` column. Diagnostics for probabilities include only IPTW", UserWarning)
            self.Weight = self.Weight * self.df[self._weight_]

        self.df['_iptw_'] = self.Weight

    def plot_kde(self, measure='probability', bw_method='scott', fill=True, color_e='b', color_u='r'):
        """Generates a density plot that can be used to check whether positivity may be violated qualitatively. The
        kernel density used is SciPy's Gaussian kernel. Either Scott's Rule or Silverman's Rule can be implemented.
        Alternative option to the boxplot of probabilities

        Parameters
        ------------
        measure : str, optional
            Measure to plot. Options include either the probabilities or log-odds stratified by treatment received.
            Default is probabilities (measure='probability'). Log-odds can be requested via measure='logit'
        bw_method : str, optional
            Method used to estimate the bandwidth. Following SciPy, either 'scott' or 'silverman' are valid options
        fill : bool, optional
            Whether to color the area under the density curves. Default is true
        color_e : str, optional
            Color of the line/area for the treated group. Default is Blue
        color_u : str, optional
            Color of the line/area for the treated group. Default is Red

        Returns
        ---------------
        matplotlib axes
        """
        if measure == 'probability':
            x = np.linspace(0, 1, 10000)
            density_t = gaussian_kde(self.df.loc[self.df[self.ex] == 1]['__denom__'].dropna(),
                                     bw_method=bw_method)
            density_u = gaussian_kde(self.df.loc[self.df[self.ex] == 0]['__denom__'].dropna(),
                                     bw_method=bw_method)
        elif measure == 'logit':
            t = np.log(probability_to_odds(self.df.loc[self.df[self.ex] == 1]['__denom__'].dropna()))
            density_t = gaussian_kde(t, bw_method=bw_method)

            u = np.log(probability_to_odds(self.df.loc[self.df[self.ex] == 0]['__denom__'].dropna()))
            density_u = gaussian_kde(u, bw_method=bw_method)
            x = np.linspace(np.min((np.min(t), np.min(u))) - 1, np.max((np.max(t), np.max(u))) + 1, 10000)
        else:
            raise ValueError("Only plots of probabilities or log-odds are supported. Please specify either "
                             "'probability' or 'logit'")

        ax = plt.gca()
        if fill:
            ax.fill_between(x, density_t(x), color=color_e, alpha=0.2, label=None)
            ax.fill_between(x, density_u(x), color=color_u, alpha=0.2, label=None)
        ax.plot(x, density_t(x), color=color_e, label='Treat = 1')
        ax.plot(x, density_u(x), color=color_u, label='Treat = 0')
        if measure == 'probability':
            ax.set_xlabel('Probability')
        else:
            ax.set_xlabel('Log-Odds')
        ax.set_ylabel('Density')
        ax.legend()
        return ax

    def plot_boxplot(self, measure='probability'):
        """Generates a stratified boxplot that can be used to visually check whether positivity may be violated,
        qualitatively. Alternative option to the kernel density plot.

        Parameters
        ----------
        measure : str, optional
            Measure to plot. Options include either the probabilities or log-odds stratified by treatment received.
            Default is probabilities (measure='probability'). Log-odds can be requested via measure='logit'

        Returns
        -------------
        matplotlib axes
        """
        if measure == 'probability':
            boxes = (self.df.loc[self.df[self.ex] == 1]['__denom__'].dropna(),
                     self.df.loc[self.df[self.ex] == 0]['__denom__'].dropna())

        elif measure == 'logit':
            boxes = (np.log(probability_to_odds(self.df.loc[self.df[self.ex] == 1]['__denom__'].dropna())),
                     np.log(probability_to_odds(self.df.loc[self.df[self.ex] == 0]['__denom__'].dropna())))
        else:
            raise ValueError("Only plots of probabilities or log-odds are supported. Please specify either "
                             "'probability' or 'logit")

        labs = ['Treat = 1', 'Treat = 0']
        meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='black')
        ax = plt.gca()
        ax.boxplot(boxes, labels=labs, meanprops=meanpointprops, showmeans=True)
        if measure == 'probability':
            ax.set_ylabel('Probability')
        else:
            ax.set_ylabel('Log-Odds')
        return ax

    def positivity(self, decimal=3):
        """Use this to assess whether positivity is a valid assumption. For stabilized weights, the mean weight should
        be approximately 1. For unstabilized weights, the mean weight should be approximately 2. If there are extreme
        outliers, this may indicate problems with the calculated weights

        Parameters
        --------------
        decimal : int, optional
            Number of decimal places to display. Default is three

        Returns
        --------------
        None
            Prints the positivity results to the console but does not return any objects
        """
        self._pos_avg = float(np.mean(self.df['_iptw_'].dropna()))
        self._pos_max = np.max(self.df['_iptw_'].dropna())
        self._pos_min = np.min(self.df['_iptw_'].dropna())
        self._pos_sd = float(np.std(self.df['_iptw_'].dropna()))
        print('======================================================================')
        print('          Inverse Probability of Treatment Weight Diagnostics')
        print('======================================================================')
        print('If the mean of the weights is far from either the min or max, this may\n indicate the model is '
              'incorrect or positivity is violated')
        print('Average weight should be')
        print('\t1.0 for stabilized')
        print('\t2.0 for unstabilized')
        print('Standard deviation can help in IPTW model selection')
        print('----------------------------------------------------------------------')
        print('Mean weight:           ', round(self._pos_avg, decimal))
        print('Standard Deviation:    ', round(self._pos_sd, decimal))
        print('Minimum weight:        ', round(self._pos_min, decimal))
        print('Maximum weight:        ', round(self._pos_max, decimal))
        print('======================================================================')

    def plot_love(self, color_unweighted='r', color_weighted='b', shape_unweighted='o', shape_weighted='o'):
        """Generates a Love-plot to detail covariate balance based on the IPTW weights. Further details on the usage of
        this plot are available in Austin PC & Stuart EA 2015 https://onlinelibrary.wiley.com/doi/full/10.1002/sim.6607

        The Love plot generates a dashed line at standardized mean difference of 0.10. Ideally, weighted SMD are below
        this level. Below 0.20 may also be sufficient. Variables above this level may be unbalanced despite the
        weighting procedure. Different functional forms (or approaches like machine learning) may be worth considering

        Returns
        -------
        axes
            Matplotlib axes of the Love plot
        """
        to_plot = self.standardized_mean_differences()
        to_plot['smd_w'] = np.absolute(to_plot['smd_w'])
        to_plot['smd_u'] = np.absolute(to_plot['smd_u'])
        to_plot = to_plot.sort_values(by='smd_u', ascending=True).reset_index(drop=True)

        # Generate plot
        ax = plt.gca()
        ax.plot(to_plot.smd_u, to_plot.index, shape_unweighted, c=color_unweighted, label='Unweighted')
        ax.plot(to_plot.smd_w, to_plot.index, shape_weighted, c=color_weighted, label='Weighted')
        ax.set_xlim([0, np.max([np.max(to_plot['smd_w']), np.max(to_plot['smd_u'])]) + 0.5])
        ax.set_xlabel('Absolute Standardized Difference')
        ax.axvline(0.1, color='gray')
        ax.set_yticks([i for i in range(to_plot.shape[0])])
        ax.set_yticklabels(to_plot['labels'])
        ax.legend()
        return ax

    def standardized_mean_differences(self):
        """Calculates the standardized mean differences for all variables. Default calculates the standardized mean
        difference for all variables included in the IPTW denominator

        Returns
        -------
        DataFrame
            Returns pandas DataFrame of calculated standardized mean differences. Columns are labels (variables labels),
            smd_u (unweighted standardized difference), and smd_w (weighted standardized difference)
        """
        vars = patsy.dmatrix(self.__mdenom + ' - 1', self.df, return_type='dataframe')
        w_diff = []
        u_diff = []
        vlabel = []

        # Pull out list of terms and the corresponding dataframe slice(s)
        term_dict = vars.design_info.term_name_slices

        # Looping through the terms
        for term in vars.design_info.terms:
            # Adding term labels
            vlabel.append(term.name())

            # Pulling out data corresponding to term
            chunk = term_dict[term.name()]
            v = vars.iloc[:, chunk].copy()

            # Detecting variable type
            if v.shape[1] != 1:
                vtype = 'categorical'
            elif np.all(v.dropna().isin([0, 1])):
                vtype = 'binary'
            else:
                vtype = 'continuous'

            # calculate the absolute standardized difference
            dat = pd.concat([v, self.df[[self.ex, '_iptw_']]], axis=1)
            wsmd = self._standardized_difference(variable=dat, var_type=vtype, weighted=True)
            w_diff.append(wsmd)
            usmd = self._standardized_difference(variable=dat, var_type=vtype, weighted=False)
            u_diff.append(usmd)

        # Setting up DataFrame to return with calculated differences
        s = pd.DataFrame()
        s['labels'] = vlabel
        s['smd_w'] = w_diff
        s['smd_u'] = u_diff
        return s

    def _standardized_difference(self, variable, var_type, weighted=True):
        """Background function to calculate the standardized mean difference between the treat and untreated for a
        specified variable. Useful for checking whether a confounder was balanced between the two treatment groups
        by the specified IPTW model SMD based on: Austin PC 2011; https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3144483/

        Parameters
        ---------------
        variable : str, list
            Label for variable to calculate the standardized difference. If categorical variables, it should be a list
            of variable labels
        var_type : str
            Variable type. Options are 'binary' 'continuous' or 'categorical'. For categorical variable should be a
            list of columns labels
        weighted : bool, optional
            Whether to return the weighted standardized mean difference or the unweighted. Default is to return the
            weighted.

        Returns
        --------------
        None
            Prints the positivity results to the console but does not return any objects
        """
        # Pulling out relevant data
        dft = variable.loc[(variable[self.ex] == 1) & (variable['_iptw_'].notnull())].copy()
        dfn = variable.loc[(variable[self.ex] == 0) & (variable['_iptw_'].notnull())].copy()
        # removing self.ex and 'iptw' from vars to calculate for
        vcols = list(variable.columns)
        vcols.remove(self.ex)
        vcols.remove('_iptw_')

        if var_type == 'binary':
            if weighted:
                dwt = DescrStatsW(dft[vcols], weights=dft['_iptw_'])
                wt = dwt.mean
                dwn = DescrStatsW(dfn[vcols], weights=dfn['_iptw_'])
                wn = dwn.mean
            else:
                wt = np.mean(dft[vcols].dropna(), axis=0)
                wn = np.mean(dfn[vcols].dropna(), axis=0)
            return float((wt - wn) / np.sqrt((wt*(1 - wt) + wn*(1 - wn))/2))

        elif var_type == 'continuous':
            if weighted:
                dwt = DescrStatsW(dft[vcols], weights=dft['_iptw_'], ddof=1)
                wmt = dwt.mean
                wst = dwt.std
                dwn = DescrStatsW(dfn[vcols], weights=dfn['_iptw_'], ddof=1)
                wmn = dwn.mean
                wsn = dwn.std
            else:
                dwt = DescrStatsW(dft[vcols], ddof=1)
                wmt = dwt.mean
                wst = dwt.std
                dwn = DescrStatsW(dfn[vcols], ddof=1)
                wmn = dwn.mean
                wsn = dwn.std
            return float((wmt - wmn) / np.sqrt((wst**2 + wsn**2)/2))

        elif var_type == 'categorical':
            if weighted:
                wt = np.average(dft[vcols], weights=dft['_iptw_'], axis=0)
                wn = np.average(dfn[vcols], weights=dfn['_iptw_'], axis=0)
            else:
                wt = np.average(dft[vcols], axis=0)
                wn = np.mean(dfn[vcols], axis=0)

            t_c = wt - wn
            s_inv = np.linalg.inv(self._categorical_cov(treated=wt, untreated=wn))
            return float(np.sqrt(np.dot(np.transpose(t_c[1:]), np.dot(s_inv, t_c[1:]))))

        else:
            raise ValueError('Not supported')

    def _weight_calculator(self, df, denominator, numerator):
        """Calculates the IPTW based on the predicted probabilities and the specified group to standardize to in the
        background for the fit() function. Not intended to be used by users

        df is the dataframe, denominator is the string indicating the column of Pr, numerator is the string indicating
        the column of Pr
        """
        if self.stabilized:  # Stabilized weights
            if self.standardize == 'population':
                df['w'] = np.where(df[self.ex] == 1, (df[numerator] / df[denominator]),
                                   ((1 - df[numerator]) / (1 - df[denominator])))
                df['w'] = np.where(df[self.ex].isna(), np.nan, df['w'])
            # Stabilizing to exposed (compares all exposed if they were exposed versus unexposed)
            elif self.standardize == 'exposed':
                df['w'] = np.where(df[self.ex] == 1, 1,
                                   ((df[denominator] / (1 - df[denominator])) * ((1 - df[numerator]) /
                                                                                 df[numerator])))
                df['w'] = np.where(df[self.ex].isna(), np.nan, df['w'])
            # Stabilizing to unexposed (compares all unexposed if they were exposed versus unexposed)
            else:
                df['w'] = np.where(df[self.ex] == 1,
                                   (((1 - df[denominator]) / df[denominator]) * (df[numerator] /
                                                                                 (1 - df[numerator]))),
                                   1)
                df['w'] = np.where(df[self.ex].isna(), np.nan, df['w'])

        else:  # Unstabilized weights
            if self.standardize == 'population':
                df['w'] = np.where(df[self.ex] == 1, 1 / df[denominator], 1 / (1 - df[denominator]))
                df['w'] = np.where(df[self.ex].isna(), np.nan, df['w'])
            # Stabilizing to exposed (compares all exposed if they were exposed versus unexposed)
            elif self.standardize == 'exposed':
                df['w'] = np.where(df[self.ex] == 1, 1, (df[denominator] / (1 - df[denominator])))
                df['w'] = np.where(df[self.ex].isna(), np.nan, df['w'])
            # Stabilizing to unexposed (compares all unexposed if they were exposed versus unexposed)
            else:
                df['w'] = np.where(df[self.ex] == 1, ((1 - df[denominator]) / df[denominator]), 1)
                df['w'] = np.where(df[self.ex].isna(), np.nan, df['w'])
        return df['w']

    @staticmethod
    def _categorical_cov(treated, untreated):
        """Turns out, pandas and numpy don't have the correct covariance matrix I need for categorical variables.
        The covariance matrix is defined as

        S = [S_{kl}] = (P_{1k}*(1-P_{1k}) + P_{2k}*(1-P{2k})) / 2     if k == l
                       (P_{1k}*P_{1l} + P_{2k}*P_{2l}) / 2            if k != l

        Returns the calculate covariance matrix for categorical variables
        """
        cv2 = []
        for i, v in enumerate(treated):
            cv1 = []
            if i == 0:
                pass
            else:
                for j, w in enumerate(untreated):
                    if j == 0:
                        pass
                    elif i == j:
                        cv1.append((v * (1 - v) + w * (1 - w)) / 2)
                    else:
                        cv1.append((treated[i] * treated[j] + untreated[i] * untreated[j]) / -2)
                cv2.append(cv1)

        return np.array(cv2)

