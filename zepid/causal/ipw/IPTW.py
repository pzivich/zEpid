import warnings
import patsy
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import DomainWarning
import matplotlib.pyplot as plt

from zepid.calc.utils import probability_bounds
from zepid.causal.utils import (check_input_data, propensity_score, plot_boxplot, plot_kde, plot_love,
                                stochastic_check_conditional, standardized_mean_differences, positivity,
                                iptw_calculator)


class IPTW:
    r"""Calculates inverse probability of treatment weights. Both stabilized or unstabilized weights are implemented.
    By default, stabilized weights are stabilized by the prevalence of the treatment in the population. `IPTW` will
    also now fit the marginal structural model and estimate inverse probability of censoring weights if requested.
    Confidence intervals are calculated using robust standard errors.

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
        Variable name of treatment of interest. Must be coded as binary
    outcome : str
        Variable name of outcome of interest. Can be either binary or continuous
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

    >>> import matplotlib.pyplot as plt
    >>> from zepid import load_sample_data, spline
    >>> from zepid.causal.ipw import IPTW
    >>> df = load_sample_data(timevary=False).drop(columns=['cd4_wk45'])
    >>> df[['cd4_rs1','cd4_rs2']] = spline(df, 'cd40', n_knots=3, term=2, restricted=True)
    >>> df[['age_rs1','age_rs2']] = spline(df, 'age0', n_knots=3, term=2, restricted=True)

    Calculate stabilized IPTW

    >>> ipt = IPTW(df, treatment='art', outcome='dead')
    >>> ipt.treatment_model('male + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>> ipt.marginal_structural_model('art')
    >>> ipt.fit()
    >>> ipt.summary()

    Diagnostics:

    >>> ipt.run_diagnostics()

    Calculate unstabilized IPTW weights

    >>> ipt = IPTW(df, treatment='art', outcome='dead')
    >>> ipt.treatment_model('male + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', stabilized=False)
    >>> ipt.marginal_structural_model('art')
    >>> ipt.fit()
    >>> ipt.summary()

    Calculate SMR weight to the exposed population

    >>> ipt = IPTW(df, treatment='art', outcome='dead', standardize='exposed')
    >>> ipt.treatment_model('male + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>> ipt.marginal_structural_model('art')
    >>> ipt.fit()
    >>> ipt.summary()

    Stabilized IPTW with IPCW

    >>> ipt = IPTW(df, treatment='art', outcome='dead')
    >>> ipt.treatment_model('male + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>> ipt.missing_model('art + male + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>> ipt.marginal_structural_model('art')
    >>> ipt.fit()
    >>> ipt.summary()

    Stabilized IPTW with effect measure modifier

    >>> ipt = IPTW(df, treatment='art', outcome='dead')
    >>> ipt.treatment_model('male + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0', model_numerator='male')
    >>> ipt.marginal_structural_model('art + male + art:male')
    >>> ipt.fit()
    >>> ipt.summary()

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
    def __init__(self, df, treatment, outcome, weights=None, standardize='population'):
        self.treatment = treatment
        self.outcome = outcome
        self._missing_indicator = '__missing_indicator__'
        self.df, self._miss_flag, self._continuous_outcome = check_input_data(data=df,
                                                                              exposure=treatment,
                                                                              outcome=outcome,
                                                                              estimator="IPTW",
                                                                              drop_censoring=False,
                                                                              drop_missing=True,
                                                                              binary_exposure_only=True)
        # TODO add detection of continuous treatments

        self.average_treatment_effect = None
        self.risk_difference = None
        self.risk_ratio = None
        self.odds_ratio = None

        if standardize in ['population', 'exposed', 'unexposed']:
            self.standardize = standardize
            if standardize in ['exposed', 'unexposed']:
                warnings.warn("For the ATT and the ATU, confidence intervals calculated using the robust-variance "
                              "approach (what is currently done in zEpid) may underestimate the variance. Therefore "
                              "when requesting the ATT or the ATU, it is recommended to use bootstrapped confidence "
                              "intervals instead.", UserWarning)
        else:
            raise ValueError('Please specify one of the currently supported weighting schemes: ' +
                             'population, exposed, unexposed')

        self._weight_ = weights
        self.iptw = None
        self.ipmw = None
        self.ms_model = None
        self.__mdenom = None
        self._fit_missing_ = False
        self._miss_model = None
        self._continuous_y_type = None
        self._pos_avg = None
        self._pos_min = None
        self._pos_max = None
        self._pos_sd = None

    def treatment_model(self, model_denominator, model_numerator='1', stabilized=True, bound=False, print_results=True):
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
        stabilized : bool, optional
            Whether to return stabilized or unstabilized weights. Default is stabilized weights (True)
        bound : float, list, optional
            Value between 0,1 to truncate predicted probabilities. Helps to avoid near positivity violations.
            Specifying this argument can improve finite sample performance for random positivity violations. However,
            inference becomes limited to the restricted population. Default is False, meaning no truncation of
            predicted probabilities occurs. Providing a single float assumes symmetric trunctation. A collection of
            floats can be provided for asymmetric trunctation
        print_results : bool, optional
            Whether to print the model results from the regression models. Default is True
        """
        # Calculating denominator probabilities
        self.__mdenom = model_denominator
        self.df['__denom__'], self.df['__numer__'], self.iptw = iptw_calculator(df=self.df,
                                                                                treatment=self.treatment,
                                                                                model_denom=model_denominator,
                                                                                model_numer=model_numerator,
                                                                                weight=self._weight_,
                                                                                stabilized=stabilized,
                                                                                standardize=self.standardize,
                                                                                bound=bound,
                                                                                print_results=print_results)

    def missing_model(self, model_denominator, model_numerator=None, stabilized=True, bound=False, print_results=True):
        """Estimation of Pr(M=0|A=a,L), which is the missing data mechanism for the outcome. The corresponding
        observation probabilities are used to account for informative censoring by observed variables. The missing_model
        only accounts for missing outcome data.

        The inverse probability weights calculated by this function account for informative censoring (missing data on
        the outcome) by observed variables. The parametric model should be sufficiently flexible to capture any
        interaction terms and functional forms of continuous variables

        Note
        ----
        The treatment variable should be included in the model

        Parameters
        ----------
        model_denominator: str
            String listing variables predicting missingness of outcomes via `patsy` syntax. For example, `
            'var1 + var2 + var3'. This is for the predicted probabilities of the denominator
        model_numerator : str, optional
            Optional string listing variables to predict the exposure, separated by +. Only used to calculate the
            numerator. Default (None) calculates the probability of censoring by treatment only. In general this is
            recommended. If assessing effect modifcation, this variable should be included in the numerator as well.
            Argument is only used when calculating stabilized weights
        stabilized : bool, optional
            Whether to use stabilized inverse probability of censoring weights
        bound : float, list, optional
            Value between 0,1 to truncate predicted probabilities. Helps to avoid near positivity violations.
            Specifying this argument can improve finite sample performance for random positivity violations. However,
            inference becomes limited to the restricted population. Default is False, meaning no truncation of
            predicted probabilities occurs. Providing a single float assumes symmetric trunctation. A collection of
            floats can be provided for asymmetric trunctation
        print_results: bool, optional
        """
        # Error if no missing outcome data
        if not self._miss_flag:
            raise ValueError("No missing outcome data is present in the data set")

        # Warning if exposure is not included in the missingness of outcome model
        if self.treatment not in model_denominator:
            warnings.warn("For the specified missing outcome model, the exposure variable should be included in the "
                          "model", UserWarning)

        self._miss_model = self._missing_indicator + ' ~ ' + model_denominator
        fitmodel = propensity_score(self.df, self._miss_model, print_results=print_results)

        if stabilized:
            if model_numerator is None:
                mnum = self.treatment
            else:
                mnum = model_numerator
            numerator_model = propensity_score(self.df, self._missing_indicator + ' ~ ' + mnum,
                                               weights=self._weight_,
                                               print_results=print_results)
            n = numerator_model.predict(self.df)
        else:
            n = 1

        if bound:  # Bounding predicted probabilities if requested
            d = probability_bounds(fitmodel.predict(self.df), bounds=bound)
        else:
            d = fitmodel.predict(self.df)

        self.ipmw = np.where(self.df[self._missing_indicator] == 1, n / d, np.nan)
        self._fit_missing_ = True

    def marginal_structural_model(self, model):
        """Specify the marginal structural model to estimate using the inverse probability of treatment weights. The
        treatment variable should always be included in the marginal structural model. In addition to the treatment,
        effect measure modification by variables can be assessed. See the main documentation for an example of how
        to specify a effect measure modifier.

        Note
        ----
        If assessing modification, be sure to use stabilized IPTW and include the modifier in the `model_numerator`

        Parameters
        ----------
        model : str
            The specified marginal structural model to fit.
        """
        if self.treatment not in model:
            raise ValueError("The treatment variable must be specified in the marginal structural model")

        self.ms_model = model

    def fit(self, continuous_distribution='gaussian'):
        """Fit the specified marginal structural model using the calculated inverse probability of treatment weights.
        """
        if self.__mdenom is None:
            raise ValueError('No model has been fit to generated predicted probabilities')
        if self.ms_model is None:
            raise ValueError('No marginal structural model has been specified')
        if self._miss_flag and not self._fit_missing_:
            warnings.warn("All missing outcome data is assumed to be missing completely at random. To relax this "
                          "assumption to outcome data is missing at random please use the `missing_model()` "
                          "function", UserWarning)

        ind = sm.cov_struct.Independence()
        full_msm = self.outcome + ' ~ ' + self.ms_model

        df = self.df.copy()
        if self.ipmw is None:
            if self._weight_ is None:
                df['_ipfw_'] = self.iptw
            else:
                df['_ipfw_'] = self.iptw * self.df[self._weight_]
        else:
            if self._weight_ is None:
                df['_ipfw_'] = self.iptw * self.ipmw
            else:
                df['_ipfw_'] = self.iptw * self.ipmw * self.df[self._weight_]
        df = df.dropna()

        if self._continuous_outcome:
            if (continuous_distribution == 'gaussian') or (continuous_distribution == 'normal'):
                f = sm.families.family.Gaussian()
            elif continuous_distribution == 'poisson':
                f = sm.families.family.Poisson()
            else:
                raise ValueError("Only 'gaussian' and 'poisson' distributions are supported")
            self._continuous_y_type = continuous_distribution
            fm = smf.gee(full_msm, df.index, df,
                         cov_struct=ind, family=f, weights=df['_ipfw_']).fit()
            self.average_treatment_effect = pd.DataFrame()
            self.average_treatment_effect['labels'] = np.asarray(fm.params.index)
            self.average_treatment_effect.set_index(keys=['labels'], inplace=True)
            self.average_treatment_effect['ATE'] = np.asarray(fm.params)
            self.average_treatment_effect['SE(ATE)'] = np.asarray(fm.bse)
            self.average_treatment_effect['95%LCL'] = np.asarray(fm.conf_int()[0])
            self.average_treatment_effect['95%UCL'] = np.asarray(fm.conf_int()[1])

        else:
            # Ignoring DomainWarnings from statsmodels
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DomainWarning)

                # Estimating Risk Difference
                f = sm.families.family.Binomial(sm.families.links.identity())
                fm = smf.gee(full_msm, df.index, df,
                             cov_struct=ind, family=f, weights=df['_ipfw_']).fit()
                self.risk_difference = pd.DataFrame()
                self.risk_difference['labels'] = np.asarray(fm.params.index)
                self.risk_difference.set_index(keys=['labels'], inplace=True)
                self.risk_difference['RD'] = np.asarray(fm.params)
                self.risk_difference['SE(RD)'] = np.asarray(fm.bse)
                self.risk_difference['95%LCL'] = np.asarray(fm.conf_int()[0])
                self.risk_difference['95%UCL'] = np.asarray(fm.conf_int()[1])

                # Estimating Risk Ratio
                f = sm.families.family.Binomial(sm.families.links.log())
                fm = smf.gee(full_msm, df.index, df,
                             cov_struct=ind, family=f, weights=df['_ipfw_']).fit()
                self.risk_ratio = pd.DataFrame()
                self.risk_ratio['labels'] = np.asarray(fm.params.index)
                self.risk_ratio.set_index(keys=['labels'], inplace=True)
                self.risk_ratio['RR'] = np.exp(np.asarray(fm.params))
                self.risk_ratio['SE(log(RR))'] = np.asarray(fm.bse)
                self.risk_ratio['95%LCL'] = np.exp(np.asarray(fm.conf_int()[0]))
                self.risk_ratio['95%UCL'] = np.exp(np.asarray(fm.conf_int()[1]))

                # Estimating Odds Ratio
                f = sm.families.family.Binomial()
                fm = smf.gee(full_msm, df.index, df,
                             cov_struct=ind, family=f, weights=df['_ipfw_']).fit()
                self.odds_ratio = pd.DataFrame()
                self.odds_ratio['labels'] = np.asarray(fm.params.index)
                self.odds_ratio.set_index(keys=['labels'], inplace=True)
                self.odds_ratio['OR'] = np.exp(np.asarray(fm.params))
                self.odds_ratio['SE(log(OR))'] = np.asarray(fm.bse)
                self.odds_ratio['95%LCL'] = np.exp(np.asarray(fm.conf_int()[0]))
                self.odds_ratio['95%UCL'] = np.exp(np.asarray(fm.conf_int()[1]))

    def summary(self, decimal=3):
        """Prints a summary of the results for the IPTW estimator.

        Parameters
        ----------
        decimal : int, optional
            Number of decimal places to display in the result. Default is 3
        """
        print('======================================================================')
        print('              Inverse Probability of Treatment Weights                ')
        print('======================================================================')
        fmt = 'Treatment:        {:<15} No. Observations:     {:<20}'
        print(fmt.format(self.treatment, self.df.shape[0]))

        fmt = 'Outcome:          {:<15} No. Missing Outcome:  {:<20}'
        print(fmt.format(self.outcome, np.sum(self.df[self.outcome].isnull())))

        fmt = 'g-Model:          {:<15} Missing Model:        {:<20}'
        if self._fit_missing_:
            m = 'Logistic'
        else:
            m = 'None'
        print(fmt.format('Logistic', m))
        print('======================================================================')

        if self._continuous_outcome:
            print('Average Treatment Effect')
            print('----------------------------------------------------------------------')
            print(np.round(self.average_treatment_effect, decimals=decimal))
        else:
            print('Risk Difference')
            print('----------------------------------------------------------------------')
            print(np.round(self.risk_difference, decimals=decimal))
            print('----------------------------------------------------------------------')
            print('Risk Ratio')
            print(np.round(self.risk_ratio, decimals=decimal))
            print('----------------------------------------------------------------------')
            print('Odds Ratio')
            print(np.round(self.odds_ratio, decimals=decimal))
        print('======================================================================')

    def run_diagnostics(self, iptw_only=True):
        """Run all currently implemented diagnostics for inverse probability of treatment weights available. Each
        diagnostic can be called individually for further optional specifications. `run_weight_diagnostics` will
        provide results for all implemented diagnostics for ease of the user. For presentation of results, I recommend
        calling each function individually and utilizing the optional parameters

        Note
        ----
        The plot presented cannot be edited. To edit the plots, call `plot_kde` or `plot_love` directly. Those
        functions return an axes object

        Parameters
        ----------
        iptw_only : bool, optional
            Whether to include weights generated from missing_model in the diagnostics. Default is True, which runs the
            diagnostics for IPTW only

        Returns
        -------
        None
        """
        self.positivity(iptw_only=iptw_only)

        print('\n======================================================================')
        print('                Standardized Mean Differences')
        print('======================================================================')
        print(self.standardized_mean_differences(iptw_only=iptw_only).set_index(keys='labels'))
        print('======================================================================')

        plt.figure(figsize=[9, 4])
        plt.subplot(122)
        self.plot_kde()
        plt.title("Kernel Density of Propensity Scores")

        plt.subplot(121)
        self.plot_love(iptw_only=iptw_only)
        plt.title("Love Plot")
        plt.tight_layout()
        plt.show()

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
        ax = plot_kde(df=self.df, treatment=self.treatment, probability='__denom__', measure=measure,
                      bw_method=bw_method, fill=fill, color_e=color_e, color_u=color_u)
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
        ax = plot_boxplot(df=self.df, treatment=self.treatment, probability='__denom__', measure=measure)
        return ax

    def positivity(self, decimal=3, iptw_only=True):
        """Use this to assess whether positivity is a valid assumption. For stabilized weights, the mean weight should
        be approximately 1. For unstabilized weights, the mean weight should be approximately 2. If there are extreme
        outliers, this may indicate problems with the calculated weights

        Parameters
        --------------
        decimal : int, optional
            Number of decimal places to display. Default is three
        iptw_only : bool, optional
            Whether the diagnostic should be run on IPTW only or include weights from the missing_model. Default is
            True, which returns IPTW only

        Returns
        --------------
        None
        """
        if iptw_only:
            df = self.df
            df['_ipfw_'] = self.iptw
        else:
            df = self.df
            df['_ipfw_'] = self.iptw * self.ipmw

        self._pos_avg, self._pos_sd, self._pos_min, self._pos_max = positivity(df=df, weights='_ipfw_')
        print('======================================================================')
        print('                     Weight Positivity Diagnostics')
        print('======================================================================')
        print('If the mean of the weights is far from either the min or max, this may\n '
              'indicate the model is incorrect or positivity is violated')
        print('Average weight should be')
        print('\t1.0 for stabilized')
        print('\t2.0 for unstabilized')
        print('----------------------------------------------------------------------')
        print('Mean weight:           ', round(self._pos_avg, decimal))
        print('Standard Deviation:    ', round(self._pos_sd, decimal))
        print('Minimum weight:        ', round(self._pos_min, decimal))
        print('Maximum weight:        ', round(self._pos_max, decimal))
        print('======================================================================')

    def standardized_mean_differences(self, iptw_only=True):
        """Calculates the standardized mean differences for all variables. Default calculates the standardized mean
        difference for all variables included in the IPTW denominator

        Parameters
        ----------
        iptw_only : bool, optional
            Whether the diagnostic should be run on IPTW only or include weights from the missing_model. Default is
            True, which returns IPTW only

        Returns
        -------
        DataFrame
            Returns pandas DataFrame of calculated standardized mean differences. Columns are labels (variables labels),
            smd_u (unweighted standardized difference), and smd_w (weighted standardized difference)
        """
        if iptw_only:
            df = self.df
            df['_ipfw_'] = self.iptw
        else:
            df = self.df
            df['_ipfw_'] = self.iptw * self.ipmw

        s = standardized_mean_differences(df=df, treatment=self.treatment, weight='_ipfw_', formula=self.__mdenom)
        return s

    def plot_love(self, color_unweighted='r', color_weighted='b', shape_unweighted='o', shape_weighted='o',
                  iptw_only=True):
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
        iptw_only : bool, optional
            Whether the diagnostic should be run on IPTW only or the weights multiplied together. Default is IPTW only

        Returns
        -------
        matplotlib axes
        """
        if iptw_only:
            df = self.df
            df['_ipfw_'] = self.iptw
        else:
            df = self.df
            df['_ipfw_'] = self.iptw * self.ipmw

        ax = plot_love(df=self.df, treatment=self.treatment, weight='_ipfw_', formula=self.__mdenom,
                       color_unweighted=color_unweighted, color_weighted=color_weighted,
                       shape_unweighted=shape_unweighted, shape_weighted=shape_weighted)
        return ax


class StochasticIPTW:
    r"""Calculates the IPTW estimate for stochastic treatment plans. `StochasticIPTW` will returns the estimated
    marginal outcome for that treatment plan. This is distinct from `IPTW`, which returns an array of weights. For
    confidence intervals, a bootstrapping procedure needs to be used.

    The formula for IPTW for a stochastic treatment is

    .. math::

        \pi_i = \frac{\overline{\Pr}(A=a|L)}{\Pr(A=a|L)}

    where :math:`\bar{\Pr}` is the new probability of treatment under the proposed stochastic treatment. This
    probability can be unconditional (everyone treated at some constant percent) or it can be conditional on observed
    covariates. The denominator is the same estimated probability of treatment in the standard IPTW formula. Basically,
    we are manipulating how many the treated individuals represent in a new pseudo-population

    Note
    ----
    `StochasticIPTW` estimates the marginal outcome at a specified treatment distribution. Unlike IPTW, it does not
    immediately result in a comparison between two treatment levels (i.e. we are not estimating a marginal structural
    model in this case). For a comparison, two different version would need to be specified.

    `StochasticIPTW` does not contain the diagnostics that are contained within `IPTW`. This IPTW estimation approach
    makes weaker assumptions regarding positivity and causal consistency.

    Parameters
    ----------
    df : DataFrame
        Pandas dataframe object containing all variables of interest
    treatment : str
        Variable name of treatment variable of interest. Must be coded as binary. 1 should indicate treatment,
        while 0 indicates no treatment
    outcome: str
        Variable name of outcome variable of interest. May be binary or continuous.
    weights: str, optional
        Optional column for weights. If specified, a weighted regression model is instead used to estimate the inverse
        probability of treatment weights. This optional is useful in the following scenario; some confounder
        information is missing and IPMW was used to correct for missing data. IPTW should be estimated with the IPMW
        to standardize to the correct pseudo-population.

    Examples
    --------
    Loading data

    >>> from zepid import load_sample_data, spline
    >>> from zepid.causal.ipw import StochasticIPTW
    >>> df = load_sample_data(timevary=False).drop(columns=['cd4_wk45'])
    >>> df[['cd4_rs1','cd4_rs2']] = spline(df,'cd40',n_knots=3,term=2,restricted=True)
    >>> df[['age_rs1','age_rs2']] = spline(df,'age0',n_knots=3,term=2,restricted=True)

    Estimating marginal outcome under treatment plan where 80% are randomly treated

    >>> ipw = StochasticIPTW(df, treatment='art', outcome='dead')
    >>> ipw.treatment_model('male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
    >>> ipw.fit(p=0.8)
    >>> ipw.summary()

    Estimating marginal outcome under treatment plan where 10% are randomly treated

    >>> ipw.fit(p=0.1)
    >>> ipw.summary()

    Estimating marginal outcome under treatment plan where 75% of men are treated and 90% of women

    >>> ipw.fit(p=[0.75, 0.90], conditional=["df['male']==1", "df['male']==0"])
    >>> ipw.summary()

    References
    ----------
    Muñoz ID & van der Laan M. (2012). Population intervention causal effects based on stochastic interventions.
    Biometrics, 68(2), 541-549.
    """
    def __init__(self, df, treatment, outcome, weights=None):
        self.treatment = treatment
        self.outcome = outcome
        self._missing_indicator = '__missing_indicator__'
        self.df, self._miss_flag, self._continuous_outcome = check_input_data(data=df,
                                                                              exposure=treatment,
                                                                              outcome=outcome,
                                                                              estimator="StochasticIPTW",
                                                                              drop_censoring=True,
                                                                              drop_missing=True,
                                                                              binary_exposure_only=True)
        self.weights = weights
        self.marginal_outcome = np.nan
        self._pdenom_ = None

    def treatment_model(self, model, print_results=True):
        r"""Specify the parametric regression model for the observed treatment conditional on the sufficient adjustment
        set. This model estimates the following component of the stochastic IPTW weights

        .. math::

            \widehat{\Pr}(A=a|L)

        Parameters
        ----------
        model : str
            String listing variables to predict the exposure via `patsy` syntax. For example, `'var1 + var2 + var3'`
        print_results : bool, optional
            Whether to print the model results from the regression models. Default is True
        """
        # Calculating denominator probabilities
        denominator_model = propensity_score(self.df, self.treatment + ' ~ ' + model,
                                             weights=self.weights,
                                             print_results=print_results)
        self._pdenom_ = denominator_model.predict(self.df)

    def fit(self, p, conditional=None):
        """Estimates the mean outcome under the specified stochastic treatment plan.As currently implemented, `p`
        percent of the population is randomly treated. Unlike the stochastic g-formula, we only need to estimate the
        marginal outcome once.

        Parameters
        ----------
        p : float, list
            Percent of the population to randomly treat
        conditional : list, optional
            Exclusive conditions to place on the data set for treatment percents. If specified, must match the length
            of the list of probabilities in `p`. For specification of conditions, the data set should be referred to
            as `df` in strings. For example, `"df['male']==1"` restricts that probability to only males

        Returns
        -------
        marginal_outcome
            Gains marginal outcome attribute. Summary function can also be called afterwards
        """
        p = np.array(p)

        if self._pdenom_ is None:
            raise ValueError("The treatment_model() function must be specified before the fit() function")
        if np.any(p > 1) or np.any(p < 0):
            raise ValueError("All specified treatment probabilities must be between 0 and 1")
        if conditional is not None:
            if len(p) != len(conditional):
                raise ValueError("'p' and 'conditional' must be the same length")

        df = self.df.copy()

        if conditional is None:
            df['_numer_'] = np.where(df[self.treatment] == 1, p, 1 - p)
        else:
            stochastic_check_conditional(df=df, conditional=conditional)
            df['_numer_'] = np.nan
            for c, prop in zip(conditional, p):
                df['_numer_'] = np.where(eval(c),
                                         np.where(df[self.treatment] == 1, prop, 1 - prop),
                                         df['_numer_'])

        df['_denom_'] = np.where(df[self.treatment] == 1, self._pdenom_, 1 - self._pdenom_)
        df['_ipw_'] = df['_numer_'] / df['_denom_']

        if self.weights is not None:
            df['_ipw_'] = df['_ipw_'] * df[self.weights]

        self.marginal_outcome = np.average(df[self.outcome], weights=df['_ipw_'])

    def summary(self, decimal=3):
        """Prints the summary information for the marginal outcome under the treatment plan of interest.

        Parameters
        ----------
        decimal : int, optional
            Number of decimal places to display. Default is 3
        """
        if np.isnan(self.marginal_outcome):
            raise ValueError('The fit() function must be specified before summary()')

        print('======================================================================')
        print('                       Stochastic IPTW                                ')
        print('======================================================================')
        fmt = 'Treatment:        {:<15} No. Observations:     {:<20}'
        print(fmt.format(self.treatment, self.df.shape[0]))
        fmt = 'Outcome:          {:<15} Treatment Model:      {:<20}'
        print(fmt.format(self.outcome, 'Logistic'))
        print('======================================================================')
        print('Risk:  ', round(self.marginal_outcome, decimal))
        print('======================================================================')
