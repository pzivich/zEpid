import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

from zepid.causal.utils import check_input_data, outcome_accuracy, plot_kde_accuracy


class TimeFixedGFormula:
    r"""G-formula for time-fixed exposure and single endpoint, also referred to as the g-computation algorithm formula.
    Uses the Snowden trick to calculate the marginal treatment under the specified exposure plan

    The g-formula can be expressed as

    .. math::

        E[Y^a] = \sum_l E[Y|A=a,L=l] \times \Pr(L=l)

    When L is continuous, the summation becomes an integral.

    Currently, `TimeFixedGFormula` only supports binary or continuous outcomes. For binary outcomes a logistic
    regression model to predict probabilities of outcomes via statsmodels. For continuous outcomes a linear regression
    or a Poisson regression model can be used to predict outcomes.

    Binary and multivariate exposures are supported. For binary exposures, a string object of the column name for
    the exposure of interest should be provided. For multivariate exposures, a list of string objects corresponding
    to disjoint indicator terms for the exposure should be provided. Multivariate exposures require the user to
    custom specify treatments when fitting the g-formula. A list of the custom treatment must be provided and be
    the same length as the number of disjoint indicator columns. See
    https://github.com/pzivich/Python-for-Epidemiologists/tree/master/3_Epidemiology_Analysis/c_causal_inference/1_time-fixed-treatments
    for examples (highly recommended)

    Key options for treatments:

        * `'all'`     -all individuals are given treatment
        * `'none'`    -no individuals are given treatment
        * custom treatments -create a custom treatment. When specifying this, the dataframe must be referred to as 'g'.
          The following is an example that selects those whose age is 30 or younger and are females:
          ``treatment="((g['age0']<=30) & (g['male']==0))``

    Note
    ----
    Custom treatments use a "magic-g" parameter. Internally, the g-formula implementation names the data set as `g`.
    Therefore, when using custom treatment specifications, the data set must be referred to as `g` when following
    the pandas selection syntax

    Parameters
    ----------
    df : DataFrame
        Pandas dataframe containing the variables of interest
    exposure : str, list
        Column name for exposure variable label or a list of disjoint indicator exposures
    outcome : str
        Column name for outcome variable
    outcome_type : str, optional
        Outcome variable type. Currently only 'binary', 'normal', and 'poisson variable types are supported
    standardize : str, optional
        Who the estimate corresponds to. Options are the entire population, the exposed, or the unexposed. See
        Sato & Matsuyama Epidemiology (2003) for details on weighting to exposed/unexposed. Weighting to the
        exposed or unexposed is also referred to as SMR weighting. Options for standardization are:
        * 'population'    :   weight to entire population
        * 'exposed'       :   weight to exposed individuals
        * 'unexposed'     :   weight to unexposed individuals
    weights : str, optional
        Column name for weights. Default is None, which assumes every observations has the same weight (i.e. 1)

    Examples
    --------
    Setting up the environment

    >>> from zepid import load_sample_data, spline
    >>> from zepid.causal.gformula import TimeFixedGFormula
    >>> df = load_sample_data(timevary=False)
    >>> df[['cd4_rs1', 'cd4_rs2']] = spline(df, 'cd40', n_knots=3, term=2, restricted=True)
    >>> df[['age_rs1', 'age_rs2']] = spline(df, 'age0', n_knots=3, term=2, restricted=True)

    G-formula with a binary treatment and outcome

    >>> g = TimeFixedGFormula(df, exposure='art', outcome='dead')
    >>> g.outcome_model(model='art + male + age0 + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')

    >>> # Return the estimated marginal outcome under treat-all
    >>> g.fit(treatment='all')
    >>> g.marginal_outcome

    >>> # Return the estimated marginal outcome under treat-none
    >>> g.fit(treatment='all')
    >>> g.marginal_outcome

    >>> # Return the estimated marginal outcome under custom treatment (treat all females under 40)
    >>> g.fit(treatment="((g['male']==0) & (g['age0']<=40))")
    >>> g.marginal_outcome

    G-formula with a categorical treatment and binary outcome

    >>> # Creating categorical variable for CD4 count
    >>> df['cd4_1'] = np.where(((df['cd40'] >= 200) & (df['cd40'] < 400)), 1, 0)
    >>> df['cd4_2'] = np.where(df['cd40'] >= 400, 1, 0)

    >>> g = TimeFixedGFormula(df,exposure=['art_male', 'art_female'], outcome='dead', exposure_type='categorical')
    >>> g.outcome_model(model='cd4_1 + cd4_2 + art + male + age0 + age_rs1 + age_rs2 + dvl0')

    >>> # Return marginal outcome under all in reference category (CD4 < 200)
    >>> g.fit(treatment=["False", "False"])

    >>> # Return marginal outcome under all in category 1 (CD4 >= 200 & CD4 < 400)
    >>> g.fit(treatment=["True", "False"])

    >>> # Return marginal outcome under all in category 2 (CD4 > 400)
    >>> g.fit(treatment=["False", "True"])

    G-formula with binary exposure and continuous (normal-distributed) outcome

    >>> g = TimeFixedGFormula(df,exposure='art', outcome='cd4', outcome_type='normal')
    >>> g.outcome_model(model='art + male + age0 + age_rs1 + age_rs2 + dvl0  + cd40 + cd4_rs1 + cd4_rs2')

    G-formula with binary exposure and continuous (Poisson-distributed) outcome

    >>> g = TimeFixedGFormula(df,exposure='art', outcome='cd4', outcome_type='poisson')
    >>> g.outcome_model(model='art + male + age0 + age_rs1 + age_rs2 + dvl0  + cd40 + cd4_rs1 + cd4_rs2')

    G-formula with binary outcome and exposure. With a stochastic treatment/intervention

    >>> g = TimeFixedGFormula(df,exposure='art', outcome='cd4', outcome_type='poisson')
    >>> g.outcome_model(model='art + male + age0 + age_rs1 + age_rs2 + dvl0  + cd40 + cd4_rs1 + cd4_rs2')
    >>> g.fit_stochastic(p=0.75)

    G-formula with binary outcome and exposure. With a conditional stochastic treatment/intervention

    >>> g = TimeFixedGFormula(df,exposure='art', outcome='cd4')
    >>> g.outcome_model(model='art + male + age0 + age_rs1 + age_rs2 + dvl0  + cd40 + cd4_rs1 + cd4_rs2')
    >>> g.fit_stochastic(p=[0.65, 0.85], conditional=["g['male']==1", "g['male']==0"])

    References
    ----------
    JM Snowden, S Rose, and KM Mortimer. "Implementation of G-computation on a simulated
    data set: demonstration of a causal inference technique." American Journal of Epidemiology 173.7 (2011):
    731-738.

    J Ahern, KE Colson, C Margerson-Zilko, A Hubbard, & S Galea. (2016). Predicting the population
    health impacts of community interventions: the case of alcohol outlets and binge drinking. American
    Journal of Public Health, 106(11), 1938-1943.

    J Ahern, A Hubbard, & S Galea. (2009). Estimating the effects of potential public health interventions on
    population disease burden: a step-by-step illustration of causal inference methods. American Journal of
    Epidemiology, 169(9), 1140-1147.
    """
    def __init__(self, df, exposure, outcome, exposure_type='binary', outcome_type='binary', standardize='population',
                 weights=None):
        self.exposure = exposure
        self.outcome = outcome
        self._missing_indicator = '__missing_indicator__'
        self.gf, self._miss_flag, self._continuous_outcome_ = check_input_data(data=df,
                                                                               exposure=exposure,
                                                                               outcome=outcome,
                                                                               estimator="TimeFixedGFormula",
                                                                               drop_censoring=False,
                                                                               drop_missing=True,
                                                                               binary_exposure_only=False)

        if (outcome_type == 'binary') or (outcome_type == 'normal') or (outcome_type == 'poisson'):
            self.outcome_type = outcome_type
        else:
            raise ValueError('Only binary or continuous outcomes are currently supported. Please specify "binary" '
                             '"normal", or "poisson"')

        if (exposure_type == 'binary') or (exposure_type == 'continuous') or (exposure_type == 'categorical'):
            self.exposure_type = exposure_type
        else:
            raise ValueError('Only binary or continuous exposures are currently supported. Please specify "binary", '
                             '"categorical", or "continuous".')

        if standardize in ['population', 'exposed', 'unexposed']:
            self.standardize = standardize
        else:
            raise ValueError('Please specify one of the currently supported standardizations: ' +
                             'population, exposed, unexposed')

        self._weights = weights
        self._outcome_model = None
        self._predicted_y_ = None
        self.marginal_outcome = np.nan
        self.predicted_df = None

    def outcome_model(self, model, print_results=True):
        """Build the outcome regression model. This is also referred to at the Q-model in various parts of the
        literature. This must be specified before the fit function. It is encouraged to make this model as flexible as
        possible

        Parameters
        ----------
        model : str
            Variables to include in the model for predicting the outcome. Must be contained within the input
            pandas dataframe when initialized. Model form should contain the exposure, i.e. 'art + age + male'
        print_results : bool, optional
            Whether to print the logistic regression results to the terminal. Default is True
        """
        if type(self.exposure) is not list:
            if self.exposure not in model:
                warnings.warn("It looks like '" + self.exposure + "' is not included in the outcome model.")

        if self.outcome_type == 'binary':
            linkdist = sm.families.family.Binomial()
        elif self.outcome_type == 'normal':
            linkdist = sm.families.family.Gaussian()
        else:
            linkdist = sm.families.family.Poisson()

        # Modeling the outcome
        if self._weights is None:
            m = smf.glm(self.outcome + ' ~ ' + model, self.gf, family=linkdist)
            self._outcome_model = m.fit()
        else:
            m = smf.glm(self.outcome + ' ~ ' + model, self.gf, family=linkdist,
                        freq_weights=self.gf[self._weights])
            self._outcome_model = m.fit()

        # Creating predicted Y variable
        self._predicted_y_ = self._outcome_model.predict(self.gf)

        # Printing results of the model and if any observations were dropped
        if print_results:
            print('==============================================================================')
            print('Outcome Model')
            print(self._outcome_model.summary())
            print('==============================================================================')

    def fit(self, treatment, predict_missing=True):
        """Fit the parametric g-formula as specified. Binary and multivariate treatments are available. This
        implementation has three options for the binary treatment courses. For multivariate treatments, the user must
        specify custom treatment plans.

        To obtain the confidence intervals, use a bootstrap procedure

        Parameters
        ----------
        treatment : str, list
            There are three options available for treatment plans. All, none, or a custom pattern
              * `'all'`     -all individuals are given treatment
              * `'none'`    -no individuals are given treatment
              * custom  -create a custom treatment. When specifying this, the dataframe must be referred to as 'g' The
                following is an example that selects those whose age is 25 or older and are females;
                ``treatment="((g['age0']>=25) & (g['male']==0))``
        predict_missing : bool, optional
            Whether missing outcome values should be predicted via the model fit to the data with observed outcomes.
            Default is set to True, which assumes that outcome data is missing completely at random conditional on
            the variables included in the outcome model. Note this is the less restrictive assumption regarding missing
            outcome data. If set to False, missing outcome data is assumed to be missing completely at random
            (also referred to as non-informative censoring)

        Returns
        -------
        marginal_outcome
            Parameter of `marginal_outcome` is filled, which is the mean predicted outcome under the treatment strategy
        """
        if self._outcome_model is None:
            raise ValueError('Before the g-formula can be calculated, the outcome model must be specified')
        if (type(treatment) != str) and (type(treatment) != list):
            raise ValueError('Specified treatment must be a string object or a list of string objects')

        # Setting outcome as blank
        g = self.gf.copy()

        # Setting treatment (either multivariate or binary)
        if self.exposure_type == 'binary':
            if type(treatment) == list:
                raise ValueError('A binary exposure is specified. Treatment plan should be a string object')
            if treatment == 'all':
                g[self.exposure] = 1
            elif treatment == 'none':
                g[self.exposure] = 0
            else:  # custom exposure pattern
                g[self.exposure] = np.where(eval(treatment), 1, 0)

        elif self.exposure_type == 'categorical':
            if (treatment == 'all') or (treatment == 'none'):  # Check to make sure custom treatment
                raise ValueError('A multivariate exposure has been specified. A custom treatment must be '
                                 'specified by the user')
            else:
                if len(self.exposure) != len(treatment):  # Check to make sure same about of treatments specified
                    raise ValueError('The list of custom treatment conditions must be the same size as the number of '
                                     'treatments')
                for i in range(len(self.exposure)):
                    g[self.exposure[i]] = np.where(eval(treatment[i]), 1, 0)

                if np.sum(np.where(g[self.exposure].sum(axis=1) > 1, 1, 0)) > 1:
                    warnings.warn('It looks like your specified treatment strategy results in some individuals '
                                  'receiving at least two exposures. Reconsider how the custom treatments are '
                                  'specified', UserWarning)

        else:
            raise ValueError('Still working on allowing for continuous treatments...')
            # TODO fill in this part of continuous exposures

        # Getting predictions
        g[self.outcome] = np.nan
        g[self.outcome] = self._outcome_model.predict(g)
        if not predict_missing:
            g[self.outcome] = np.where(self.gf[self.outcome].isna(), np.nan, g[self.outcome])

        self.predicted_df = g

        if self._weights is None:  # unweighted marginal estimate
            g = g.dropna()
            if self.standardize == 'population':
                self.marginal_outcome = np.mean(g[self.outcome])
            elif self.standardize == 'exposed':
                self.marginal_outcome = np.mean(g.loc[self.gf[self.exposure] == 1, self.outcome])
            else:
                self.marginal_outcome = np.mean(g.loc[self.gf[self.exposure] == 0, self.outcome])
        else:  # weighted marginal estimate
            g = g.dropna()
            if self.standardize == 'population':
                self.marginal_outcome = np.average(g[self.outcome], weights=g[self._weights])
            elif self.standardize == 'exposed':
                self.marginal_outcome = np.average(g.loc[self.gf[self.exposure] == 1, self.outcome],
                                                   weights=g[self._weights])
            else:
                self.marginal_outcome = np.average(g.loc[self.gf[self.exposure] == 0, self.outcome],
                                                   weights=g[self._weights])

    def fit_stochastic(self, p, conditional=None, samples=100, predict_missing=True, seed=None):
        """Fits the g-formula for a stochastic intervention. As currently implemented, `p` percent of the population
        is randomly treated. This process is repeated `n` times and the mean is the marginal stochastic outcome.

        Parameters
        ----------
        p: float, list
            Percent of the population to randomly treat
        conditional: list
            Exclusive conditions to place on the data set for treatment percents. If specified, must match the length
            of the list of probabilities in 'p'
        samples: int, optional
            Number of resamples to calculate the marginal outcome. Default is 100
        predict_missing : bool, optional
            Whether missing outcome values should be predicted via the model fit to the data with observed outcomes.
            Default is set to True, which assumes that outcome data is missing completely at random conditional on
            the variables included in the outcome model. Note this is the less restrictive assumption regarding missing
            outcome data. If set to False, missing outcome data is assumed to be missing completely at random
            (also referred to as non-informative censoring)
        seed: int, optional
            Seed for the random process selection

        References
        ----------
        J Ahern, KE Colson, C Margerson-Zilko, A Hubbard, & S Galea. (2016). Predicting the population
        health impacts of community interventions: the case of alcohol outlets and binge drinking. American
        Journal of Public Health, 106(11), 1938-1943.

        Muñoz, ID, & van der Laan, M (2012). Population intervention causal effects based on stochastic
        interventions. Biometrics, 68(2), 541-549.
        """
        # Checking for common problems before estimation
        if self._outcome_model is None:
            raise ValueError('Before the g-formula can be calculated, the outcome model must be specified')
        if self.exposure_type != 'binary':
            raise ValueError('Only binary exposures are currently available for the stochastic treatment g-formula')
        if conditional is not None:
            if len(p) != len(conditional):
                raise ValueError("'p' and 'conditional' must be the same length")
        else:
            if type(p) != float:
                raise ValueError("Specified percent must be a float when 'conditional' is not specified")

        if seed is None:
            pass
        else:
            np.random.seed(seed)

        if conditional is not None:  # Check exclusive conditional categories
            self._check_conditional(conditional)

        marginals = []
        for s in range(samples):
            g = self.gf.reset_index(drop=True).copy()

            # Selecting out observations
            if conditional is None:  # Unconditional probabilities
                pr = g[self._weights] / np.sum(g[self._weights]) if self._weights is not None else None
                treated = np.random.choice(g.index, size=int(p*g.shape[0]), replace=False, p=pr)

            else:  # Conditional probabilities based on eval()
                treated = []
                for c, prop in zip(conditional, p):
                    gs = g.loc[eval(c)].copy()
                    pr = gs[self._weights] / np.sum(gs[self._weights]) if self._weights is not None else None
                    tr = np.random.choice(gs.index, size=int(prop*gs.shape[0]), replace=False, p=pr)
                    treated.extend(tr)

            # Applying treatment (binary only currently)
            g[self.exposure] = np.where(g.index.isin(treated), 1, 0)

            # Getting predictions
            g[self.outcome] = np.nan
            g[self.outcome] = self._outcome_model.predict(g)
            if not predict_missing:
                g[self.outcome] = np.where(self.gf[self.outcome].isna(), np.nan, g[self.outcome])
                g = g.dropna()

            if self._weights is None:  # unweighted marginal estimate
                if self.standardize == 'population':
                    marginals.append(np.mean(g[self.outcome]))
                elif self.standardize == 'exposed':
                    marginals.append(np.mean(g.loc[self.gf[self.exposure] == 1, self.outcome]))
                else:
                    marginals.append(np.mean(g.loc[self.gf[self.exposure] == 0, self.outcome]))
            else:  # weighted marginal estimate
                if self.standardize == 'population':
                    marginals.append(np.average(g[self.outcome], weights=g[self._weights]))
                elif self.standardize == 'exposed':
                    marginals.append(np.average(g.loc[self.gf[self.exposure] == 1, self.outcome],
                                                weights=g[self._weights]))
                else:
                    marginals.append(np.average(g.loc[self.gf[self.exposure] == 0, self.outcome],
                                                weights=g[self._weights]))

        self.marginal_outcome = np.mean(marginals)

    def run_diagnostics(self, decimal=3):
        """Runs diagnostics for the g-formula regression model used. Diagnostics include summary statistics and a
        Kernel Density plot for the predictive accuracy of the model. The model compares the model predicted value to
        the observed outcome value.
        """
        # Summary statistics of prediction accuracy
        outcome_accuracy(true=self.gf[self.outcome], predicted=self._predicted_y_, decimal=decimal)

        # Distribution plot of accuracy
        self.plot_kde()
        plt.title("Kernel Density of Accuracy")
        plt.tight_layout()
        plt.show()

    def plot_kde(self, bw_method='scott', fill=True, color='b'):
        """Generates a Kernel Density plot of the accuracy of the model predicted outcomes. The plot compares the
        model predicted outcome to the observed outcome. This can be used as a diagnostic for the g-formula.

        Parameters
        ----------
        bw_method : str, optional
            Method used to estimate the bandwidth. Following SciPy, either 'scott' or 'silverman' are valid options
        fill : bool, optional
            Whether to color the area under the density curves. Default is true
        color : str, optional
            Color of the line/area. Default is blue

        Returns
        -------
        matplotlib axes
        """
        if self._predicted_y_ is None:
            raise ValueError("The outcome_model function must be ran before any diagnostics")

        v = self._predicted_y_ - self.gf[self.outcome]
        return plot_kde_accuracy(values=v.dropna(),
                                 bw_method=bw_method, fill=fill, color=color)

    def _check_conditional(self, conditional):
        """Check that conditionals are exclusive for the stochastic fit process
        """
        g = self.gf.copy()
        a = np.array([0]*g.shape[0])
        for c in conditional:
            a = np.add(a, np.where(eval(c), 1, 0))

        if np.sum(np.where(a > 1, 1, 0)):
            warnings.warn("It looks like your conditional categories are NOT exclusive. For appropriate estimation, "
                          "the conditions that designate each category should be exclusive", UserWarning)


class SurvivalGFormula:
    r"""G-formula for time-to-event data where the exposure is fixed at baseline. Only supports binary exposures and
    outcomes. Outcomes are predicted using a logistic regression model. Input data set should be in a long format,
    where each row corresponds to an individual observed for one unit of time

    Key options for treatments:

        * `'all'`     -all individuals are given treatment
        * `'none'`    -no individuals are given treatment
        * custom treatments -create a custom treatment. When specifying this, the dataframe must be referred to as 'g'.
          The following is an example that selects those whose age is 30 or younger and are females:
          ``treatment="((g['age0']<=30) & (g['male']==0))``

    Parameters
    ----------
    df : DataFrame
        Pandas dataframe containing the variables of interest
    idvar : str
        Column name for the ID label
    exposure : str, list
        Column name for exposure variable label or a list of disjoint indicator exposures
    outcome : str
        Column name for outcome variable
    time : str
        Column name for time variable
    weights : str, optional
        Column name for weights. Default is None, which assumes every observations has the same weight (i.e. 1)

    Note
    ----
    Custom treatments use a "magic-g" parameter. Internally, the g-formula implementation names the data set as `g`.
    Therefore, when using custom treatment specifications, the data set must be referred to as `g` when following
    the pandas selection syntax

    Examples
    --------
    Setting up data in long format

    >>> from zepid import load_sample_data
    >>> from zepid.causal.gformula import SurvivalGFormula
    >>> import matplotlib.pyplot as plt
    >>> df = load_sample_data(False).drop(columns=['cd4_wk45'])

    >>> df['t'] = np.round(df['t']).astype(int)
    >>> df = pd.DataFrame(np.repeat(df.values, df['t'], axis=0), columns=df.columns)
    >>> df['t'] = df.groupby('id')['t'].cumcount() + 1
    >>> df.loc[((df['dead'] == 1) & (df['id'] != df['id'].shift(-1))), 'd'] = 1
    >>> df['d'] = df['d'].fillna(0)
    >>> df['t_sq'] = df['t']**2
    >>> df['t_cu'] = df['t']**3

    Estimating the time-to-event mean effect under treat-all plan

    >>> sgf = SurvivalGFormula(df.drop(columns=['dead']), idvar='id', exposure='art', outcome='d', time='t')
    >>> sgf.outcome_model(model='art + male + age0 + cd40 + dvl0 + t + t_sq + t_cu')
    >>> sgf.fit(treatment='all')
    >>> print(sgf.marginal_outcome)

    Plotting cumulative incidence function

    >>> sgf.plot(color='r')
    >>> plt.show()

    Estimating the time-to-event mean effect under treat-none plan

    >>> sgf = SurvivalGFormula(df.drop(columns=['dead']), idvar='id', exposure='art', outcome='d', time='t')
    >>> sgf.outcome_model(model='art + male + age0 + cd40 + dvl0 + t + t_sq + t_cu')
    >>> sgf.fit(treatment='none')

    Estimating the time-to-event mean effect under custom treatment plan

    >>> sgf = SurvivalGFormula(df.drop(columns=['dead']), idvar='id', exposure='art', outcome='d', time='t')
    >>> sgf.outcome_model(model='art + male + age0 + cd40 + dvl0 + t + t_sq + t_cu')
    >>> sgf.fit(treatment="((g['age0']>=25) & (g['male']==0))")

    Notes
    -----
    The following process is used to estimate the cumulative incidence function.
    (1) A pooled logistic regression model is fit to the data. The model should predict the outcome conditional on
    treatment, baseline confounders, and time. Time should be modeled using flexible functional forms (e.g. splines)
    (2) Survival probabilities are estimated by predicting values at each time from the pooled logistic model and taking
    the cumulative product. The survival probabilities are predicted under the treatment plan of interest
    (3) Average the cumulative incidence function for each time period from all the subjects.

    References
    ----------
    Hernán MA. (2010). The hazards of hazard ratios. Epidemiology, 21(1), 13–15.
    doi:10.1097/EDE.0b013e3181c1ea43
    """
    def __init__(self, df, idvar, exposure, outcome, time, weights=None):
        self.exposure = exposure
        self.outcome = outcome
        self.t = time
        self.id = idvar
        self._missing_indicator = '__missing_indicator__'
        self.gf, self._miss_flag, self._continuous_outcome_ = check_input_data(data=df,
                                                                               exposure=exposure,
                                                                               outcome=outcome,
                                                                               estimator="SurvivalGFormula",
                                                                               drop_censoring=True,
                                                                               drop_missing=True,
                                                                               binary_exposure_only=True)
        self.gf = self.gf.copy().sort_values(by=[idvar, time]).reset_index(drop=True)
        if self._continuous_outcome_:
            raise ValueError("SurvivalGFormula does not support continuous outcomes")

        self._weights = weights
        self._outcome_model = None
        self.marginal_outcome = None
        self.predicted_df = None

    def outcome_model(self, model, print_results=True):
        """Build the pooled logistic model. This must be specified before the fit function. It is encouraged to model
        time as flexible as possible. For example, use spline terms

        Parameters
        ----------
        model : str
            Variables to include in the model for predicting the outcome. Must be contained within the input
            pandas dataframe when initialized. Model form should contain the exposure, i.e. 'art + age + male'
        print_results : bool, optional
            Whether to print the logistic regression results to the terminal. Default is True
        """
        if self.exposure not in model:
            warnings.warn("It looks like '" + self.exposure + "' is not included in the outcome model.")

        # Modeling the outcome
        linkdist = sm.families.family.Binomial()
        if self._weights is None:
            m = smf.glm(self.outcome + ' ~ ' + model, self.gf, family=linkdist)
        else:
            m = smf.glm(self.outcome + ' ~ ' + model, self.gf, family=linkdist,
                        freq_weights=self.gf[self._weights])

        self._outcome_model = m.fit()

        # Printing results of the model
        if print_results:
            print('==============================================================================')
            print('Outcome Model')
            print(self._outcome_model.summary())
            print('==============================================================================')

    def fit(self, treatment):
        """Fit the parametric g-formula for time-to-event data. To obtain the confidence intervals, use a bootstrap
        procedure

        Parameters
        ----------
        treatment : str, list
            There are three options available for treatment plans. All, none, or a custom pattern
              * `'all'`     -all individuals are given treatment
              * `'none'`    -no individuals are given treatment
              * `'natural'` -all individuals retain their observed exposure
              * custom  -create a custom treatment. When specifying this, the dataframe must be referred to as 'g' The
                following is an example that selects those whose age is 25 or older and are females;
                ``treatment="((g['age0']>=25) & (g['male']==0))``

        Returns
        -------
        marginal_outcome
            Parameter of `marginal_outcome` is filled, which is the mean predicted outcome under the treatment strategy
        """
        if self._outcome_model is None:
            raise ValueError('Before the g-formula can be calculated, the outcome model must be specified')

        # Setting outcome as blank
        g = self.gf.copy()

        # Setting treatment (either multivariate or binary)
        if treatment == 'all':
            g[self.exposure] = 1
        elif treatment == 'none':
            g[self.exposure] = 0
        elif treatment == 'natural':
            pass
        else:  # custom exposure pattern
            g[self.exposure] = np.where(eval(treatment), 1, 0)

        g[self.outcome] = np.nan
        g[self.outcome] = 1 - self._outcome_model.predict(g)

        # Cumulative product
        g[self.outcome] = 1 - g.groupby(self.id)[self.outcome].cumprod()

        if self._weights is None:  # unweighted marginal estimate
            marginal = g.groupby(self.t)[self.outcome].mean()

        else:  # weighted marginal estimate
            marginal = self._weighted_average(data=g, y_col=self.outcome, weight_col=self._weights, by_col=self.t)

        self.marginal_outcome = marginal.rename(index='timeline')
        self.predicted_df = g

    def plot(self, **plot_kwargs):
        """Plots the estimated cumulative incidence function

        Parameters
        ----------
        **plot_kwargs : optional
            Optional keyword arguments for matplotlib. kwargs will pass matplotlib.pyploy.step kwargs are accepted. See
            matplotlib 'step()' function documentation for further details

        Returns
        -------
        matplotlib axes
        """
        if self.marginal_outcome is None:
            raise ValueError('Before plotting, the marginal outcomes must be estimated with the fit() function')

        ax = plt.gca()
        ax.step(self.marginal_outcome.index, self.marginal_outcome, where='post', **plot_kwargs)
        ax.set_xlabel(self.t)
        ax.set_ylabel('Risk of ' + self.outcome)
        return ax

    @staticmethod
    def _weighted_average(data, y_col, weight_col, by_col):
        """Background function to calculate the weighted mean by time
        """
        data['_w_y_'] = data[y_col] * data[weight_col]
        data['_w_t_'] = data[weight_col] * pd.notnull(data[y_col])
        g = data.groupby(by_col)
        result = g['_w_y_'].sum() / g['_w_t_'].sum()
        return result
