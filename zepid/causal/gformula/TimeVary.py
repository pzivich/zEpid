import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import links


# TODO add_competing_risk function
# TODO censoring_model function
class MonteCarloGFormula:
    def __init__(self, df, idvar, exposure, outcome, time_in, time_out, weights=None):
        """Time-varying implementation of the Monte Carlo g-formula, also referred to as the g-computation algorithm
        formula. The Monte Carlo estimator is useful for survival data. For an extensive walkthrough of the Monte Carlo
        g-formula, see Keil et al. 2014 and other listed references. This implementation has four options for the
        treatment courses:

        Options for treatments
        * all : all individuals are given treatment
        * none : no individuals are given treatment
        * natural : individuals retain their observed treatment
        * custom : create a custom treatment. When specifying this, the dataframe must be  referred to as 'g' The
            following is an example that selects those whose age is 25 or older and are females
            Ex) treatment="((g['age0']>=25) & (g['male']==0))

        Currently, only binary exposures and a binary outcomes are supported. Logistic regression models are used to
        predict exposures and outcomes via statsmodels.
        See http://zepid.readthedocs.io/en/latest/ for an example (highly recommended)

        Parameters
        ----------
        df : DataFrame
            Pandas dataframe containing the variables of interest
        exposure : str
            Treatment column label
        outcome : str
            Outcome column label
        time_out : str
            End of follow-up period time column label
        time_in : str
            Start of follow-up period time label
        weights : str, optional
            Column label for weights. Default is None, which assumes every observations has the same weight (i.e. 1)

        Notes
        -----
        1) Monte Carlo increases by time units of one. Input dataset should reflect this
        2) Only binary exposures and binary outcomes are supported
        3) Binary and continuous covariates are supported
        4) The labeling of the covariate models is important. They are fit in the order that they are labeled!
        5) Fit the natural course model first and compare to the observed data. They should be similar

        Process for the Monte Carlo g-formula
            1) run lines in "in_recode"
            2) time-varying covariates, order ascending in from "labels"
                a) predict time-varying covariate
                b) run lines in "recode" from "add_covariate_model()"
            3) predict exposure / apply exposure pattern
            4) predict outcome
            5) run lines in "out_recode"
            6) lag variables in "lags"
            7) append current-time rows to full dataframe
            8) Repeat till t_max is met

        Examples
        --------
        Setting up the environment
        >>>import numpy as np
        >>>from zepid import load_sample_data, spline
        >>>from zepid.causal.gformula import TimeVaryGFormula
        >>>df = load_sample_data(timevary=True)
        >>>df['lag_art'] = df['art'].shift(1)
        >>>df['lag_art'] = np.where(df.groupby('id').cumcount() == 0, 0, df['lag_art'])
        >>>df['lag_cd4'] = df['cd4'].shift(1)
        >>>df['lag_cd4'] = np.where(df.groupby('id').cumcount() == 0, df['cd40'], df['lag_cd4'])
        >>>df['lag_dvl'] = df['dvl'].shift(1)
        >>>df['lag_dvl'] = np.where(df.groupby('id').cumcount() == 0, df['dvl0'], df['lag_dvl'])
        >>>df[['age_rs0', 'age_rs1', 'age_rs2']] = spline(df, 'age0', n_knots=4, term=2, restricted=True)
        >>>df['cd40_sq'] = df['cd40'] ** 2
        >>>df['cd40_cu'] = df['cd40'] ** 3
        >>>df['cd4_sq'] = df['cd4'] ** 2
        >>>df['cd4_cu'] = df['cd4'] ** 3
        >>>df['enter_sq'] = df['enter'] ** 2
        >>>df['enter_cu'] = df['enter'] ** 3

        Estimating the g-formula with the Monte Carlo estimator
        >>>g = TimeVaryGFormula(df, idvar='id', exposure='art', outcome='dead', time_in='enter', time_out='out')

        >>># Specifying the exposure/treatment model
        >>>exp_m = 'male + age0 + age_rs0 + age_rs1 + age_rs2 + cd40 + cd40_sq + cd40_cu + dvl0 + cd4 + cd4_sq +' +
        >>>        'cd4_cu + dvl + enter + enter_sq + enter_cu'
        >>>g.exposure_model(exp_m, restriction="g['lag_art']==0")  # restriction enforces intent-to-treat

        >>># Specifying the outcome model
        >>>out_m = 'art + male + age0 + age_rs0 + age_rs1 + age_rs2 + cd40 + cd40_sq + cd40_cu + dvl0 + cd4 +' +
        >>>        'cd4_sq + cd4_cu + dvl + enter + enter_sq + enter_cu'
        >>>g.outcome_model(out_m, restriction="g['drop']==0")  # restriction enforces loss-to-follow-up

        >>># Specifying the time-varying confounder models
        >>>dvl_m = 'male + age0 + age_rs0 + age_rs1 + age_rs2 + cd40 + cd40_sq + cd40_cu + dvl0 + lag_cd4 + ' +
        >>>        'lag_dvl + lag_art + enter + enter_sq + enter_cu'
        >>>g.add_covariate_model(label=1, covariate='dvl', model=dvl_m, var_type='binary')
        >>>cd4_m = 'male + age0 + age_rs0 + age_rs1 + age_rs2 +  cd40 + cd40_sq + cd40_cu + dvl0 + lag_cd4 + ' +
        >>>        'lag_dvl + lag_art + enter + enter_sq + enter_cu'
        >>>cd4_recode_scheme = ("g['cd4'] = np.maximum(g['cd4'],1);"  # Recode scheme makes sure variables are recoded
        >>>                     "g['cd4_sq'] = g['cd4']**2;"
        >>>                     "g['cd4_cu'] = g['cd4']**3")
        >>>g.add_covariate_model(label=2, covariate='cd4', model=cd4_m, recode=cd4_recode_scheme, var_type='continuous')

        >>># Estimating outcomes under a simulated Markov Chain Monte Carlo for natural course
        >>>g.fit(treatment="((g['art']==1) | (g['lag_art']==1))",  # Treatment plan (natural course in this case)
        >>>      lags={'art': 'lag_art',  # Creating variables to lag in the process
        >>>            'cd4': 'lag_cd4',
        >>>            'dvl': 'lag_dvl'},
        >>>      sample=50000,  # Number of resamples to use (should be large number to reduce simulation error)
        >>>      t_max=None,  # Maximum time to simulate to (None uses data set maximum time)
        >>>      in_recode=("g['enter_sq'] = g['enter']**2;"
        >>>                 "g['enter_cu'] = g['enter']**3"))  # How to recode time in each time-step
        >>># See website documentation for further instructions
        >>># (https://zepid.readthedocs.io/en/latest/Causal.html#g-computation-algorithm-monte-carlo)

        References
        ----------
        Keil, AP, Edwards, JK, Richardson, DB, Naimi, AI, Cole, SR (2014). The Parametric g-Formula for Time-
        to-Event Data: Intuition and a Worked Example. Epidemiology 25(6), 889-897
        """
        self.gf = df.sort_values(by=[idvar, time_out]).copy()
        self.idvar = idvar

        # Checking that treatment is binary
        if df[exposure].dropna().value_counts().index.isin([0, 1]).all():
            self.exposure = exposure
        else:
            raise ValueError('Only binary treatments/exposures are currently implemented')

        # Checking that outcome is binary
        if df[outcome].dropna().value_counts().index.isin([0, 1]).all():
            self.outcome = outcome
        else:
            raise ValueError('Only binary outcomes are currently implemented')

        # Generating an indicator for censoring
        self.gf['__uncensored__'] = np.where((self.gf[idvar] != self.gf[idvar].shift(-1)) & (self.gf[outcome] == 0),
                                             0, 1)
        self.gf['__uncensored__'] = np.where(self.gf[time_out] == np.max(df[time_out]),
                                             1, self.gf['__uncensored__'])

        self.exp_model = None
        self.out_model = None
        self.cens_model = None
        self.time_in = time_in
        self.time_out = time_out
        self._exposure_model_fit = False
        self._outcome_model_fit = False
        self._censor_model_fit = False
        self._covariate_models = []
        self._covariate_model_index = []
        self._covariate = []
        self._covariate_type = []
        self._covariate_recode = []
        self._weights = weights
        self.predicted_outcomes = None

    def exposure_model(self, model, restriction=None, print_results=True):
        """Add a specified regression model for the exposure. This is used for natural course estimation of the Monte
        Carlo g-formula. This must be specified before calling the fit function.

        Parameters
        ----------
        model : str
            Variables to include in the model for predicting the exposure. Must be contained within the input
            pandas dataframe when initialized. Format follows patsy standards
            For example) 'var1 + var2 + var3 + var4'
        restriction : str, optional
            Used to restrict the population that the regression model is fit to. Useful for Intent-to-Treat model
            fitting. The pandas dataframe must be referred to as 'g'. For example) "g['art']==1"
        print_results : bool, optional
            Whether to print the logistic regression model results to the terminal. Default is True
        """
        g = self.gf.copy()
        if restriction is not None:
            g = g.loc[eval(restriction)].copy()
        linkdist = sm.families.family.Binomial()

        if self._weights is None:  # Unweighted g-formula
            self.exp_model = smf.glm(self.exposure + ' ~ ' + model, g, family=linkdist).fit()
        else:  # Weighted g-formula
            self.exp_model = smf.gee(self.exposure + ' ~ ' + model, self.idvar, g, weights=g[self._weights],
                                     family=linkdist).fit()

        if print_results:
            print(self.exp_model.summary())
        self._exposure_model_fit = True

    def outcome_model(self, model, restriction=None, print_results=True):
        """Add a specified regression model for the outcome. Must be specified before the fit function.

        Parameters
        ----------
        model:
            Variables to include in the model for predicting the outcome. Must be contained within the input
            pandas dataframe when initialized. Format follows patsy standards
            For example) 'var1 + var2 + var3 + var4'
        restriction : str, optional
            Used to restrict the population that the regression model is fit to. Useful for Intent-to-Treat model
            fitting. The pandas dataframe must be referred to as 'g'. For example) "g['art']==1"
        print_results : bool, optional
            Whether to print the logistic regression model results to the terminal. Default is True
        """
        g = self.gf.copy()
        if restriction is not None:
            g = g.loc[eval(restriction)].copy()
        linkdist = sm.families.family.Binomial()

        if self._weights is None:  # Unweighted g-formula
            self.out_model = smf.glm(self.outcome + ' ~ ' + model, g, family=linkdist).fit()
        else:  # Weighted g-formula
            self.out_model = smf.gee(self.outcome + ' ~ ' + model, self.idvar, g, weights=g[self._weights],
                                     family=linkdist).fit()
        if print_results:
            print(self.out_model.summary())

        self._outcome_model_fit = True

    def censoring_model(self, model, restriction=None, print_results=True):
        """Add a specified regression model for censoring. Specifying this model is optional, but is recommended when
        censoring occurs in your data set. Otherwise, you will be assuming non-informative censoring

        Parameters
        ----------
        model:
            Variables to include in the model for predicting the outcome. Must be contained within the input
            pandas dataframe when initialized. Format follows patsy standards
            For example) 'var1 + var2 + var3 + var4'
        restriction : str, optional
            Used to restrict the population that the regression model is fit to. Useful for Intent-to-Treat model
            fitting. The pandas dataframe must be referred to as 'g'. For example) "g['art']==1"
        print_results : bool, optional
            Whether to print the logistic regression model results to the terminal. Default is True
        """
        g = self.gf.copy()
        if restriction is not None:
            g = g.loc[eval(restriction)].copy()
        linkdist = sm.families.family.Binomial()

        if self._weights is None:  # Unweighted g-formula
            self.cens_model = smf.glm('__uncensored__ ~ ' + model, g, family=linkdist).fit()
        else:  # Weighted g-formula
            self.cens_model = smf.gee('__uncensored__ ~ ' + model, self.idvar, g, weights=g[self._weights],
                                      family=linkdist).fit()
        if print_results:
            print(self.cens_model.summary())

        self._censor_model_fit = True

    def add_covariate_model(self, label, covariate, model, restriction=None, recode=None, var_type='binary',
                            print_results=True):
        """Add a specified regression model for time-varying confounders. Unlike the exposure and outcome models, a
        covariate model does NOT have to be specified. Additionally, *n* covariate models can be specified for *n*
        time-varying covariates. Additional models are added by repeated calls for this function with the corresponding
        covariates and predictive regression equations

        Parameters
        ----------
        label : int
            Integer label for the covariate model. Covariate models are fit in ascending order within
             TimeVaryGFormula
        covariate : str
            Column label for time-varying confounder to be predicted
        model : str
            Variables to include in the model for predicting the outcome. Must be contained within the input
            pandas dataframe when initialized. Format follows patsy
            For example) 'var1 + var2 + var3 + var4'
        restriction : str, optional
            Used to restrict the population to fit the logistic regression model to. Useful for Intent-to-Treat
            model fitting. The pandas dataframe must be referred to as 'g'. For example) "g['art']==1"
        recode : str, optional
            This variable is vitally important for various functional forms implemented later in models. This
            is used to run some background code to recreate functional forms as the g-formula is estimated via fit()
            For an example, let's say we have age but we want the functional form to be quadratic. For this, we
            would set the recode="g['age_sq'] = g['age']**2;" Similar to TimeFixedGFormula, 'g' must be specified as the
            DataFrame object with the corresponding indexes. Also lines of executable code should end with ';', so
            Python knows that the line ends there. My apologies for this poor solution... I am working on a better way.
            In the background, Python executes the code input into recode
        var_type : str, optional
            Type of variable that the covariate is. Current options include 'binary' or 'continuous'
        print_results : bool, optional
            Whether to print the logistic regression model results to the terminal. Default is True
        """
        if type(label) is not int:
            raise ValueError('Label must be an integer')

        # Building predictive model
        g = self.gf.copy()
        if restriction is not None:
            g = g.loc[eval(restriction)].copy()

        if self._weights is None:  # Unweighted g-formula
            if var_type == 'binary':
                linkdist = sm.families.family.Binomial()
                m = smf.glm(covariate + ' ~ ' + model, g, family=linkdist)
            elif var_type == 'continuous':
                m = smf.gls(covariate + ' ~ ' + model, g)
            else:
                raise ValueError('Only binary or continuous covariates are currently supported')
        else:  # Weighted g-formula
            if var_type == 'binary':
                linkdist = sm.families.family.Binomial()
                m = smf.gee(covariate + ' ~ ' + model, self.idvar, g, weights=g[self._weights], family=linkdist)
            elif var_type == 'continuous':
                linkdist = sm.families.family.Gaussian(sm.families.links.identity)
                m = smf.gee(covariate + ' ~ ' + model, self.idvar, g, weights=g[self._weights], family=linkdist)
            else:
                raise ValueError('Only binary or continuous covariates are currently supported')

        f = m.fit()
        if print_results:
            print(f.summary())

        # Adding to lists, it is used to predict variables later on for the time-varying...
        self._covariate_models.append(f)
        self._covariate_model_index.append(label)
        self._covariate.append(covariate)
        self._covariate_type.append(var_type)
        if recode is None:
            self._covariate_recode.append('None')  # Must be string for exec() to use later
        else:
            self._covariate_recode.append(recode)

    def fit(self, treatment, lags=None, sample=10000, t_max=None, in_recode=None, out_recode=None, low_memory=True):
        """Estimate the counterfactual outcomes under the specified treatment plan using the previously specified
        regression models. Both the exposure and outcome models need to be specified before fit can be called

        Parameters
        ----------
        treatment : str
            Treatment strategy. Options include
            * all : all individuals are given treatment
            * none : no individuals are given treatment
            * natural : individuals retain their observed treatment. Only available for MonteCarlo
            * custom : create a custom treatment. When specifying this, the dataframe must be  referred to as 'g'
                The following is an example that selects those whose age is 25 or older and are females
                Ex) treatment="((g['age0']>=25) & (g['male']==0))"
        lags : dict, optional
            Dictionary of variable names and the corresponding lagged variable name. This should be specified for all
            variables that are lagged. This parameter is only used for the Monte Carlo g-formula.
             As an example, {'art':'lagged_art'} would correctly lag the variable 'art' to be 'lagged_art' for each time
             step of the Monte Carlo procedure
        sample : int, optional
            Number of individuals to sample from the original data with replacement. This argument is only used by the
            Monte Carlo g-formula. The number of samples to use should be a large number to reduce simulation error.
            The default is 10000
        t_max : int, optional
            Maximum time to run Monte Carlo g-formula until. Default is None, which uses the maximum time of the input
            dataframe.
        in_recode : str, optional
            On the fly recoding of variables done before the Monte Carlo loop starts. Needed to do any kind of
            functional forms for entry times. This is executed at each start of the Monte Carlo g-formula time steps
        out_recode : str, optional
            On the fly recoding of variables done at the end of the Monte Carlo loop. Needed for operations like
            counting the number of days with a treatment. This is executed at each end of the Monte Carlo g-formula
            time steps
        low_memory : bool, optional
            This optional parameter controls whether the g-formula outputs a condensed data set or a data set containing
            each step of the Monte Carlo procedure. When outputting each step of the Monte Carlo procedure, this can
            be intensive on memory requirements. `low_memory` is set to `True` (the default) only the last observation
            per ID is added to the full data set.
            From the condensed data (`low_memory`), you are able to directly fit a Kaplan-Meier and estimate the effect
            of treatment.
            The full Monte Carlo data set is useful during the model building phase and checking whether models are
            correctly specified. It is less useful during estimation and bootstrapping confidence intervals
        """
        if self._outcome_model_fit is False:
            raise ValueError('Before the g-formula can be calculated, the outcome model must be specified')
        if not isinstance(treatment, str) and not isinstance(treatment, list):
            raise ValueError('Specified treatment must be a string object')
        if self._exposure_model_fit is False:
            raise ValueError('Before the g-formula can be calculated, the exposure model must be specified for '
                             'Monte Carlo estimation')

        # Monte Carlo Estimation
        # Re-sampling from data set for estimation procedure
        if self._weights is None:
            gs = self.gf.loc[(self.gf.groupby(self.idvar).cumcount() == 0) == True].sample(n=sample, replace=True)
        else:
            gs = self.gf.loc[(self.gf.groupby(self.idvar).cumcount() == 0) == True].sample(n=sample,
                                                                                           weights=self._weights,
                                                                                           replace=True)
        gs['uid_g_zepid'] = [v for v in range(sample)]

        # Background preparations
        gs[self.outcome] = 0
        mc_simulated_data = []
        g = gs.copy()
        g['uncensored'] = 1  # this tricks the Monte-Carlo procedure into not censoring anyone if cens_model None
        if t_max is None:  # getting maximum time steps to run g-formula
            t_max = np.max(self.gf[self.time_out])

        if len(self._covariate_models) != 0:
            cov_model_order = (sorted(range(len(self._covariate_model_index)),
                                      key=self._covariate_model_index.__getitem__))
            run_cov = True
        else:
            run_cov = False

        # Monte Carlo for loop
        for i in range(int(t_max)):
            g = g.loc[(g[self.outcome] == 0) & (g['uncensored'] == 1)].reset_index(drop=True).copy()
            g[self.time_in] = i
            if in_recode is not None:
                exec(in_recode)

            # predict time-varying covariates
            if run_cov:
                for j in cov_model_order:
                    g[self._covariate[j]] = self._predict(df=g, model=self._covariate_models[j],
                                                          variable=self._covariate_type[j])
                    exec(self._covariate_recode[j])

            # predict exposure when customized treatments
            if treatment == 'all':
                g[self.exposure] = 1
            elif treatment == 'none':
                g[self.exposure] = 0
            elif treatment == 'natural':
                g[self.exposure] = self._predict(df=g, model=self.exp_model, variable='binary')
            else:  # custom exposure pattern
                g[self.exposure] = self._predict(df=g, model=self.exp_model, variable='binary')
                g[self.exposure] = np.where(eval(treatment), 1, 0)

            # predict outcome
            g[self.outcome] = self._predict(df=g, model=self.out_model, variable='binary')
            g[self.time_out] = i + 1
            # TODO evaluate competing risks

            # predict censoring
            if self._censor_model_fit:
                g['uncensored'] = self._predict(df=g, model=self.cens_model, variable='binary')
                g[self.outcome] = np.where(g['uncensored'] == 1, g[self.outcome], 0)

            # last iteration, marking everyone as censored
            if i == t_max-1:
                g['uncensored'] = 0

            # executing any code before appending
            if out_recode is not None:
                exec(out_recode)

            # updating lagged variables
            if lags is not None:
                for k, v in lags.items():
                    g[v] = g[k]

            # stacking simulated data in a list
            if low_memory:  # Only stacking when censored or failed
                mc_simulated_data.append(g.loc[(g[self.outcome] > 0) | (g['uncensored'] == 0)])
            else:  # Stacks all simulated observations
                mc_simulated_data.append(g)

        try:
            gs = pd.concat(mc_simulated_data, ignore_index=True, sort=False)
        except TypeError:  # gets around pandas <0.22 error
            gs = pd.concat(mc_simulated_data, ignore_index=True)
        if self._weights is None:
            self.predicted_outcomes = gs[['uid_g_zepid', self.exposure, self.outcome, self.time_in,
                                          self.time_out] + self._covariate].sort_values(by=['uid_g_zepid',
                                                                        self.time_in]).reset_index(drop=True)
        else:
            self.predicted_outcomes = gs[['uid_g_zepid', self.exposure, self.outcome, self._weights, self.time_in,
                                          self.time_out] + self._covariate].sort_values(by=['uid_g_zepid',
                                                                         self.time_in]).reset_index(drop=True)

    @staticmethod
    def _predict(df, model, variable):
        """Hidden predict method to shorten the Monte Carlo estimation code
        """
        pp = model.predict(df)
        if variable == 'binary':
            pred = np.random.binomial(1, pp, size=len(pp))
        elif variable == 'continuous':
            pred = np.random.normal(loc=pp, scale=np.std(model.resid), size=len(pp))
        else:
            raise ValueError('That option is not supported')
        return pred

    @staticmethod
    def __predict__(df, model, variable, formula):
        """UNUSED FUNCTION

        New hidden predict method to shorten Monte Carlo estimation code. This is slower than statsmodels predict,
        but likely can be optimized. I would need to find a faster alternative to np.dot()...

        For this to work, I need to do the following:
            -need to have each ..._model() store a self.formula
            -switch all functions to sm.GLM or sm.GEE
        """
        import patsy
        from zepid.calc import odds_to_probability

        xdata = patsy.dmatrix(formula, df) #, return_type='dataframe')
        # pred = xdata.mul(np.array(model.params), axis='columns').sum(axis=1)
        pred = xdata.dot(model.params) # TODO optimize this...

        if variable == 'binary':
            pred = np.random.binomial(1, odds_to_probability(np.exp(pred)), size=xdata.shape[0])
        elif variable == 'continuous':
            pred = np.random.normal(loc=pred, scale=np.std(model.resid), size=len(pp))
        else:
            raise ValueError('That option is not supported')


class IterativeCondGFormula:
    def __init__(self, df, idvar, exposure, outcome, time_out, weights=None):
        """Time-varying implementation of the g-formula, also referred to as the g-computation algorithm formula. The
        time-varying parametric g-formula uses the iterative conditional estimator (also referred to as the sequential
        regression estimator). The iterative conditional estimator is useful for longitudinal data and requires less
        model specification than the Monte Carlo g-formula. This implementation has three options for treatment courses:

        Options for treatments
        * all : all individuals are given treatment
        * none : no individuals are given treatment
        * custom : create a custom treatment. When specifying this, the dataframe must be  referred to as 'g' The following
            is an example that selects those whose age is 25 or older and are females
            Ex) treatment="((g['age0']>=25) & (g['male']==0))

        Currently, only binary exposures and a binary outcomes are supported. Logistic regression models are used to
        predict exposures and outcomes via statsmodels. See Kreif et al. 2017 for a good description of the
        iterative conditional g-formula. See http://zepid.readthedocs.io/en/latest/ for an example

        Parameters
        ----------
        df : DataFrame
            Pandas dataframe containing the variables of interest
        exposure : str
            Treatment column label
        outcome : str
            Outcome column label
        time_out : str
            End of follow-up period time column label
        weights : str, optional
            Column label for weights. Default is None, which assumes every observations has the same weight (i.e. 1)

        Notes
        -----
        Process for the sequential regression g-formula
            1) Convert long data set to wide data set
            2) Identify individuals who followed the counterfactual treatment plan and had the outcome
            3) Fit a regression model for the outcome at time t for Y
            4) Predict outcomes under the observed treatment and the counterfactual treatment
            5) Repeat regression model fitting for t-1 to min(t)
            6) Take the mean predicted Y at the end to obtain the cumulative probability

        Examples
        --------
        Setting up the environment
        >>>from zepid.causal.gformula import IterativeCondGFormula
        >>>from zepid import load_longitudinal_data
        >>>df = load_longitudinal_data()
        >>>g = IterativeCondGFormula(df, idvar='id', exposure='A', outcome='Y', time_out='t',
        >>>                     method='SequentialRegression')  # Specify for sequential regression estimator

        >>># Specify the outcome model
        >>>g.outcome_model('A + L', print_results=False)

        >>># Estimate the marginal risk under treat-all for max(t)
        >>>g.fit(treatment="all")
        >>>print(g.predicted_outcomes)

        >>># Estimate the marginal risk under treat-all for t=2
        >>>g.fit(treatment="all", t_max=2)
        >>>print(g.predicted_outcomes)

        >>># Estimate the marginal risk under a custom treatment (treat all except at time 2)
        >>>g.fit(treatment="g['t'] != 2")
        >>>print(g.predicted_outcomes)

        References
        ----------
        Kreif, N., Tran, L., Grieve, R., De Stavola, B., Tasker, R. C., & Petersen, M. (2017). Estimating the
        comparative effectiveness of feeding interventions in the pediatric intensive care unit: a demonstration of
        longitudinal targeted maximum likelihood estimation. American Journal of Epidemiology, 186(12), 1370-1379.
        """
        self.gf = df.copy()
        self.idvar = idvar

        # Checking that treatment is binary
        if df[exposure].dropna().value_counts().index.isin([0, 1]).all():
            self.exposure = exposure
        else:
            raise ValueError('Only binary treatments/exposures are currently implemented')

        # Checking that outcome is binary
        if df[outcome].dropna().value_counts().index.isin([0, 1]).all():
            self.outcome = outcome
        else:
            raise ValueError('Only binary outcomes are currently implemented')

        self.exp_model = None
        self.out_model = None
        self.time_out = time_out
        self._outcome_model_fit = False
        self._weights = weights
        self.marginal_outcome = None
        self._modelform = None
        self._printseqregresults = None

    def outcome_model(self, model, print_results=True):
        """Add a specified regression model for the outcome. Must be specified before the fit function.

        Parameters
        ----------
        model:
            Variables to include in the model for predicting the outcome. Must be contained within the input
            pandas dataframe when initialized. Format follows patsy standards
            For example) 'var1 + var2 + var3 + var4'
        print_results : bool, optional
            Whether to print the logistic regression model results to the terminal. Default is True
        """
        self._modelform = model
        self._printseqregresults = print_results
        self._outcome_model_fit = True

    def fit(self, treatment, t_max=None):
        """Estimate the counterfactual outcomes under the specified treatment plan using the previously specified
        regression models

        Parameters
        ----------
        treatment : str
            Treatment strategy. Options include
            * all : all individuals are given treatment
            * none : no individuals are given treatment
            * custom : create a custom treatment. When specifying this, the dataframe must be  referred to as 'g'
                The following is an example that selects those whose age is 25 or older and are females
                Ex) treatment="((g['age0']>=25) & (g['male']==0))
        t_max : int, optional
            Maximum time to run Monte Carlo g-formula until. Default is None, which uses the maximum time of the input
            dataframe.
        """
        if self._outcome_model_fit is False:
            raise ValueError('Before the g-formula can be calculated, the outcome model must be specified')
        if (type(treatment) != str) and (type(treatment) != list):
            raise ValueError('Specified treatment must be a string object')

        # TODO allow option to include different estimation models for each time point or the same model
        # If custom treatment, it gets evaluated here
        g = self.gf
        if treatment not in ['all', 'none']:
            g['__indicator'] = np.where(eval(treatment), 1, 0)

        # Restricting based on tmax argument
        if t_max is None:
            pass
        elif t_max in list(self.gf[self.time_out].unique()):
            g = g.loc[g[self.time_out] <= t_max].copy()
        else:
            warnings.warn("The t_max argument specifies a time that is not observed in the data. All times less than"
                          "the specified t_max argument included in the estimation procedure", UserWarning)
            g = g.loc[g[self.time_out] <= t_max].copy()

        # Converting dataframe from long-to-wide for easier estimation
        column_labels = list(g.columns)  # Getting all column labels (important to match with formula)
        df = self._long_to_wide(df=g, id=self.idvar, t=self.time_out)
        linkdist = sm.families.family.Binomial()
        rt_points = sorted(list(g[self.time_out].unique()), reverse=True)  # Getting all t's to backward loop
        t_points = sorted(list(g[self.time_out].unique()), reverse=False)  # Getting all t's to forward loop

        # Checking for recurrent outcomes. Recurrent are not currently supported
        if pd.Series(df[[self.outcome + '_' + str(t) for t in sorted(t_points, reverse=False)
                         ]].sum(axis=1, skipna=True) > 1).any():
            raise ValueError('Looks like your data has multiple outcomes. Recurrent outcomes are not currently '
                             'supported')

        # Step 1: Creating indicator for individuals who followed counterfactual outcome
        treat_t_points = []
        for t in t_points:
            # Following treatment strategy
            # alternative: if treat all, can do simple multiplication. if treat none, can do (1-A) simple multiplication
            if treatment == 'all':
                df['__indicator_' + str(t)] = np.where(df[self.exposure + '_' + str(t)] == 0, 0, np.nan)
                df['__indicator_' + str(t)] = np.where(df[self.exposure + '_' + str(t)] == 1, 1,
                                                       df['__indicator_' + str(t)])
            elif treatment == 'none':
                df['__indicator_' + str(t)] = np.where(df[self.exposure + '_' + str(t)] == 0, 1, np.nan)
                df['__indicator_' + str(t)] = np.where(df[self.exposure + '_' + str(t)] == 1, 0,
                                                       df['__indicator_' + str(t)])
            else:  # custom exposure pattern
                pass

            treat_t_points.append('__indicator_' + str(t))
            df['__check_' + str(t)] = df[treat_t_points + [self.outcome + '_' + str(t)]].prod(axis=1, skipna=True)

            # This following check carries forward the outcome under the counterfactual treatment
            if t_points.index(t) == 0:
                pass
            else:
                df['__check_' + str(t)] = np.where(df['__check_' + str(t_points[t_points.index(t) - 1])] == 1,
                                                   1, df['__check_' + str(t)])

        # Step 2: Sequential Regression Estimation
        for t in rt_points:
            # 2.1) Relabel everything to match with the specified model (selecting out that timepoint is within)
            d_labels = {}
            for c in column_labels:
                d_labels[c + '_' + str(t)] = c
            g = df.filter(regex='_' + str(t)).rename(mapper=d_labels, axis=1).reset_index().copy()
            g[self.time_out] = t

            # 2.2) Fit the model to the observed data
            if rt_points.index(t) == 0:
                if self._weights is None:
                    m = smf.glm(self.outcome + ' ~ ' + self._modelform, g, family=linkdist).fit()  # GLM
                else:
                    m = smf.gee(self.outcome + ' ~ ' + self._modelform, g[self.idvar], g,
                                weights=df[self._weights + '_' + str(t)], family=linkdist).fit()  # Weighted, so GEE
                if self._printseqregresults:
                    print(m.summary())
            else:
                # Uses previous predicted values to estimate
                g[self.outcome] = np.where(df['__pred_'+self.outcome+'_'+str(t_points[t_points.index(t)+1])].isna(),
                                           g[self.outcome],
                                           df['__pred_' + self.outcome + '_' + str(t_points[t_points.index(t)+1])])

                if self._weights is None:
                    m = smf.glm(self.outcome + ' ~ ' + self._modelform, g, family=linkdist).fit()  # GLM
                else:
                    m = smf.gee(self.outcome + ' ~ ' + self._modelform, g[self.idvar], g,
                                weights=df[self._weights + '_' + str(t)], family=linkdist).fit()  # Weighted, so GEE
                if self._printseqregresults:
                    print(m.summary())

            # 2.3) Getting Counterfactual Treatment Values
            # TODO need to think about how to allow for more complex treatments (lag_A + lag2_A...)
            if treatment == 'all':
                g[self.exposure] = 1
            elif treatment == 'none':
                g[self.exposure] = 0
            else:
                g[self.exposure] = np.where(eval(treatment), 1, 0)

            # Predicted values based on counterfactual treatment strategy from predicted model
            df['__pred_' + self.outcome + '_' + str(t)] = np.where(df[self.outcome + '_' + str(t)].isna(), np.nan,
                                                                   m.predict(g))
            # If followed counterfactual treatment & had outcome, then always considered to have outcome past that t
            df['__cf_' + self.outcome + '_' + str(t)] = np.where((df['__check_' + str(t)] == 1), 1,
                                                                 df['__pred_' + self.outcome + '_' + str(t)])

        # Step 3) Returning estimated results
        if self._weights is None:
            self.marginal_outcome = np.mean(df['__pred_' + self.outcome + '_' + str(t_points[0])])
        else:
            self.marginal_outcome = np.average(df['__pred_' + self.outcome + '_' + str(t_points[0])],
                                               weights=df[self._weights + '_' + str(t_points[0])])

    @staticmethod
    def _long_to_wide(df, id, t):
        """Hidden function for sequential regression that converts from long to wide data set
        """
        reshaped = []
        for c in df.columns:
            if c == id or c == t:
                pass
            else:
                df['v'] = c + '_' + df[t].astype(str)
                reshaped.append(df.pivot(index='id', columns='v', values=c))

        return pd.concat(reshaped, axis=1)
