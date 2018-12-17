import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import links


class TimeVaryGFormula:
    def __init__(self, df, idvar, exposure, outcome, time_out, time_in=None, method='MonteCarlo', weights=None):
        """Time-varying implementation of the g-formula, also referred to as the g-computation algorithm formula. The
        time-varying parametric g-formula uses either the Monte Carlo or the sequential regression (iterative
        expectations) estimators. The Monte Carlo estimator is useful for survival data and the sequential regression
        estimator is useful for longitudinal data. This implementation has four options for the treatment courses:

        Options for treatments
        * all : all individuals are given treatment
        * none : no individuals are given treatment
        * natural : individuals retain their observed treatment
        * custom : create a custom treatment. When specifying this, the dataframe must be  referred to as 'g' The following
            is an example that selects those whose age is 25 or older and are females
            Ex) treatment="((g['age0']>=25) & (g['male']==0))

        Currently, only binary exposures and a binary outcomes are supported. Logistic regression models are used to
        predict exposures and outcomes via statsmodels. See Keil et al. (2014) for a good description of the
        time-varying g-formula. See http://zepid.readthedocs.io/en/latest/ for an example (highly recommended)

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
        time_in : str, optional
            Start of follow-up period time label. This column is only required for the Monte Carlo estimator
        method : str, optional
            Estimator to use estimate the g-formula. Default is the Monte Carlo estimator. The Monte Carlo estimator is
            requested via 'MonteCarlo' and the sequential regression (iterative conditional) is requested with
            'SequentialRegression'
        weights : str, optional
            Column label for weights. Default is None, which assumes every observations has the same weight (i.e. 1)

        Notes
        -----
        1) Monte Carlo increases by time units of one. Input dataset should reflect this
        2) Only binary exposures and binary outcomes are supported
        3) Binary and continuous covariates are supported
        4) The labeling of the covariate models is important. They are fit in the order that they are labeled!
        5) For the Monte Carlo estimator, fit the natural course model first and compare to the observed data. They should
           be similar

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

        Process for the sequential regression g-formula
            1) Convert long data set to wide data set
            2) Identify individuals who followed the counterfactual treatment plan and had the outcome
            3) Fit a regression model for the outcome at time t for Y
            4) Predict outcomes under the observed treatment and the counterfactual treatment
            5) Repeat regression model fitting for t-1 to min(t)
            6) Take the mean predicted Y at the end to obtain the cumulative probability
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
        self.time_in = time_in
        self.time_out = time_out
        self._exposure_model_fit = False
        self._outcome_model_fit = False
        self._covariate_models = []
        self._covariate_model_index = []
        self._covariate = []
        self._covariate_type = []
        self._covariate_recode = []
        self._weights = weights
        self.predicted_outcomes = None

        # Different Estimator Approaches
        if method == 'MonteCarlo':
            self._mc = True
            if time_in is None:
                raise ValueError('"time_in" must be specified for MonteCarlo estimation')
        elif method == 'SequentialRegression':
            self._mc = False
            self._modelform = None
            self._printseqregresults = None
        else:
            raise ValueError('Either "MonteCarlo" or "SequentialRegression" must be specified as the estimation '
                             'procedure for TimeVaryGFormula')

    def exposure_model(self, model, restriction=None, print_results=True):
        """Add a specified regression model for the exposure. This is used for natural course estimation of the Monte
        Carlo g-formula. If the Monte Carlo estimation procedure is used, this must be specified before calling the
        fit function.

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
        """Add a specified regression model for the outcome. This is used for both Monte Carlo and sequential regression
        estimators and must be specified before the fit function.

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
        if self._mc:
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
        else:
            self._modelform = model
            self._printseqregresults = print_results

        self._outcome_model_fit = True

    def add_covariate_model(self, label, covariate, model, restriction=None, recode=None, var_type='binary',
                            print_results=True):
        """Add a specified regression model for time-varying confounders. Unlike the exposure and outcome models, a
        covariate model does NOT have to be specified. Additionally, *n* covariate models can be specified for *n*
        time-varying covariates. Additional models are added by repeated calls for this function with the corresponding
        covariates and predictive regression equations

        This argument is only used for the Monte Carlo g-formula. The sequential regression only requires specification
        of the outcome model.

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
                linkdist = sm.families.family.Gaussian(sm.families.links.identity)
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

    def fit(self, treatment, lags=None, sample=10000, t_max=None, in_recode=None, out_recode=None):
        """Estimate the counterfactual outcomes under the specified treatment plan using the previously specified
        regression models. For the Monte Carlo g-formula, both the exposure and outcome models need to be specified
        before fit can be called. For sequential regression, only the outcome model needs to be specified.

        Parameters
        ----------
        treatment : str
            Treatment strategy. Options include
            * all : all individuals are given treatment
            * none : no individuals are given treatment
            * natural : individuals retain their observed treatment. Only available for MonteCarlo
            * custom : create a custom treatment. When specifying this, the dataframe must be  referred to as 'g'
                The following is an example that selects those whose age is 25 or older and are females
                Ex) treatment="((g['age0']>=25) & (g['male']==0))
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
        """
        if self._outcome_model_fit is False:
            raise ValueError('Before the g-formula can be calculated, the outcome model must be specified')
        if (type(treatment) != str) and (type(treatment) != list):
            raise ValueError('Specified treatment must be a string object')

        # Monte Carlo Estimation
        if self._mc:
            if self._exposure_model_fit is False:
                raise ValueError('Before the g-formula can be calculated, the exposure model must be specified for '
                                 'Monte Carlo estimation')
            # Getting data all set
            if self._weights is None:
                gs = self.gf.loc[(self.gf.groupby(self.idvar).cumcount() == 0) == True].sample(n=sample, replace=True)
            else:
                gs = self.gf.loc[(self.gf.groupby(self.idvar).cumcount() == 0) == True].sample(n=sample,
                                                                                               weights=self._weights,
                                                                                               replace=True)
            gs['uid_g_zepid'] = [v for v in range(sample)]

            # Background preparations
            gs[self.outcome] = 0

            # getting maximum time steps to run g-formula
            if t_max is None:
                t_max = np.max(self.gf[self.time_out])

            # Estimating via MC process
            self.predicted_outcomes = self._monte_carlo(gs, treatment, t_max, in_recode, out_recode, lags)

        # Sequential Regression Estimation
        else:
            if self._exposure_model_fit or len(self._covariate_models) > 0:
                raise ValueError('Only the outcome model needs be specified for the sequential regression estimator')
            self.predicted_outcomes = self._sequential_regression(treatment=treatment, tmax=t_max)

    def _monte_carlo(self, gs, treatment, t_max, in_recode, out_recode, lags):
        """Hidden function that executes the Monte Carlo estimation process for the g-formula
        """
        # setting up some parts outside of Monte Carlo loop to speed things up
        mc_simulated_data = []
        g = gs.copy()
        if len(self._covariate_models) != 0:
            cov_model_order = (sorted(range(len(self._covariate_model_index)),
                                      key=self._covariate_model_index.__getitem__))
            run_cov = True
        else:
            run_cov = False

        # Monte Carlo for loop
        for i in range(int(t_max)):
            g = g.loc[g[self.outcome] == 0].reset_index(drop=True).copy()
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

            # executing any code before appending
            if out_recode is not None:
                exec(out_recode)

            # updating lagged variables
            if lags is not None:
                for k, v in lags.items():
                    g[v] = g[k]

            # stacking simulated data in a list
            mc_simulated_data.append(g)

        try:
            gs = pd.concat(mc_simulated_data, ignore_index=True, sort=False)
        except TypeError:  # gets around pandas <0.22 error
            gs = pd.concat(mc_simulated_data, ignore_index=True)
        if self._weights is None:
            return gs[['uid_g_zepid', self.exposure, self.outcome, self.time_in, self.time_out] +
                      self._covariate].sort_values(by=['uid_g_zepid',
                                                       self.time_in]).reset_index(drop=True)
        else:
            return gs[['uid_g_zepid', self.exposure, self.outcome, self._weights, self.time_in,
                       self.time_out] + self._covariate].sort_values(by=['uid_g_zepid',
                                                                         self.time_in]).reset_index(drop=True)

    def _sequential_regression(self, treatment, tmax):
        """Hidden function that executes the sequential regression estimation for g-formula
        """
        # TODO allow option to include different estimation models for each time point or the same model
        if treatment == 'natural':
            # Thoughts: MC estimator needs natural course as a check. This should not apply to SR estimator
            raise ValueError('Natural course estimation is not clear to me with Sequential Regression Estimator. '
                             'Therefore, "natural" is not implemented')

        # If custom treatment, it gets evaluated here
        g = self.gf
        if treatment not in ['all', 'none']:
            g['__indicator'] = np.where(eval(treatment), 1, 0)

        # Restricting based on tmax argument
        if tmax is None:
            pass
        elif tmax in list(self.gf[self.time_out].unique()):
            g = g.loc[g[self.time_out] <= tmax].copy()
        else:
            warnings.warn("The t_max argument specifies a time that is not observed in the data. All times less than"
                          "the specified t_max argument included in the estimation procedure", UserWarning)
            g = g.loc[g[self.time_out] <= tmax].copy()

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
                    m = smf.gee(self.outcome + ' ~ ' + self._modelform, self.idvar, g,
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
                    m = smf.gee(self.outcome + ' ~ ' + self._modelform, self.idvar, g,
                                weights=df[self._weights + '_' + str(t)], family=linkdist).fit()  # Weighted, so GEE
                if self._printseqregresults:
                    print(m.summary())

            # 2.3) Getting Counterfactual Treatment Values
            if treatment == 'all':
                g[self.exposure] = 1
            elif treatment == 'none':
                g[self.exposure] = 0
            else:
                g[self.exposure] = np.where(eval(treatment), 1, 0)

            # Predicted values based on counterfactual treatment strategy from predicted model
            df['__pred_' + self.outcome + '_' + str(t)] = np.where(df[self.outcome + '_' + str(t)].isna(),
                                                                   np.nan,
                                                                   m.predict(g))
            # If followed counterfactual treatment & had outcome, then always considered to have outcome past that t
            df['__cf_' + self.outcome + '_' + str(t)] = np.where((df['__check_' + str(t)] == 1),
                                                                 1,
                                                                 df['__pred_' + self.outcome + '_' + str(t)])

        # Step 3) Returning estimated results
        if self._weights is None:
            return np.mean(df['__pred_' + self.outcome + '_' + str(t_points[0])])
        else:
            return np.average(df['__pred_' + self.outcome + '_' + str(t_points[0])],
                              weights=df[self._weights + '_' + str(t_points[0])])

    @staticmethod
    def _predict(df, model, variable):
        """Hidden predict method to shorten the Monte Carlo estimation code
        """
        # pp = data.mul(model.params).sum(axis=1) # Alternative to statsmodels.predict(), but too much too implement
        pp = model.predict(df)
        if variable == 'binary':
            # pp = odds_to_probability(np.exp(pp))  # assumes a logit model. For non-statsmodel.predict() option
            pred = np.random.binomial(1, pp, size=len(pp))
        elif variable == 'continuous':
            pred = np.random.normal(loc=pp, scale=np.std(model.resid), size=len(pp))
        else:
            raise ValueError('That option is not supported')
        return pred

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

