import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import links


class TimeVaryGFormula:
    """Time-varying implementation of the g-formula, also referred to as the g-computation
    algorithm formula. This implementation has four options for the treatment courses:
    
    Key options for treatments
        all     -all individuals are given treatment
        none    -no individuals are given treatment
        natural -individuals retain their observed treatment
        custom treatment
                -create a custom treatment. When specifying this, the dataframe must be 
                 referred to as 'g' The following is an example that selects those whose
                 age is 25 or older and are females
                 ex) treatment="((g['age0']>=25) & (g['male']==0))
    
    Currently, only supports a binary exposure and a binary outcome. Uses a logistic regression model to predict
    exposures and outcomes via statsmodels. See Keil et al. (2014) for a good description of the time-varying
    g-formula. See http://zepid.readthedocs.io/en/latest/ for an example (highly recommended)
    
    IMPORTANT CONSIDERATIONS:
    1) TimeVaryGFormula increases by time unit increase of one. Your input dataset should reflect this
    2) Only binary exposures and binary outcomes are supported
    3) Binary and continuous covariates are supported
    4) The labeling of the covariate models is important. They are fit in the order that they are labeled!
    5) Fit the natural course model first and compare to the observed data. They should be similar
    6) Check to make sure the predicted values are reasonable. If not, you may need to code in restrictions into
        'restriction', 'recode', or 'in_recode' statements in the fitted models

    Process for the g-formula monte carlo:
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

    Inputs:
    df:
        -pandas dataframe containing the variables of interest
    exposure:
        -exposure variable label / column name
    outcome:
        -outcome variable label / column name
    time_in:
        -time variable label / column name for start of row time period
    time_out:
        -time variable label / column name for end of row time period
    weights:
        -weights for weighted data. Default is None, which assumes every observations has the same weight (i.e. 1)
    """

    def __init__(self, df, idvar, exposure, outcome, time_out, time_in=None, method='MonteCarlo', weights=None):
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
        """Build the model for the exposure. This must be specified before the fit function. If it is not,
        an error will be raised.
        
        model:
            -variables to include in the model for predicting the outcome. Must be contained within the input
             pandas dataframe when initialized. Format is the same as the functional form
             Example) 'var1 + var2 + var3 + var4'
        restriction:
            -used to restrict the population to fit the logistic regression model to. Useful for Intent-to-Treat
             model fitting. The pandas dataframe must be referred to as 'g'
             Example) "g['art']==1"
        print_results:
            -whether to print the logistic regression results to the terminal. Default is True
        """
        g = self.gf.copy()
        if restriction is not None:
            g = g.loc[eval(restriction)].copy()
        linkdist = sm.families.family.Binomial(sm.families.links.logit)

        if self._weights is None:  # Unweighted g-formula
            self.exp_model = smf.glm(self.exposure + ' ~ ' + model, g, family=linkdist).fit()
        else:  # Weighted g-formula
            self.exp_model = smf.gee(self.exposure + ' ~ ' + model, self.idvar, g, weights=g[self._weights],
                                     family=linkdist).fit()

        if print_results:
            print(self.exp_model.summary())
        self._exposure_model_fit = True

    def outcome_model(self, model, restriction=None, print_results=True):
        """Build the model for the outcome. This must be specified before the fit function. If it is not,
        an error will be raised.
        
        Input:
        
        model:
            -variables to include in the model for predicting the outcome. Must be contained within the input
             pandas dataframe when initialized. Format is the same as the functional form,
             i.e. 'var1 + var2 + var3 + var4'
        restriction:
            -used to restrict the population to fit the logistic regression model to. Useful for Intent-to-Treat
             model fitting. The pandas dataframe must be referred to as 'g'
             Example) "g['art']==1"
        print_results:
            -whether to print the logistic regression results to the terminal. Default is True
        """
        if self._mc:
            g = self.gf.copy()
            if restriction is not None:
                g = g.loc[eval(restriction)].copy()
            linkdist = sm.families.family.Binomial(sm.families.links.logit)

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
        """
        Build the model for the specified covariate. This is to deal with time-varying confounders.
        Does NOT have to be specified, unlike the exposure and outcome models. The order in which these
        models are fit is based on the provided integer labels
        
        Input:
        
        label:
            -integer label for the covariate model. Covariate models are fit in ascending order within 
             TimeVaryGFormula
        covariate:
            -variable to be predicted
        model:
            -variables to include in the model for predicting the outcome. Must be contained within the input
             pandas dataframe when initialized. Format is the same as the functional form,
             i.e. 'var1 + var2 + var3 + var4'
        restriction:
            -used to restrict the population to fit the logistic regression model to. Useful for Intent-to-Treat
             model fitting. The pandas dataframe must be referred to as 'g'
             Example) "g['art']==1"
        recode:
            -This variable is vitally important for various functional forms implemented later in models. This
             is used to run some background code to recreate functional forms as the g-formula is fit via fit()
             For an example, let's say we have age but we want the functional form to be cubic. For this, we 
             would set the recode="g['']" Similar to TimeFixedGFormula, 'g' must be specified as the data frame 
             object with the corresponding indexes. Also lines of executable code should end with ';', so Python
             knows that the line ends there. My apologies for this poor solution... I am working on a better way
        var_type:
            -type of variable that the covariate is. Current options include 'binary' or 'continuous'
        print_results:
            -whether to print the logistic regression results to the terminal. Default is True
        """
        if type(label) is not int:
            raise ValueError('Label must be an integer')

        # Building predictive model
        g = self.gf.copy()
        if restriction is not None:
            g = g.loc[eval(restriction)].copy()

        if self._weights is None:  # Unweighted g-formula
            if var_type == 'binary':
                linkdist = sm.families.family.Binomial(sm.families.links.logit)
                m = smf.glm(covariate + ' ~ ' + model, g, family=linkdist)
            elif var_type == 'continuous':
                linkdist = sm.families.family.Gaussian(sm.families.links.identity)
                m = smf.gls(covariate + ' ~ ' + model, g)
            else:
                raise ValueError('Only binary or continuous covariates are currently supported')
        else:  # Weighted g-formula
            if var_type == 'binary':
                linkdist = sm.families.family.Binomial(sm.families.links.logit)
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
        """
        Implementation of fit
        
        treatment:
            -treatment strategy
        lags:
            -variables to generate a lagged variable for. It should be a dictionary with the variable name
             as the key and the lagged variable name as the value.
             Ex) {'art':'lagged_art'}
        sample:
            -number of individuals to sample from the original data with replacement. It should be a large 
             number. Default is 10000
        t_max:
            -maximum time to run g-formula until. Default is None, which uses the maximum time of the input 
             dataframe. Input should be integer (but will also accept floats)
        in_recode:
            -on the fly recoding of variables done before the loop starts. Needed to do any kind of functional
             forms for entry times
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
            self.predicted_outcomes = self._sequential_regression(treatment=treatment)

    def _monte_carlo(self, gs, treatment, t_max, in_recode, out_recode, lags):
        """Monte Carlo estimation process for the g-formula
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

    def _sequential_regression(self, treatment):
        """Sequential regression estimation for g-formula
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

        # Converting dataframe from long-to-wide for easier estimation
        column_labels = list(g.columns)  # Getting all column labels (important to match with formula)
        df = self._long_to_wide(df=g, id=self.idvar, t=self.time_out)
        linkdist = sm.families.family.Binomial(sm.families.links.logit)
        rt_points = sorted(list(self.gf[self.time_out].unique()), reverse=True)  # Getting all t's to backward loop
        t_points = sorted(list(self.gf[self.time_out].unique()), reverse=False)  # Getting all t's to forward loop

        # Checking for recurrent outcomes. Recurrent are not currently supported
        if pd.Series(df[[self.outcome + '_' + str(t) for t in sorted(t_points, reverse=False)
                         ]].sum(axis=1, skipna=True) > 1).any():
            raise ValueError('Looks like your data has multiple outcomes. Recurrent outcomes are not supported')

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
        results = pd.DataFrame()
        for t in rt_points:
            # 2.1) Relabel everything to match with the specified model (selecting out that timepoint is within)
            d_labels = {}
            for c in column_labels:
                d_labels[c + '_' + str(t)] = c
            g = df.filter(regex='_' + str(t)).rename(mapper=d_labels, axis=1).reset_index().copy()

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
                g[self.outcome] = df['__pred_' + self.outcome + '_' + str(t_points[t_points.index(t)+1])].copy()

                if self._weights is None:
                    m = smf.glm(self.outcome + ' ~ ' + self._modelform, g, family=linkdist).fit()  # GLM
                else:
                    m = smf.gee(self.outcome + ' ~ ' + self._modelform, self.idvar, g,
                                weights=df[self._weights + '_' + str(t)], family=linkdist).fit()  # Weighted, so GEE
                if self._printseqregresults:
                    print(m.summary())

            # 2.3) Getting Predicted values out
            if treatment == 'all':
                g[self.exposure] = 1
            elif treatment == 'none':
                g[self.exposure] = 0
            else:
                g[self.exposure] = np.where(eval(treatment), 1, 0)

            # Predicted values based on counterfactual treatment strategy from predicted model
            df['__pred_' + self.outcome + '_' + str(t)] = np.where(df[self.outcome + '_' + str(t)].isna(),
                                                                   np.nan, m.predict(g))
            # If followed counterfactual treatment & had outcome, then always considered to have outcome past that t
            df['__pred_' + self.outcome + '_' + str(t)] = np.where((df['__check_' + str(t)] == 1) &
                                                                   df[self.outcome + '_' + str(t)].isna(),
                                                                   1, df['__pred_' + self.outcome + '_' + str(t)])

            # 2.4) Extracting E[Y] for each time point
            q = df.dropna(subset=['__pred_' + self.outcome + '_' + str(t)]).copy()
            if self._weights is None:
                results['Q' + str(t)] = [np.mean(q['__pred_' + self.outcome + '_' + str(t)])]
            else:
                results['Q' + str(t)] = [np.average(q['__pred_' + self.outcome + '_' + str(t)],
                                                    weights=q[self._weights + str(t)])]
        # Step 3) Returning estimated results
        if len(t_points) == 1:
            return results
        else:
            return results.squeeze().sort_index()

    @staticmethod
    def _predict(df, model, variable):
        """Predict method to shorten the _montecarlo_est procedure code"""
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
        """Converts from long to wide dataframe for sequential regression estimation
        """
        reshaped = []
        for c in df.columns:
            if c == id or c == t:
                pass
            else:
                df['v'] = c + '_' + df[t].astype(str)
                reshaped.append(df.pivot(index='id', columns='v', values=c))

        return pd.concat(reshaped, axis=1)

