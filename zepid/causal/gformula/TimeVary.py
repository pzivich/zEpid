import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import links
from zepid.calc import odds_to_prop


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
    time:
        -time variable label / column name
    """

    def __init__(self, df, idvar, exposure, outcome, time_in, time_out):
        self.gf = df.copy()
        self.idvar = idvar
        self.exposure = exposure
        self.outcome = outcome
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
        self._covariate_se = []
        self.predicted_outcomes = None

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
        if ('np.' in model) or ('*' in model) or (':' in model) or ('**' in model) or ('I(' in model):
            raise ValueError('Due to the need to speed up some background processes, certain patsy process are not '
                             'supported. Please make sure to recode all variables for the regression models, the '
                             'input dataframe, and all the recodes during the Monte Carlo loop')
        g = self.gf.copy()
        if restriction is not None:
            g = g.loc[eval(restriction)].copy()
        linkdist = sm.families.family.Binomial(sm.families.links.logit)
        self.exp_model = smf.glm(self.exposure + ' ~ ' + model, g, family=linkdist).fit()
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
        if ('np.' in model) or ('*' in model) or (':' in model) or ('**' in model) or ('I(' in model):
            raise ValueError('Due to the need to speed up some background processes, certain patsy process are not '
                             'supported. Please make sure to recode all variables for the regression models, the '
                             'input dataframe, and all the recodes during the Monte Carlo loop')
        g = self.gf.copy()
        if restriction is not None:
            g = g.loc[eval(restriction)].copy()
        linkdist = sm.families.family.Binomial(sm.families.links.logit)
        self.out_model = smf.glm(self.outcome + ' ~ ' + model, g, family=linkdist).fit()
        if print_results:
            print(self.out_model.summary())
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
        if ('np.' in model) or ('*' in model) or (':' in model) or ('**' in model) or ('I(' in model):
            raise ValueError('Due to the need to speed up some background processes, certain patsy process are not '
                             'supported. Please make sure to recode all variables for the regression models, the '
                             'input dataframe, and all the recodes during the Monte Carlo loop')
        if type(label) is not int:
            raise ValueError('Label must be an integer')

        # Building predictive model
        g = self.gf.copy()
        if restriction is not None:
            g = g.loc[eval(restriction)].copy()
        if var_type == 'binary':
            linkdist = sm.families.family.Binomial(sm.families.links.logit)
            m = smf.glm(covariate + ' ~ ' + model, g, family=linkdist)
            f = m.fit()
            self._covariate_se.append(None)
        elif var_type == 'continuous':
            m = smf.gls(covariate + ' ~ ' + model, g)
            f = m.fit()
            self._covariate_se.append(np.std(f.resid))
        else:
            raise ValueError('Only binary or continuous covariates are currently supported')

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
        if self._exposure_model_fit is False:
            raise ValueError('Before the g-formula can be calculated, the exposure model must be specified')
        if self._outcome_model_fit is False:
            raise ValueError('Before the g-formula can be calculated, the outcome model must be specified')
        if (type(treatment) != str) and (type(treatment) != list):
            raise ValueError('Specified treatment must be a string object')

        # Getting data all set
        gs = self.gf.loc[(self.gf.groupby(self.idvar).cumcount() == 0) == True].sample(n=sample, replace=True)
        gs['uid_g_zepid'] = [v for v in range(sample)]

        # Background preparations
        gs[self.outcome] = 0

        # getting maximum time steps to run g-formula
        if t_max is None:
            t_max = np.max(self.gf[self.time_out])

        # setting up some parts outside of Monte Carlo loop to speed things up
        mc_simulated_data = []
        g = gs.copy()
        g['Intercept'] = 1  # this keeps intercept term for _predict()
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
                    g[self._covariate[j]] = self._predict(df=g,
                                                          model=self._covariate_models[j],
                                                          variable=self._covariate_type[j],
                                                          se=self._covariate_se[j])
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

        try:  # work-around for 0.23.0 update
            gs = pd.concat(mc_simulated_data, ignore_index=True, sort=False)  # concatenating all that stacked data
        except TypeError:
            gs = pd.concat(mc_simulated_data, ignore_index=True)  # concatenating all that stacked data

        self.predicted_outcomes = gs[['uid_g_zepid', self.exposure, self.outcome,
                                      self.time_in, self.time_out] +
                                     self._covariate].sort_values(by=['uid_g_zepid',
                                                                      self.time_in]).reset_index(drop=True)

    @staticmethod
    def _predict(df, model, variable, se=None):
        """
        This predict method gains me a small ammount of increased speed each time a model is fit, compared to
        statsmodels.predict(). Because this is repeated so much, it actually decreases time a fair bit
        """
        # Commented out lines are compatible with patsy/statsmodels BUT are slower
        if variable == 'binary':
            pred = np.random.binomial(1,
                                      odds_to_prop(np.exp(df.mul(model.params).sum(axis=1))),
                                      size=df.shape[0])
            # model.predict(df),
            # size=df.shape[0])
        else:
            pred = df.mul(model.params).sum(axis=1).add(se*np.random.normal(loc=0, scale=1, size=df.shape[0]))
            # pred = model.predict(df).add(se*np.random.normal(loc=0, scale=1, size=df.shape[0]))
        return pred
