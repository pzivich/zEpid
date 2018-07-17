import warnings
import numpy as np 
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import family
from statsmodels.genmod.families import links

class TimeVaryGFormula:
    '''Time-varying implementation of the g-formula, also referred to as the g-computation
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
    
    Currently, only supports a binary exposure and a binary outcome. Uses a logistic regression
    model to predict outcomes via statsmodels. See Keil et al. (2014) for a good description of 
    the time-varying g-formula. See http://zepid.readthedocs.io/en/latest/ for an example 
    (highly recommended)
    
    IMPORTANT CONSIDERATIONS:
    1) TimeVaryGFormula increases by time unit increase of one. Your input dataset should reflect this
    2) Only binary exposures and binary outcomes are supported
    3) Only binary and continuous covariates are supported
    4) The labeling of the covariate models is important. They are fit in the order that they are labeled!
    5) Fit the natural course model first and compare to the observed data. They should be similar
    
    Inputs:
    df:
        -pandas dataframe containing the variables of interest
    exposure:
        -exposure variable label / column name
    outcome:
        -outcome variable label / column name
    time:
        -time variable label / column name
    '''
    def __init__(self,df,idvar,exposure,outcome,time_in,time_out):
        warnings.warn("This implementation is experimental at this state. I wouldn't use it yet")
        self.gf = df.copy()
        self.idvar = idvar
        self.exposure = exposure
        self.outcome = outcome
        self.time_in = time_in
        self.time_out = time_out
        self._exposure_model_fit = False
        self._outcome_model_fit = False
        self._covariate_models = []
        self._covariate_model_index = []
        self._covariate = []
        self._covariate_type = []
        self._covariate_recode = []

    def exposure_model(self,model,print_results=True):
        '''Build the model for the exposure. This must be specified before the fit function. If it is not, 
        an error will be raised.
        
        model:
            -variables to include in the model for predicting the outcome. Must be contained within the input
             pandas dataframe when initialized. Format is the same as the functional form,
             i.e. 'var1 + var2 + var3 + var4'
        print_results:
            -whether to print the logistic regression results to the terminal. Default is True
        '''
        linkdist=sm.families.family.Binomial(sm.families.links.logit)
        self.exposure_model = smf.glm(self.exposure+' ~ '+model,self.gf,family=linkdist).fit()
        if print_results == True:
            print(self.exposure_model.summary())
        self._exposure_model_fit = True

    def outcome_model(self,model,print_results=True):
        '''Build the model for the outcome. This must be specified before the fit function. If it is not, 
        an error will be raised.
        
        Input:
        
        model:
            -variables to include in the model for predicting the outcome. Must be contained within the input
             pandas dataframe when initialized. Format is the same as the functional form,
             i.e. 'var1 + var2 + var3 + var4'
        print_results:
            -whether to print the logistic regression results to the terminal. Default is True
        '''
        linkdist=sm.families.family.Binomial(sm.families.links.logit)
        self.outcome_model = smf.glm(self.outcome+' ~ '+model,self.gf,family=linkdist).fit()
        if print_results == True:
            print(self.outcome_model.summary())
        self._outcome_model_fit = True
    
    def add_covariate_model(self,label,covariate,model,recode=None,var_type='binary',print_results=True):
        '''Build the model for the specified covariate. This is to deal with time-varying confounders. 
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
        recode:
            -This variable is vitally important for various functional forms implemented later in models. This
             is used to run some background code to recreate functional forms as the g-formula is fit via fit()
             For an example, let's say we have age but we want the functional form to be cubic. For this, we 
             would set the recode="g['']" Similar to TimeFixedGFormula, 'g' must be specified as the data frame 
             object with the corresponding indeces. Also lines of executable code should end with ';', so Python
             knows that the line ends there. My apologies for this poor solution... I am working on a better way
        var_type:
            -type of variable that the covariate is. Current options include 'binary' or 'continuous'
        print_results:
            -whether to print the logistic regression results to the terminal. Default is True
        '''
        if type(label) != int:
            raise ValueError('Label must be an integer')
        
        #Building predictive model
        if var_type == 'binary':
            linkdist=sm.families.family.Binomial(sm.families.links.logit)
        elif var_type == 'continuous':
            linkdist=sm.families.family.Gaussian(sm.families.links.identity)
        else:
            raise ValueError('Only binary or continuous covariates are currently supported')

        m = smf.glm(covariate+' ~ '+model,self.gf,family=linkdist)
        f = m.fit()
        if print_results == True:
            print(f.summary())

        #Adding to lists, it is used to predict variables later on for the time-varying...
        self._covariate_models.append(f)
        self._covariate_model_index.append(label)
        self._covariate.append(covariate)
        self._covariate_type.append(var_type)
        if recode == None:
            self._covariate_recode.append('None') #Must be string for exec() to use later
        else:
            self._covariate_recode.append(recode)
        
    def fit(self,treatment,lags,sample=10000,t_max=None,in_recode=None):
        '''Implementation of fit
        
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
        '''
        #Checking to make sure everything is specified correctly.
        if self._exposure_model_fit == False:
            raise ValueError('Before the g-formula can be calculated, the exposure model must be specified')
        if self._outcome_model_fit == False:
            raise ValueError('Before the g-formula can be calculated, the outcome model must be specified')
        if (type(treatment) != str) and (type(treatment) != list):
            raise ValueError('Specified treatment must be a string object')
        
        #Getting data all set
        gf = self.gf.sort_values(by=[self.idvar,self.time_in]).copy()
        gf1 = gf.loc[(gf.groupby(self.idvar).cumcount()==0)==True] #allows late-entries by default
        gc = gf1.sample(n=sample,replace=True)
        gc['uid_g_zepid'] = [v for v in range(sample)]
        gs = gc.copy()
        
        #Background preparations
        gs[self.outcome] = 0
        if type(treatment) == list:
            raise ValueError('Only binary exposures are currently supported by TimeVaryGFormula')
        elif treatment == 'all':
            gs[self.exposure] = 1
        elif treatment == 'none':
            gs[self.exposure] = 0
        elif treatment == 'natural':
            pass #keep same treatments if no difference
        else: #custom exposure pattern
            gs[self.exposure] = np.where(eval(treatment),1,0)
        
        #getting maximum time steps to run g-formula
        if t_max == None:
            t_max = np.max(self.gf[self.time_out])
        
        #fitting g-formula piece by piece
        for i in range(int(t_max)):
            #select out last observation that has not had outcome yet
            g = gs.loc[(gs[self.outcome]==0) & (gs.groupby('uid_g_zepid').cumcount()==i)].copy()
            g[self.time_in] = i
            if in_recode != None:
                exec(in_recode)
            #predict all other covariates
            if len(self._covariate_models) != 0:
                for j in (sorted(range(len(self._covariate_model_index)),
                                    key=self._covariate_model_index.__getitem__)): #get list index by ascending order
                    g[self._covariate[j]+'pred'] = self._covariate_models[j].predict(g)
                    if self._covariate_type[j] == 'binary':
                        g[self._covariate[j]] = np.random.binomial(1,g[self._covariate[j]+'pred'],size=g.shape[0])
                    if self._covariate_type[j] == 'continuous':
                        if hasattr((self._covariate_models[j]), "bse"): #used to deal with statsmodels upcoming change
                            np.random.normal(loc=g[self._covariate[j]+'pred'],
                                             scale=0.4141, #Need to fix SE bit...
                                             size=g.shape[0])
                        else:
                            np.random.normal(loc=g[self._covariate[j]+'pred'],
                                             scale=self._covariate_models[j].stand_errors[self._covariate[j]],
                                             size=g.shape[0])
                    exec(self._covariate_recode[j])
            #predict exposure when customized treatments
            if (treatment == 'all') or (treatment == 'none'):
                pass
            elif treatment == 'natural':
                g[self.exposure+'pred'] = self.exposure_model.predict(g)
                g[self.exposure] = np.random.binomial(1,g[self.exposure+'pred'],size=g.shape[0])
            else: #custom exposure pattern
                g[self.exposure] = np.where(eval(treatment),1,0)
            #predict outcome
            g['pevent_zepid'] = self.outcome_model.predict(g)
            g[self.outcome] = np.random.binomial(1,g['pevent_zepid'],size=g.shape[0])
            for k,v in lags.items(): #creates lagged variables based on the predictions
                g[v] = g[k]
            #update time
            g[self.time_out] = i + 1
            
            #APPEND SIMULATED DATA 
            gs = gs.append(g,ignore_index=True)
            gs.sort_values(by=['uid_g_zepid',self.time_in],inplace=True)
        
        #so the append loop adds an extra, so we need to drop that extra
        gs.dropna(subset=['pevent_zepid'],inplace=True)
        
        #Gains new attributes
        self.predicted_outcomes = gs[['uid_g_zepid',self.outcome,self.time_in,
                            self.time_out]].sort_values(by=['uid_g_zepid',
                            self.time_in,self.time_out]).reset_index(drop=True)
        self.predicted_data_full = gs.sort_values(by=['uid_g_zepid',self.time_in]).reset_index(drop=True)
