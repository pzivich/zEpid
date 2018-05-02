import warnings
import math 
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import family
from statsmodels.genmod.families import links
from zepid.ipw.ipw import propensity_score

class SimpleDoublyRobust:
    '''Implementation of a simple doubly robust estimator as described in Funk et al. American Journal
    of Epidemiology 2011;173(7):761-767. The exact formulas used are available in Table 1 and Equation 1.
    Properties of the doubly robust estimator are often misunderstood, so we direct the reader to 
    Keil et al. American Journal of Epidemiology 2018;187(4):891-892. SimplyDoublyRobust only supports a 
    binary outcome. Also this model does not deal with missing data. This is only meant to be a simple 
    implementation of the doubly robust estimator
    
    df:
        -pandas DataFrame object containing all variables of interest
    exposure:
        -column name of the exposure variable. Currently only binary is supported
    outcome:
        -column name of the outcome variable. Currently only binary is supported
    '''
    def __init__(self,df,exposure,outcome):
        self.df = df.copy()
        self.exposure = exposure
        self.outcome = outcome
        self.exposure_model = False
        self.outcome_model = False
        self.generated_ci = False
    
    def exposure_model(self,model,mresult=True):
        '''Used to specify the propensity score model. Model used to predict the exposure via a 
        logistic regression model
        
        model:
            -Independent variables to predict the exposure. Example) 'var1 + var2 + var3'
        mresult:
            -Whether to print the fitted model results. Default is True (prints results)
        '''
        self.exposure_model = self.exposure + ' ~ '+ model
        self.df['ps'] = propensity_score(self.df,self.exposure_model,mresult=mresult)
        
    def outcome_model(self,model,mresult=True):
        '''Used to specify the outcome model. Model used to predict the outcome via a logistic
        regression model
        
        model:
            -Independent variables to predict the outcome. Example) 'var1 + var2 + var3 + var4'
        mresult:
            -Whether to print the fitted model results. Default is True (prints results)
        '''
        self.outcome_model = self.outcome + ' ~ '+ model
        self.df['pY'] = propensity_score(self.df,self.outcome_model,mresult=mresult)
    
    def fit(self):
        '''Once the exposure and outcome models are specified, we can estimate the Risk Ratio 
        and Risk Difference. This function generates the estimated risk difference and risk ratio. 
        To view results, use SimpleDoublyRobust.summary() For confidence intervals, bootstrap() must
        be run, otherwise only point estimates will be generated
        '''
        if ((self.exposure_model == False) or (self.exposure_model == False)):
            raise ValueError('The exposure and outcome models must be specified before the doubly robust estimate can be generated')
        
        #Doubly robust estimator for exposed
        self.df['dr1'] = np.where(self.df[self.exposure]==1,
                                  ((self.df[self.outcome])/self.df['ps'] - 
                                        ((self.df['pY'] * (1 - self.df['ps'])) / (self.df['ps']))),
                                  self.df['pY'])
        
        
        #Doubly robust estimator for unexposed
        self.df['dr0'] = np.where(self.df[self.exposure]==0,
                                  (1 - self.df['pY']),
                                  ((1 - self.df[self.outcome])/(1 - self.df['ps']) - 
                                        (((1 -self.df['pY']) * (self.df['ps'])) / (1 - self.df['ps']))))
        
        #Generating estimates for the risk difference and risk ratio
        self.riskdiff = np.mean(self.df['dr1']) - np.mean(self.df['dr0'])
        self.riskratio = np.mean(self.df['dr1']) / np.mean(self.df['dr0'])
    
    def bootstrap(self,alpha=0.05,b_sample=250,seed=None):
        '''Used to generate bootstrapped confidence intervals
        '''
        self.b_sample = b_sample
        self.alpha = alpha
        
        #Resampling to generate bootstrapped confidence intervals
        rd_sample = []
        rr_sample = []
        for i in range(b_sample):
            sf = self.df.sample(n=self.df.shape[0],replace=True,random_state=seed)
            sf['ps'] = propensity_score(sf,self.exposure_model,mresult=False)
            sf['pY'] = propensity_score(sf,self.outcome_model,mresult=False)
            sf['dr1'] = np.where(sf[self.exposure]==1,
                                  ((sf[self.outcome])/sf['ps'] - ((sf['pY'] * (1 - sf['ps'])) / (sf['ps']))),
                                  sf['pY'])
            sf['dr0'] = np.where(sf[self.exposure]==0,
                                  (1 - sf['pY']),
                                  ((1 - sf[self.outcome])/(1 - sf['ps']) - (((1 -sf['pY']) * (sf['ps'])) / (1 - sf['ps']))))
            rd_sample.append(np.mean(sf['dr1']) - np.mean(sf['dr0']))
            rr_sample.append(np.mean(sf['dr1']) / np.mean(sf['dr0']))
        
        #Extracting percentiles to generate the confidence intervals
        self.rd_lower = np.percentile(rd_sample, (alpha/2))
        self.rd_upper = np.percentile(rd_sample, (1-alpha/2))
        self.rd_lower = np.percentile(rr_sample, (alpha/2))
        self.rd_upper = np.percentile(rr_sample, (1-alpha/2))
        self.generated_ci = True

    def summary(self):
        '''Prints a summary of the results for the doubly robust estimator. 
        '''
        print('----------------------------------------------------------------------')
        print('Risk Difference: ',round(self.riskdifference,4))
        if self.generated_ci == True:
            print(str(round(self.alpha,1))+'% CI: ',self.rd_lower,', ',self.rd_upper)
        print('Risk Ratio: ',round(self.riskratio,4))
        if self.generated_ci == True:
            print(str(round(self.alpha,1))+'% CI: ',self.rr_lower,', ',self.rr_upper)
        print('----------------------------------------------------------------------')
        if self.generated_ci == True:
            print('Confidence intervals were generated from nonparametric bootstraps')
            print('Number of resamplings for bootstraps = ',self.b_sample)
            print('----------------------------------------------------------------------')

    
