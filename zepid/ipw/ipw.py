import warnings
import math 
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import family
from statsmodels.genmod.families import links
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def propensity_score(df,model,mresult=True):
    '''Generate propensity scores (probability) based on the model input. Uses logistic regression model 
    to calculate
    
    Returns a pandas Series of calculated probabilities for all observations
    within the dataframe
    
    df:
        -Dataframe for the model
    model:
        -Model to fit the logistic regression to. Example) 'y ~ var1 + var2'
    mresult:
        -Whether to print the logistic regression results. Default is True
    '''
    f = sm.families.family.Binomial(sm.families.links.logit) 
    log = smf.glm(model,df,family=f).fit()
    if mresult == True:
        print(log.summary())
    p = log.predict(df)
    return pd.Series(p)


def iptw(df,treatment,model_denominator,model_numerator=None,stabilized=True,standardize='population',return_probability=False):
        '''Calculates the weight for inverse probability of treatment weights through logistic regression. 
        Function can return either stabilized or unstabilized weights. Stabilization is based on model_numerator.
        Default is just to calculate the prevalence of the treatment in the population.
    
        Returns a pandas Series of the weights of observations. Probabilities can also be requested instead
        
        df: 
            -pandas dataframe object containing all variables of interest
        treatment:
            -Variable name of treatment variable of interest. Must be coded as binary.
             1 should indicate treatment, while 0 indicates no treatment
        model_denominator:
            -statsmodels glm format for modeling data. Only includes predictor variables Example) 'var1 + var2 + var3'
        model_numerator:
            -statsmodels glm format for modeling data. Only includes predictor variables for the numerator. Default (None)
             calculates the overall probability. In general this is recommended. If confounding variables are included in
             the numerator, they would later need to be adjusted for. Example) 'var1'
        stabilized:
            -Whether to return stabilized or unstabilized weights. Input is True/False. Default is stabilized weights (True)
        standardize:
            -Who to standardize the estimate to. Options are the entire population, the exposed, or the unexposed. 
             See Sato & Matsuyama Epidemiology (2003) for details on weighting to exposed/unexposed
             Options for standardization are:
                'population'    :   weight to entire population
                'exposed'       :   weight to exposed individuals
                'unexposed'     :   weight to unexposed individuals
        return_probability:
            -Whether to return the calculates weights or the denominator probabilities. Default is to return the weights.
             For diagnostics, the two plots to generate are meant for probabilities and not weights. 
             
             For the available diagnostic functions:
                standardized_diff   :   weight
                p_hist              :   probability
                p_boxplot           :   probability
                positivity          :   weight
        '''
        #Generating probabilities of treatment by covariates
        pd = propensity_score(df,treatment + ' ~ ' + model_denominator)
        twdf = pd.DataFrame()
        twdf['t'] = self.data[treatment].copy()
        twdf['pd'] = pd
        if return_probability == True:
            return twdf.pd
        
        #Generating Stabilized Weights if Requested
        if stabilized==True:
            #Calculating probabilities for numerator, default goes to Pr(A=a)
            if model_numerator == None:
                pn = propensity_score(df,treatment + ' ~ 1')
            else:
                pn = propensity_score(df,treatment + ' ~ ' + model_numerator)
            twdf['pn'] = pn
            
            #Stabilizing to population (compares all exposed to unexposed)
            if standardize == 'population':
                twdf['w'] = np.where(twdf['t']==1, (twdf['pn'] / twdf['pd']), ((1-twdf['pn']) / (1-twdf['pd'])))
                twdf.loc[(twdf['t']!=1)&(twdf['t']!=0),'w'] = np.nan
            #Stabilizing to exposed (compares all exposed if they were exposed versus unexposed)
            elif standardize == 'exposed':
                twdf['w'] = np.where(twdf['t']==1, 1, ((twdf['pd']/(1-twdf['pd'])) * ((1-twdf['pn'])/twdf['pn'])))
                twdf.loc[(twdf['t']!=1)&(twdf['t']!=0),'w'] = np.nan
            #Stabilizing to unexposed (compares all unexposed if they were exposed versus unexposed)
            elif standardize == 'unexposed':
                twdf['w'] = np.where(twdf['t']==1, (((1-twdf['pd'])/twdf['pd']) * (twdf['pn']/(1-twdf['pn']))), 1)
                twdf.loc[(twdf['t']!=1)&(twdf['t']!=0),'w'] = np.nan
            else:
                raise ValueError('Please specify one of the currently supported weighting schemes: population, exposed, unexposed')
        
        #Generating Unstabilized Weights if Requested
        else:
            #Stabilizing to population (compares all exposed to unexposed)
            if standardize == 'population':
                twdf['w'] = np.where(twdf['t']==1, 1 / twdf['pd'], 1 / (1-twdf['pd']))
                twdf.loc[(twdf['t']!=1)&(twdf['t']!=0),'w'] = np.nan
            #Stabilizing to exposed (compares all exposed if they were exposed versus unexposed)
            elif standardize == 'exposed':
                twdf['w'] = np.where(twdf['t']==1, 1, (twdf['pd']/(1-twdf['pd'])))
                twdf.loc[(twdf['t']!=1)&(twdf['t']!=0),'w'] = np.nan
            #Stabilizing to unexposed (compares all unexposed if they were exposed versus unexposed)
            elif standardize == 'unexposed':
                twdf['w'] = np.where(twdf['t']==1, ((1-twdf['pd'])/twdf['pd']), 1)
                twdf.loc[(twdf['t']!=1)&(twdf['t']!=0),'w'] = np.nan
            else:
                raise ValueError('Please specify one of the currently supported weighting schemes: population, exposed, unexposed')
        if return_probability == False:
            return twdf.w
    
    


def ipmw(df,missing,model,stabilized=True):
        '''Calculates the weight for inverse probability of missing weights using logistic regression. 
        Function automatically codes a missingness indicator (based on np.nan), so data can be directly
        input.
        
        Returns a pandas Series of calculated inverse probability of missingness weights for all observations
        
        df: 
            -pandas dataframe object containing all variables of interest
        missing:
            -Variable with missing data. numpy.nan values should indicate missing observations
        model:
            -statsmodels glm format for modeling data. Independent variables should be predictive of missingness
             of variable of interest. Example) 'var1 + var2 + var3'
        stabilized:
            -Whether to return the stabilized or unstabilized IPMW. Default is to return stabilized weights
        '''
        #Generating indicator for Missing data
        mdf = df.copy()
        mdf.loc[mdf[missing].isnull(),'observed_indicator'] = 0
        mdf.loc[mdf[missing].notnull(),'observed_indicator'] = 1
        
        #Generating probability of being observed based on model
        p = propensity_score(self.data,'obs ~ '+model)
        
        #Generating weights
        if stabilized==True:
            p_ = np.mean(mdf['observed_indicator'])
            w = p_ / p
        else:
            w = p**-1
        return w


def ipcw(df,idvar,model_denominator,model_numerator,stabilized=True):
    '''Calculate the inverse probability of censoring weights (IPCW). Note that this function will 
    only operate as expected when a valid dataframe is input. For a valid style of dataframe, see 
    below or the documentation for ipcw_data_converter(). IPCW is calculated via logistic regression
    and weights are cumulative products per unique ID. IPCW can be used to correct for missing at 
    random data by the generated model in weighted Kaplan-Meier curves
    
    df:
        -pandas DataFrame object containing all the variables of interest. This object must be sorted
         and have a variable called 'uncensored' indicating if an individual remained uncensored for that
         time period. It is highly recommended that ipcw_prep() is used prior to this function
    idvar:
        -Variable indicating a unique identifier for each individual followed over time
    model_denominator:
        -statsmodels glm format for modeling data. Only predictor variables for the denominator (variables
         determined to be related to censoring). All variables included in the numerator, should be included
         in the denominator. 
         Example) 'var1 + var2 + var3 + t_start + t_squared'
    model_numerator:
        -statsmodels glm format for modeling data. Only includes predictor variables for the numerator.
         In general, time is included in the numerator. Example) 't_start + t_squared'
    stabilized:
        -Whether to return stabilized or unstabilized weights. Input is True/False. Default is stabilized weights (True)

    Important notes/limitations:
    1) The dataframe MUST be sorted by ID and ascending time. If not, generated weights will be incorrect
    2) An indicator variable called 'uncensored' must be generated for the function. This can be accomplished
       by the ipcw_prep() function

    Input data format:
    cid     t_start     t_end   event   uncensored    ...
    101     0           1       0       1
    101     1           2       0       1
    101     2           2.1     1       1
    102     0           1       0       1
    102     1           2       0       1
    102     2           3       0       1
    102     3           3.5     0       0
    '''
    cf = df.copy()
    print('Numerator model:')
    cf['pn'] = propensity_score(cf,'uncensored ~ ' + model_numerator)
    print('Denominator model:')
    cf['pd'] = propensity_score(cf,'uncensored ~ ' + model_denominator)
    cf['cn'] = cf.groupby(idvar)['pn'].cumprod()
    cf['cd'] = cf.groupby(idvar)['pd'].cumprod()
    
    #Calculating weight
    w = cf['cn'] / cf['cd']
    return w    

def ipcw_prep(df,idvar,t_start,t_end,event):
    '''Function to prepare the data to an appropriate format for the function ipcw()'''
    #Need to pull this from other specific code I have previously written


def ipw_fitter(constant_weights=True):
    #Future function to fit IP models (either GEE or weighted KM/NA/AJ models)


class diagnostic:
    '''Class containing diagnostic tools for IPW. It is recommended that IPW
    diagnostics are done before fitting the IPW. Class accepts only an IPW 
    object as input. Therefore, an IPW model must be constructed before using 
    this class and its corresponding functions.
    
    Balance diagnostics
        p_hist()
            -Graphical display of weights by actual treatment
        p_boxplot()
            -Graphical display of weights by actual treatment
    Positivity diagnostics
        positivity()
            -Only valid for stabilized IPTW
    '''
    def __init__(self,ipw,weight,probability):
        self.data = ipw.data
        self.w = weight
        self.p = probability

    def standardized_diff(self,treatment,var,var_type='binary',decimal=3):
        '''Compute the standardized differences for the IP weights by treatment. 
        Note that this can be used to compare both mean(continuous variable) and
        proportions (binary variable) between baseline variables. To compare interactions
        and higher-order moments of continuous variables, calculate a corresponding 
        variable then simply put in this new variable as the variable of interest 
        regarding balance. Note that comparing the mean of squares of continuous variables
        is akin to comparing the variances between the treatment groups.
        
        treatment:
            -Column name for variable that is regarded as the treatment. Currently, 
             only binary (0,1) treatments are supported
        var:
            -Column name for variable of interest regarding balanced by weights. 
             Variable can be binary (0,1) or continuous. If continuous, var_type option 
             must be changed
        var_type:
            -The type of variable in the var option. Default is binary. For continuous variables
             var_type must be specified as 'continuous' 
        decimal:
            -Number of decimal places to display in result
        '''
        def _weighted_avg(df,v,w,t):
            '''This function is only used to calculate the weighted mean for 
            standardized_diff function for continuous variables'''
            l = []
            for i in [0,1]:
                n = sum(df.loc[df[t]==i][v] * df.loc[df[t]==i][w])
                d = sum(df.loc[df[t]==i][w])
                a = n / d
                l.append(a)
            return l[0],l[1]
        def _weighted_std(df,v,w,t,xbar):
            '''This function is only used to calculated the weighted mean for
            standardized_diff function for continuous variables'''
            l = []
            for i in [0,1]:
                n1 = sum(df.loc[df[t]==i][w])
                d1 = sum(df.loc[df[t]==i][w]) ** 2
                d2 = sum(df.loc[df[t]==i][w]**2)
                n2 = sum(df.loc[df[t]==i][w] * ((df.loc[df[t]==i][v] - xbar[i])**2))
                a = ((n1/(d1-d2))*n2)
                l.append(a)
            return l[0],l[1]
        if var_type == 'binary':
            wtre = np.sum(self.data.loc[(self.data[var]==1) & (self.data[treatment]==1)][self.w].dropna()) / np.sum(self.data.loc[self.data[treatment]==1][self.w].dropna())
            wnot = np.sum(self.data.loc[(self.data[var]==1) & (self.data[treatment]==0)][self.w].dropna()) / np.sum(self.data.loc[self.data[treatment]==0][self.w].dropna())
            wsmd = 100 * ((wtre - wnot) / math.sqrt(((wtre*(1-wtre)+wnot*(1-wnot)) / 2)))
        elif var_type == 'continuous':
            wmn,wmt = _weighted_avg(self.data,v=var,w=self.w,t=treatment)
            wsn,wst = _weighted_std(self.data,v=var,w=self.w,t=treatment,xbar=[wmn,wmt])
            wsmd = 100 * (wmt - wmn) / (math.sqrt((wst+wsn) / 2))
        else:
            raise ValueError('The only variable types currently supported are binary and continuous')
        print('----------------------------------------------------------------------')
        print('IPW Diagnostic for balance: Standardized Differences')
        if var_type == 'binary':
            print('\tBinary variable: '+var)
        if var_type == 'continuous':
            print('\tContinuous variable: '+var)
        print('----------------------------------------------------------------------')
        print('Weighted SMD: \t',round(wsmd,decimal))
        print('----------------------------------------------------------------------')
    
    def p_hist(self,treatment): #add KWARGS about color...
        '''Generates a histogram that can be used to check whether positivity may be violated 
        qualitatively. Note input probability variable, not the weight
        
        treatment:
            -Binary variable that indicates treatment. Must be coded as 0,1
        '''
        import matplotlib.pyplot as plt
        plt.hist(self.data.loc[self.data[treatment]==1][self.p].dropna(),label='Treat = 1',color='b',alpha=0.8)
        plt.hist(self.data.loc[self.data[treatment]==0][self.p].dropna(),label='Treat = 0',color='r',alpha=0.5)
        plt.xlabel('Probability')
        plt.ylabel('Number of observations')
        plt.legend()
        plt.show()
    
    def p_boxplot(self,treatment):
        '''Generates a stratified boxplot that can be used to visually check whether positivity
        may be violated, qualitatively. Note input probability, not the weight
        
        treatment:
            -Binary variable that indicates treatment. Must be coded as 0,1
        '''
        import matplotlib.pyplot as plt
        boxes = self.data.loc[self.data[treatment]==1][self.p].dropna(), self.data.loc[self.data[treatment]==0][self.p].dropna()
        labs = ['Treat = 1','Treat = 0']
        meanpointprops = dict(marker='D', markeredgecolor='black',markerfacecolor='black')
        plt.boxplot(boxes,labels=labs,meanprops=meanpointprops,showmeans=True)
        plt.ylabel('Probability')
        plt.show()
    
    def positivity(self,decimal=3):
        '''Use this to assess whether positivity is a valid assumption. Note that this
        should only be used for stabilized weights generated from IPTW. This diagnostic method
        is based on recommendations from Cole SR & Hernan MA (2008). For more information, see the
        following paper:
        
        Cole SR, Hernan MA. Constructing inverse probability weights for marginal structural models. 
        American Journal of Epidemiology 2008; 168(6):656–664.
        
        decimal:
            -number of decimal places to display. Default is three
        '''
        avg = np.mean(self.data[self.w].dropna())
        mx = np.max(self.data[self.w].dropna())
        mn = np.min(self.data[self.w].dropna())
        sd = np.std(self.data[self.w].dropna())
        print('----------------------------------------------------------------------')
        print('IPW Diagnostic for positivity')
        print('If the mean of the weights is far from either the min or max, this may\n indicate the model is mispecified or positivity is violated')
        print('Standard deviation can help in IPTW model selection')
        print('----------------------------------------------------------------------')
        print('Mean weight:\t\t',round(avg,decimal))
        print('Standard Deviation:\t\t',round(sd,decimal))
        print('Minimum weight:\t\t\t',round(mn,decimal))
        print('Maximum weight:\t\t\t',round(mx,decimal))
        print('----------------------------------------------------------------------')
