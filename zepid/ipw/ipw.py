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

def ipw(df,model,mresult=True):
    '''Generate propensity scores (probability) based on the model input.
    Uses logistic regression model to calculate
    
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



class iptw:
    '''Class to construct and fit an IPTW model
    '''
    def __init__(self,df,idvar):
        '''Initiliaze with dataframe containing vars
        '''
        self.data = df
        self.idvar = idvar
    
    def weight(self,treatment,model,stabilized=True,standardize='population'):
        '''Calculates the weight for inverse probability of treatment weights
        through logistic regression. Function can return either stabilized or
        unstabilized weights. Stabilization is based on the proportion that 
        were treated
    
        Returns a pandas Series of calculated probability and a pandas Series
        of the weights of observations
        
        treatment:
            -Variable name of treatment variable of interest. Must be coded as binary.
             1 should indicate treatment, while 0 indicates no treatment
        model:
            -statsmodels glm format for modeling data. Example) 'var1 + var2 + var3'
        stabilized:
            -Whether to return stabilized or unstabilized weights. Input is True/False
             Default is stabilized weights (True)
        standardize:
            -Who to standardize the estimate to. Options are the entire population, the,
             exposed, or the unexposed. See Sato & Matsuyama. Epidemiology (2003)
             Options are:
                'population'    :   entire population
                'exposed'       :   exposed individuals
                'unexposed'     :   unexposed individuals
        '''
        p = ipw(self.data,treatment+' ~ '+model)
        twdf = pd.DataFrame()
        twdf['t'] = self.data[treatment].copy()
        twdf['p'] = p
        if stabilized==True:
            prev_t = float(np.mean(twdf['t']))
            if standardize == 'population':
                twdf['w'] = np.where(twdf['t']==1, (prev_t / twdf['p']), ((1-prev_t) / (1-twdf['p'])))
                twdf.loc[(twdf['t']!=1)&(twdf['t']!=0),'w'] = np.nan
            elif standardize == 'exposed':
                twdf['w'] = np.where(twdf['t']==1, 1, ((twdf['p']/(1-twdf['p'])) * ((1-prev_t)/prev_t))) #something is wrong with formula
                twdf.loc[(twdf['t']!=1)&(twdf['t']!=0),'w'] = np.nan
            elif standardize == 'unexposed':
                twdf['w'] = np.where(twdf['t']==1, (((1-twdf['p'])/twdf['p']) * (prev_t/(1-prev_t))), 1)
                twdf.loc[(twdf['t']!=1)&(twdf['t']!=0),'w'] = np.nan
            else:
                raise ValueError('Please specify one of the following weighting schemes: population, exposed, unexposed')
        else:
            if standardize == 'population':
                twdf['w'] = np.where(twdf['t']==1, 1 / twdf['p'], 1 / (1-twdf['p']))
                twdf.loc[(twdf['t']!=1)&(twdf['t']!=0),'w'] = np.nan
            elif standardize == 'exposed':
                twdf['w'] = np.where(twdf['t']==1, 1, ((1-twdf['p'])/twdf['p']))
                twdf.loc[(twdf['t']!=1)&(twdf['t']!=0),'w'] = np.nan
            elif standardize == 'unexposed':
                twdf['w'] = np.where(twdf['t']==1, (twdf['p']/(1-twdf['p'])), 1)
                twdf.loc[(twdf['t']!=1)&(twdf['t']!=0),'w'] = np.nan
            else:
                raise ValueError('Please specify one of the following weighting schemes: population, exposed, unexposed')
        self.data['tprob'] = twdf.p
        self.data['tweight'] = twdf.w
    
    def merge_weights(self,otherweight):
        '''Combine weights from another IPW model (such as adding IPCW or IPMW).
        This is done simply by multipying the weights together. Note that this process
        must be done to generate the combined weights for the fit() method to work 
        as expected 
        
        other_weight:
            -Other weight variable column name in the dataset 
        '''
        self.data['tweight'] *= self.data[otherweight]
    
    def fit(self,model,link_dist=None,fitmodel='gee',time=None):
        '''Fits an IPTW based on a weight generated weights. Model is fit by using 
        Generalized Estimation Equations with an independent covariance structure. 
        GEE and the robust variance estimator is meant to correct the confidence 
        interval deviation due to the added weights of the model
        
        df:
            -pandas dataframe containing the variables of interest
        model:
            -Model to fit. Example) 'y ~ treat'
        link_dist:
            -Generalized model link and distribution. Default is a logistic
             binomial model. Will fit any supported statsmodels GEE model type 
             Some options include:
                Log-Binomial:  sm.families.family.Binomial(sm.families.links.log)
                Linear-Risk:   sm.families.family.Binomial(sm.families.links.identity)
             See statsmodels documentation for a complete list 
        fitmodel:
            -Which model to use to fit the data. Options include:
                keyword         model
                'gee'       :   fit using a GEE (valid CI compared to GLM)
                'km'        :   weighted Kaplan-Meier (requires time)
                'na'        :   weighted Nelson-Aalen (requires time)
        time:
            -variable containing time. Default is None. Must be specified for KM/NA
        '''
        if fitmodel=='gee':
            ind = sm.cov_struct.Independence()
            if link_dist == None:
                f = sm.families.family.Binomial(sm.families.links.logit)
            else:
                f = link_dist
            iptw_m = smf.gee(model,self.idvar,self.data,cov_struct=ind,family=f,weights=self.data['tweight']).fit()
            self.result = iptw_m
        elif fitmodel=='km':
            if time == None:
                raise ValueError('A time variable must be specified')
            print('Not currently implemented... Waiting on lifelines v0.14.0')
        elif fitmodel=='na':
            if time == None:
                raise ValueError('A time variable must be specified')
            print('Not currently implemented... Waiting on lifelines v0.14.0')
        else:
            raise ValueError('Please specify an implmented model')


class ipmw:
    '''IPMW can put raw data into...
    missing: variable for weights to be generated for 
    '''
    def __init__(self,df,idvar,missing):
        '''Initiliaze with dataframe containing vars
        '''
        self.data = df
        self.idvar = idvar
        self.data.loc[self.data[missing].isnull(),'obs'] = 0
        self.data.loc[self.data[missing].notnull(),'obs'] = 1
    
    def weight(self,model,stabilized=True):
        '''Calculates the weight for inverse probability of missing weights
        through logistic regression. 
        
        Returns a pandas Series of calculated probability and a pandas Series 
        of weights for all observations
        
        model:
            -statsmodels glm format for modeling data. Example) 'var1 + var2 + var3'
        '''
        p = ipw(self.data,'obs ~ '+model)
        if stabilized==True:
            p_ = np.mean(self.data.obs)
            w = p_ / p
        else:
            w = p**-1
        self.data['mprob'] = p 
        self.data['mweight'] = w
    
    def merge_weights(self,otherweight):
        '''Combine weights from another IPW model (such as adding IPTW or IPCW).
        This is done simply by multipying the weights together. Note that this process
        must be done to generate the combined weights for the fit() method to work 
        as expected 
        
        other_weight:
            -Other weight variable column name in the dataset 
        '''
        self.data['mweight'] *= self.data[otherweight]

    def fit(self,model,link_dist=None,fitmodel='gee',time=None):
        '''Fits an IPMW based on a weight generated weights. Model is fit by using 
        Generalized Estimation Equations with an independent covariance structure. 
        GEE and the robust variance estimator is meant to correct the confidence 
        interval deviation due to the added weights of the model

        Returns a fitted statsmodel GEEResultsWrapper object
            
        df:
            -pandas dataframe containing the variables of interest
        model:
            -Model to fit. Example) 'y ~ treat'
        link_dist:
            -Generalized model link and distribution. Default is a logistic
             binomial model. Will fit any supported statsmodels GEE model type 
             Some options include:
                Log-Binomial:  sm.families.family.Binomial(sm.families.links.log)
                Linear-Risk:   sm.families.family.Binomial(sm.families.links.identity)
             See statsmodels documentation for a complete list 
        fitmodel:
            -Which model to use to fit the data. Options include:
                keyword         model
                'gee'       :   fit using a GEE (valid CI compared to GLM)
                'km'        :   weighted Kaplan-Meier (requires time)
                'na'        :   weighted Nelson-Aalen (requires time)
        time:
            -variable containing time. Default is None. Must be specified for KM/NA
        '''
        if fitmodel=='gee':
            ind = sm.cov_struct.Independence()
            if link_dist == None:
                f = sm.families.family.Binomial(sm.families.links.logit)
            else:
                f = link_dist
            iptw_m = smf.gee(model,self.idvar,self.data,cov_struct=ind,family=f,weights=self.data['mweight']).fit()
            self.result = iptw_m
        elif fitmodel=='km':
            if time == None:
                raise ValueError('A time variable must be specified')
            print('Not currently implemented... Waiting on lifelines v0.14.0')
        elif fitmodel=='na':
            if time == None:
                raise ValueError('A time variable must be specified')
            print('Not currently implemented... Waiting on lifelines v0.14.0')
        else:
            raise ValueError('Please specify an implmented model')



class ipcw():
    '''Class to generate and fit a IPCW model
    '''
    def __init__(self,df,idvar,time):
        '''Initiliaze with dataframe containing vars. Needs df, idvar, time
        '''
        self.data = df
        self.idvar = idvar
        self.time = time

    def longdata_converter(self,outcome):
        '''Converts a pandas dataframe from a condensed format to a long format for calculation of IPCW. 
        The conversion works as follows; each participant's observations are duplicated for the total number
        of time points observed (by a unit of one), the event time or drop out times are generated for each
        participant for each time period, and this dataframe is returned.Note that is dataframe. This function 
        is used to prepare a dataframe for inverse probability of censoring weights. To generate IPC weights, 
        the function ipcw() can be used with the generated dataframe. Please see the end of this documentation
        for an applied example.
        
        Important notes/limitations:
        1) This does NOT currently support left truncated dataframes
        
        Returns dataframe with columns for the ID (inherited from input dataframe), start of the observation 
        period (t_start), end of the observation period (t_end), event indicator for time period (event), and 
        whether uncensored  for time period (uncensor) 
        
        outcome:
            -Indicator of whether participant had outcome by end of follow-up (True = 1, False = 0)

        This is an example of what an input dataset will look like and the subsequent output:
            Input data:
        
        cid     time    outcome     ...
        101     2.1     1  
        102     3.5     0 
        ...
        
            Output data:
        cid     t_start     t_end   event   uncensor    ...
        101     0           1       0       1
        101     1           2       0       1
        101     2           2.1     1       1
        102     0           1       0       1
        102     1           2       0       1
        102     2           3       0       1
        102     3           3.5     0       0
        
        '''
        cf = self.data.copy()
        cf['t_int'] = cf[self.time].astype(int)
        cfl = pd.DataFrame(np.repeat(cf.values,cf['t_int']+1,axis=0),columns=cf.columns)
        cfl['lastobs'] = (cfl[self.idvar] != cfl[self.idvar].shift(-1)).astype(int)
        cfl['tpoint'] = cfl.groupby(self.idvar)['t_int'].cumcount()
        cfl['tdiff'] =  cfl[self.time] - cfl['tpoint']
        cfl['event'] = 0
        cfl['uncensor'] = 1
        cfl.loc[((cfl['lastobs']==1)&(cfl[outcome]==1)),'event'] = 1
        cfl.loc[((cfl['lastobs']==1)&(cfl[outcome]==0)),'uncensor'] = 0
        cfl['t_start'] = cfl['tpoint']
        cfl['t_end'] = np.where(cfl['tdiff']<1,cfl['t'],cfl['t_start']+1)
        self.data = cfl.drop(columns=['lastobs','tdiff','tpoint','t_int'])
    
    def weight(self,n_model,d_model,print_models=True,stabilized=True):
        '''Calculate the inverse probability of censoring weights (IPCW). Note that this function will 
        only operate as expected when a valid dataframe is input. For a valid style of dataframe, see 
        below or the documentation for ipcw_data_converter(). IPCW is calculated via logistic regression
        and weights are cumulative products per unique ID. IPCW can be used to correct for missing at 
        random data by the generated model in weighted Kaplan-Meier curves
        
        Important notes/limitations:
        1) The dataframe MUST be sorted by ID and ascending time. If not, generated weights will be incorrect
        
        Must be sorted by ID and time points to generate weights properly
        Add in unstablilized option, once I know it...
        '''
        cf = self.data.copy()
        linkdist=sm.families.family.Binomial(sm.families.links.logit)
        logn = smf.glm('uncensor ~ '+n_model,cf,family=linkdist).fit()
        logd = smf.glm('uncensor ~ '+d_model,cf,family=linkdist).fit()
        if print_models==True:
            print('Numerator model:')
            print(logn.summary(),'\n\n')
            print('Denominator model:')
            print(logd.summary())
        cf['num_p'] = logn.predict()
        cf['den_p'] = logd.predict()
        cf['num'] = cf.groupby(self.idvar)['num_p'].cumprod()
        cf['den'] = cf.groupby(self.idvar)['den_p'].cumprod()
        w = cf['num'] / cf['den']
        self.data['cweight'] = w
    
    def merge_weights(self,otherweight):
        '''Combine weights from another IPW model (such as adding IPTW or IPMW).
        This is done simply by multipying the weights together. Note that this process
        must be done to generate the combined weights for the fit() method to work 
        as expected 
        
        other_weight:
            -Other weight variable column name in the dataset 
        '''
        self.data['cweight'] *= self.data[otherweight]

    def fit(self,reps=1000):
        '''Future implemented function. Waiting till lifelines package v0.14 comes out with weighted Kaplan Meier.
        Confidence intervals are generated via bootstrapping
        '''
        print('Not currently implemented... Waiting on lifelines v0.14.0')


class diagnostic:
    '''Class containing diagnostic tools for IPW. It is recommended that IPW
    diagnostics are done before fitting the IPW. Class accepts only an IPW 
    object as input. Therefore, an IPW model must be constructed before using 
    this class and its corresponding functions.
    
    Balance diagnostics
        weight_hist()
            -Graphical display of weights by actual treatment
        weight_boxplot()
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
        def weighted_avg(df,v,w,t):
            '''This function is only used to calculate the weighted mean for 
            standardized_diff function for continuous variables'''
            l = []
            for i in [0,1]:
                n = sum(df.loc[df[t]==i][v] * df.loc[df[t]==i][w])
                d = sum(df.loc[df[t]==i][w])
                a = n / d
                l.append(a)
            return l[0],l[1]
        def weighted_std(df,v,w,t,xbar):
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
            wmn,wmt = weighted_avg(self.data,v=var,w=self.w,t=treatment)
            wsn,wst = weighted_std(self.data,v=var,w=self.w,t=treatment,xbar=[wmn,wmt])
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
