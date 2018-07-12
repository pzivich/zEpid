'''Calculators for various inverse probability weights. Current inclusions are inverse probability of 
treatment weights, inverse probability of missingness weights, inverse probability of censoring weights.
Predicted probabilites are generated throug logistic regression. Plans for the future include allowing the
user to specify the model to generate predicted probabilities. Diagnostics are also available for inverse 
probability of treatment weights.

Contents:
-propensity_score(): generate probabilities/propensity scores via logit model
-iptw(): calculate inverse probability of treament weights
-ipmw(): calculate inverse probability of missing weights
-ipcw_prep(): transform data into long format compatible with ipcw()
-ipcw(): calculate inverse probability of censoring weights
-ipt_weight_diagnostic(): generate diagnostics for IPTW
    |-positivity(): diagnostic values for positivity issues
    |-standardized_diff(): calculates the standardized differences of IP weights
-ipt_probability_diagnostic(): generate diagnostics for treatment propensity scores
    |-p_boxplot():generate boxplot of probabilities by exposure
    |-p_hist(): generates histogram of probabilities by exposure
'''

import warnings
import math 
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import family
from statsmodels.genmod.families import links
import matplotlib.pyplot as plt

def propensity_score(df, model, mresult=True):
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
    
    Example)
    >>>zepid.ipw.propensity_score(df=data,model='X ~ Z + var1 + var2')
    '''
    f = sm.families.family.Binomial(sm.families.links.logit) 
    log = smf.glm(model,df,family=f).fit()
    if mresult == True:
        print('\n----------------------------------------------------------------')
        print('MODEL: '+model)
        print('-----------------------------------------------------------------')
        print(log.summary())
    p = log.predict(df)
    return p


def iptw(df,treatment, model_denominator, model_numerator='1', stabilized=True, standardize='population',
         return_probability=False,print_model_results=True):
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
        -statsmodels glm format for modeling data. Only includes predictor variables for the numerator. Default ('1')
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
    print_model_results:
        -Whether to print the model results from the regression models. Default is True
             
    Example)
    >>>zepid.ipw.iptw(df=data,treatment='X',model_denominator='Z + var1 + var2')
    '''
    #Generating probabilities of treatment by covariates
    pde = propensity_score(df,treatment + ' ~ ' + model_denominator,mresult=print_model_results)
    twdf = pd.DataFrame()
    twdf['t'] = df[treatment]
    twdf['pde'] = pde

    #Generating Stabilized Weights if Requested
    if stabilized==True:
        #Calculating probabilities for numerator, default goes to Pr(A=a)
        pn = propensity_score(df,treatment + ' ~ ' + model_numerator,mresult=print_model_results)
        twdf['pnu'] = pn
        
        #Stabilizing to population (compares all exposed to unexposed)
        if standardize == 'population':
            twdf['w'] = np.where(twdf['t']==1, (twdf['pnu'] / twdf['pde']), ((1-twdf['pnu']) / (1-twdf['pde'])))
            twdf.loc[(twdf['t']!=1)&(twdf['t']!=0),'w'] = np.nan
        #Stabilizing to exposed (compares all exposed if they were exposed versus unexposed)
        elif standardize == 'exposed':
            twdf['w'] = np.where(twdf['t']==1, 1, ((twdf['pde']/(1-twdf['pde'])) * ((1-twdf['pnu'])/twdf['pnu'])))
            twdf.loc[(twdf['t']!=1)&(twdf['t']!=0),'w'] = np.nan
        #Stabilizing to unexposed (compares all unexposed if they were exposed versus unexposed)
        elif standardize == 'unexposed':
            twdf['w'] = np.where(twdf['t']==1, (((1-twdf['pde'])/twdf['pde']) * (twdf['pnu']/(1-twdf['pnu']))), 1)
            twdf.loc[(twdf['t']!=1)&(twdf['t']!=0),'w'] = np.nan
        else:
            raise ValueError('Please specify one of the currently supported weighting schemes: population, exposed, unexposed')
        
    #Generating Unstabilized Weights if Requested
    else:
        #Stabilizing to population (compares all exposed to unexposed)
        if standardize == 'population':
            twdf['w'] = np.where(twdf['t']==1, 1 / twdf['pde'], 1 / (1-twdf['pde']))
            twdf.loc[(twdf['t']!=1)&(twdf['t']!=0),'w'] = np.nan
        #Stabilizing to exposed (compares all exposed if they were exposed versus unexposed)
        elif standardize == 'exposed':
            twdf['w'] = np.where(twdf['t']==1, 1,  ((1-twdf['pde'])/twdf['pde']))
            twdf.loc[(twdf['t']!=1)&(twdf['t']!=0),'w'] = np.nan
        #Stabilizing to unexposed (compares all unexposed if they were exposed versus unexposed)
        elif standardize == 'unexposed':
            twdf['w'] = np.where(twdf['t']==1,(twdf['pde']/(1-twdf['pde'])), 1)
            twdf.loc[(twdf['t']!=1)&(twdf['t']!=0),'w'] = np.nan
        else:
            raise ValueError('Please specify one of the currently supported weighting schemes: population, exposed, unexposed')
    if (return_probability == True) and (model_numerator!='1'):
        return twdf[['pde','pnu']]
    if (return_probability == True):
        return twdf['pde']
    return twdf.w
    
    


def ipmw(df, missing, model, stabilized=True,print_model_results=True):
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
    print_model_results:
        -Whether to print the model results. Default is True
        
    Example)
    >>>zepid.ipw.ipmw(df=data,missing='X',model='Z + var3')
    '''
    #Generating indicator for Missing data
    mdf = df.copy()
    mdf.loc[mdf[missing].isnull(),'observed_indicator'] = 0
    mdf.loc[mdf[missing].notnull(),'observed_indicator'] = 1
    
    #Generating probability of being observed based on model
    p = propensity_score(mdf,'observed_indicator ~ '+model,mresult=print_model_results)
    
    #Generating weights
    if stabilized==True:
        p_ = np.mean(mdf['observed_indicator'])
        w = p_ / p
    else:
        w = p**-1
    return w


def ipcw_prep(df, idvar, time, event, enter=None):
    '''Function to prepare the data to an appropriate format for the function ipcw(). It breaks the dataset into 
    single observations for event one unit increase in time. It prepares the dataset to be eligible for IPCW calculation. 
    If your datasets is already in a long format, there is not need for this conversion
    
    Returns:
    cid     t_start     t_end   event   uncensored    ...
    101     0           1       0       1
    101     1           2       0       1
    101     2           2.1     1       1
    102     0           1       0       1
    102     1           2       0       1
    102     2           3       0       1
    102     3           3.5     0       0
    
    df:
        -pandas dataframe to convert into a long format
    idvar:
        -ID variable to retain for observations
    time: 
        -Last follow-up visit for participant
    event:
        -indicator of whether participant had the event (1 is yes, 0 is no)
    enter:
        -entry time for the participant. Default is None, which means all participants are assumed
         to enter at time zero. Input should be column name of entrance time
    
    Example)
    >>>data['pid'] = data.index
    >>>ipc_data = zepid.ipw.ipcw_prep(df=data,idvar='pid',time='t',event='y')
    '''
    cf = df.copy()
    
    #Copying observations over times
    cf['t_int_zepid'] = cf[time].astype(int)
    lf = pd.DataFrame(np.repeat(cf.values,cf['t_int_zepid']+1,axis=0),columns=cf.columns)
    lf['tpoint_zepid'] = lf.groupby(idvar)['t_int_zepid'].cumcount()
    lf['tdiff_zepid'] =  lf[time] - lf['tpoint_zepid']
    lf = lf.loc[lf['tdiff_zepid']!=0].copy() #gets rid of censored at absolute time point (ex. censored at time 10)
    lf.loc[lf['tdiff_zepid']>1,'delta_indicator_zepid'] = 0
    lf.loc[((lf['tdiff_zepid']<=1)&(lf[event]==0)),'delta_indicator_zepid'] = 0
    lf.loc[((lf['tdiff_zepid']<=1)&(lf[event]==1)),'delta_indicator_zepid'] = 1
    lf['t_enter_zepid'] = lf['tpoint_zepid']
    lf['t_out_zepid'] = np.where(lf['tdiff_zepid']<1,lf[time],lf['t_enter_zepid']+1)
    lf['uncensored_zepid'] = np.where((lf[idvar] != lf[idvar].shift(-1)) & (lf['delta_indicator_zepid']==0),0,1)
    
    #Removing blocks of observations that would have occurred before entrance into the sample
    if enter != None:
        lf = lf.loc[lf['t_enter_zepid']>=lf[enter]].copy()

    #Cleaning up the edited dataframe to return to user
    if enter == None:
        lf.drop(columns=['tdiff_zepid','tpoint_zepid','t_int_zepid',time,event],inplace=True)
    else:
        lf.drop(columns=['tdiff_zepid','tpoint_zepid','t_int_zepid',time,event,enter],inplace=True)
    lf.rename(columns={"delta_indicator_zepid":event,'uncensored_zepid':'uncensored','t_enter_zepid':'t_enter',
              't_out_zepid':'t_out'},inplace=True)
    return lf


def ipcw(df, uncensored, idvar, model_denominator, model_numerator, stabilized=True,print_model_results=True):
    '''Calculate the inverse probability of censoring weights (IPCW). Note that this function will 
    only operate as expected when a valid dataframe is input. For a valid style of dataframe, see 
    below or the documentation for ipcw_data_converter(). IPCW is calculated via logistic regression
    and weights are cumulative products per unique ID. IPCW can be used to correct for missing at 
    random data by the generated model in weighted Kaplan-Meier curves
    
    df:
        -pandas DataFrame object containing all the variables of interest. This object must be sorted
         and have a variable called 'uncensored' indicating if an individual remained uncensored for that
         time period. It is highly recommended that ipcw_prep() is used prior to this function
    uncensored:
        -column label for indicator that variable is uncensored. Must be 0,1 with 1 indicating an individual
         was NOT censored over that time period
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
    print_model_results:
        -Whether to print the model results. Default is True

    Important notes/limitations:
    1) The dataframe MUST be sorted by ID and ascending time. If not, generated weights will be incorrect

    Input data format:
    cid     t_start     t_end   event   uncensored    ...
    101     0           1       0       1
    101     1           2       0       1
    101     2           2.1     1       1
    102     0           1       0       1
    102     1           2       0       1
    102     2           3       0       1
    102     3           3.5     0       0
    
    Example)
    >>>ipc_data = zepid.ipw.ipcw_prep(df=data,idvar='pid',time='t',event='D')
    >>>ipc_data['t'] = ipc_data['t_enter']
    >>>ipc_data['t2'] = ipc_data['t']**2
    >>>ipc_data['t3'] = ipc_data['t']**3
    >>>zepid.ipw.ipcw(df=ipc_data,uncensored='uncensored',idvar='pid',model_denominator='var1 + var2 + t',model_numerator='t')
    '''
    cf = df.copy()
    print('Numerator model:')
    cf['pn'] = propensity_score(cf,uncensored + ' ~ ' + model_numerator,mresult=print_model_results)
    print('Denominator model:')
    cf['pd'] = propensity_score(cf,uncensored + ' ~ ' + model_denominator,mresult=print_model_results)
    cf['cn'] = cf.groupby(idvar)['pn'].cumprod()
    cf['cd'] = cf.groupby(idvar)['pd'].cumprod()
    
    #Calculating weight
    w = cf['cn'] / cf['cd']
    return w    


class iptw_weight_diagnostic:
    '''Class containing diagnostic tools for IPTW. It is recommended that IPTW diagnostics are done before 
    fitting the IPTW. 

    Balance diagnostics
        positivity()
            -Only valid for stabilized IPTW
        standardized_diff()
    '''
    def __init__(self, df, weight):
        self.data = df
        self.w = weight

    def positivity(self, decimal=3):
        '''Use this to assess whether positivity is a valid assumption. Note that this should only be used for 
        stabilized weights generated from IPTW. This diagnostic method is based on recommendations from 
        Cole SR & Hernan MA (2008). For more information, see the following paper:
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
        print('Mean weight:\t\t\t',round(avg,decimal))
        print('Standard Deviation:\t\t',round(sd,decimal))
        print('Minimum weight:\t\t\t',round(mn,decimal))
        print('Maximum weight:\t\t\t',round(mx,decimal))
        print('----------------------------------------------------------------------')

    def standardized_diff(self, treatment, var, var_type='binary', decimal=3):
        '''Compute the standardized differences for the IP weights by treatment. Note that this can be used to compare 
        both mean(continuous variable) and proportions (binary variable) between baseline variables. To compare interactions
        and higher-order moments of continuous variables, calculate a corresponding variable then simply put in this new 
        variable as the variable of interest regarding balance. Note that comparing the mean of squares of continuous variables
        is akin to comparing the variances between the treatment groups.
        
        treatment:
            -Column name for variable that is regarded as the treatment
        var:
            -Column name for variable of interest regarding balanced by weights. Variable can be binary (0,1) or continuous. 
             If continuous, var_type option must be changed
        var_type:
            -The type of variable in the var option. Default is binary. For continuous variables var_type must be specified as 
             'continuous' 
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


class iptw_probability_diagnostic:
    '''Class containing diagnostic tools for IPTW. It is recommended that IPTW diagnostics are done before 
    fitting the IPTW.
    
    Balance diagnostics
        p_hist()
            -Graphical display of weights by actual treatment
        p_boxplot()
            -Graphical display of weights by actual treatment
    '''
    def __init__(self, df, probability):
        if ((np.max(df[probability]>1)) | (np.min(df[probability]<0))):
            raise ValueError('Input column must be probability')
        self.data = df
        self.p = probability
    
    def p_hist(self, treatment,bins=None,color_e='b',color_u='r'):
        '''Generates a normalized histogram that can be used to check whether positivity may be violated qualitatively.
        Note input probability variable, not the weight!
        
        treatment:
            -Binary variable that indicates treatment. Must be coded as 0,1
        bins:
            -Number of bins to generate the histograms with. Default is matplotlib's default number of bins
        color_e:
            -color of the line/area for the treated group. Default is Blue
        color_u:
            -color of the line/area for the treated group. Default is Red
        '''
        ax = plt.gca()
        if bins == None:
            ax.hist(self.data.loc[self.data[treatment]==1][self.p].dropna(),label='Treat = 1',color=color_e,alpha=0.8,normed=True)
            ax.hist(self.data.loc[self.data[treatment]==0][self.p].dropna(),label='Treat = 0',color=color_u,alpha=0.5,normed=True)
        else:
            ax.hist(self.data.loc[self.data[treatment]==1][self.p].dropna(),label='Treat = 1',bins=bins,color=color_e,alpha=0.8,normed=True)
            ax.hist(self.data.loc[self.data[treatment]==0][self.p].dropna(),label='Treat = 0',bins=bins,color=color_u,alpha=0.5,normed=True)            
        ax.set_xlabel('Probability')
        ax.set_ylabel('Number of observations')
        ax.legend()
        return ax

    def p_kde(self,treatment,bw_method='scott',fill=True,color_e='b',color_u='r'):
        '''Generates a density plot that can be used to check whether positivity may be violated qualitatively. Note
        input probability variable, not the weight! The kernel density used is SciPy's Gaussian kernel. Either Scott's
        Rule or Silverman's Rule can be implemented.
        
        This is an alternative to the p_hist() function. I would recommend this over p_hist() generally since it makes
        a nicer looking plot. 

        treatment:
            -Binary variable that indicates treatment. Must be coded as 0,1
        bw_method:
            -method used to estimate the bandwidth. Following SciPy, either 'scott' or 'silverman' are valid options
        fill:
            -whether to color the area under the density curves. Default is true
        color_e:
            -color of the line/area for the treated group. Default is Blue
        color_u:
            -color of the line/area for the treated group. Default is Red
        '''
        #Getting Gaussian Kernel Density
        x = np.linspace(0,1,10000)
        density_t = stats.kde.gaussian_kde(self.data.loc[self.data[treatment]==1][self.p].dropna(),bw_method=bw_method)
        density_u = stats.kde.gaussian_kde(self.data.loc[self.data[treatment]==0][self.p].dropna(),bw_method=bw_method)

        #Creating density plot
        ax = plt.gca()
        if fill == True:
            ax.fill_between(x,density_t(x),color=color_e,alpha=0.2,label=None)
            ax.fill_between(x,density_u(x),color=color_u,alpha=0.2,label=None)
        ax.plot(x, density_t(x),color=color_e,label='Treat = 1')
        ax.plot(x, density_u(x),color=color_u,label='Treat = 0')
        ax.set_xlabel('Probability')
        ax.set_ylabel('Density')
        ax.legend()
        return ax
    
    def p_boxplot(self, treatment):
        '''Generates a stratified boxplot that can be used to visually check whether positivity may be violated, qualitatively. 
        Note input probability, not the weight!
        
        treatment:
            -Binary variable that indicates treatment. Must be coded as 0,1
        '''
        boxes = (self.data.loc[self.data[treatment]==1][self.p].dropna(), self.data.loc[self.data[treatment]==0][self.p].dropna())
        labs = ['Treat = 1','Treat = 0']
        meanpointprops = dict(marker='D', markeredgecolor='black',markerfacecolor='black')
        ax = plt.gca()
        ax.boxplot(boxes,labels=labs,meanprops=meanpointprops,showmeans=True)
        ax.set_ylabel('Probability')
        return ax
    
