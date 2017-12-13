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
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.genmod.families import family
    from statsmodels.genmod.families import links
    f = sm.families.family.Binomial(sm.families.links.logit) 
    log = smf.glm(model,df,family=f).fit()
    if mresult == True:
        print(log.summary())
    p = log.predict(df)
    return pd.Series(p)
    

def ipmw(df,missing,model):
    '''Calculates the weight for inverse probability of missing weights
    through logistic regression. 
    
    Returns a pandas Series of calculated probability and a pandas Series 
    of weights for all observations
    
    df:
        -pandas dataframe containing the variables of interest
    missing:
        -Variable name that contains missing data. 
    model:
        -statsmodels glm format for modeling data. Example) 'var1 + var2 + var3'
    '''
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.genmod.families import family
    from statsmodels.genmod.families import links
    dm = df.copy()
    dm.loc[dm[missing].isnull(),'miss'] = 1
    dm.loc[dm[missing].notnull(),'miss'] = 0
    p = ipw(dm,'miss ~ '+model)
    w = p**-1
    return pd.Series(p),pd.Series(w)


def iptw(df,treatment,model,stabilized=True):
    '''Calculates the weight for inverse probability of treatment weights
    through logistic regression. Function can return either stabilized or
    unstabilized weights. Stabilization is based on the proportion that 
    were treated
    
    Returns a pandas Series of calculated probability and a pandas Series
    of the weights of observations
    
    df:
        -pandas dataframe containing the variables of interest
    treatment:
        -Variable name of treatment variable of interest. Must be coded as binary.
         1 should indicate treatment, while 0 indicates no treatment
    model:
        -statsmodels glm format for modeling data. Example) 'var1 + var2 + var3'
    stabilized:
        -Whether to return stabilized or unstabilized weights. Input is True/False
         Default is stabilized weights (True)
    '''
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.genmod.families import family
    from statsmodels.genmod.families import links
    p = ipw(df,treatment+' ~ '+model)
    twdf = pd.DataFrame()
    twdf['t'] = df[treatment]
    twdf['p'] = p
    if stabilized==True:
        prev_t = np.mean(twdf['t'])
        twdf.loc[twdf['t']==1,'w'] = prev_t / twdf['p']
        twdf.loc[twdf['t']==0,'w'] = (1-prev_t) / (1-twdf['p'])
    else:
        twdf.loc[twdf['t']==1,'w'] = 1 / twdf['p']
        twdf.loc[twdf['t']==0,'w'] = 1 / (1-twdf['p'])
    return twdf.p,twdf.w


def ipw_fit(df,model,match,weight,link_dist=None):
    '''Fit an inverse probability weighted model based on a weight
    variable. Model is fit by using Generalized Estimation Equations
    with an independent covariance structure. GEE and the robust variance
    estimator is meant to correct the confidence interval deviation due
    to the added weights of the model
    
    Returns a fitted statsmodel GEEResultsWrapper object
    
    df:
        -pandas dataframe containing the variables of interest
    model:
        -Model to fit. Example) 'y ~ treat'
    match:
        -Variable name to match the GEE on. Should be a unique identifier for each observation.
         Commonly an ID variable
    weight:
        -Calculated weight for the observations
    link_dist:
        -Generalized model link and distribution. Default is a logistic
         binomial model. Will fit any supported statsmodels GEE model type 
         Some options include:
            Log-Binomial:  sm.families.family.Binomial(sm.families.links.log)
            Linear-Risk:   sm.families.family.Binomial(sm.families.links.identity)
         See statsmodels documentation for a complete list 
    '''
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.genmod.families import family
    from statsmodels.genmod.families import links
    ind = sm.cov_struct.Independence()
    if link_dist == None:
        f = sm.families.family.Binomial(sm.families.links.logit)
    else:
        f = link_dist
    IPW = smf.gee(model,match,df,cov_struct=ind,family=f,weights=df[weight]).fit()
    return IPW 


class diagnostic:
    '''Contains diagnostic tools to check IPW weighted models. Diagnostic options
    include:
    
    Balance diagnostics
        weight_hist()
            -Graphical display of weights by actual treatment
        weight_boxplot()
            -Graphical display of weights by actual treatment

    Positivity diagnostics
        positivity()
            -Only valid for stabilized IPTW
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
        print(l)
        return l[0],l[1]
    
    def standardized_diff(df,treatment,var,weight,var_type='binary',decimal=3):
        '''Compute the standardized differences for the weighted and unweighted 
        probability of treatment.
        '''
        import math
        if var_type == 'binary':
            wtre = np.sum(df.loc[(df[var]==1) & (df[treatment]==1)][weight].dropna()) / np.sum(df.loc[df[treatment]==1][weight].dropna())
            wnot = np.sum(df.loc[(df[var]==1) & (df[treatment]==0)][weight].dropna()) / np.sum(df.loc[df[treatment]==0][weight].dropna())
            wsmd = 100 * ((wtre - wnot) / math.sqrt(((wtre*(1-wtre)+wnot*(1-wnot)) / 2)))
        elif var_type == 'continuous':
            wmn,wmt = diagnostic.weighted_avg(df,v=var,w=weight,t=treatment)
            wsn,wst = diagnostic.weighted_std(df,v=var,w=weight,t=treatment,xbar=[wmn,wmt])
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
    
    def w_hist(df,treatment,probability):
        '''Generates a histogram that can be used to check whether positivity may be violated 
        qualitatively. Note input probability variable, not the weight
        
        df:
            -dataframe that contains the weights
        treatment:
            -Binary variable that indicates treatment. Must be coded as 0,1
        probability:
            -dataframe column name of the probability
        '''
        import matplotlib.pyplot as plt
        plt.hist(df.loc[df[treatment]==1][probability].dropna(),label='T=1',color='b',alpha=0.8)
        plt.hist(df.loc[df[treatment]==0][probability].dropna(),label='T=0',color='r',alpha=0.5)
        plt.xlabel('Probability')
        plt.ylabel('Number of observations')
        plt.legend()
        plt.show()
    
    def w_boxplot(df,treat,probability):
        '''Generates a stratified boxplot that can be used to visually check whether positivity
        may be violated, qualitatively. Note input probability, not the weight
        
        df:
            -dataframe that contains the weights
        treatment:
            -Binary variable that indicates treatment. Must be coded as 0,1
        probability:
            -dataframe column name of the probability
        '''
        import matplotlib.pyplot as plt
        boxes = df.loc[df[treat]==1][probability].dropna(), df.loc[df[treat]==0][probability].dropna()
        labs = ['T=1','T=0']
        meanpointprops = dict(marker='D', markeredgecolor='black',markerfacecolor='black')
        plt.boxplot(boxes,labels=labs,meanprops=meanpointprops,showmeans=True)
        plt.ylabel('Probability')
        plt.show()
    
    def positivity(df,weight,decimal=3):
        '''Use this to assess whether positivity is a valid assumption. Note that this
        should only be used for stabilized weights generated from IPTW. This diagnostic method
        is based on recommendations from Cole SR & Hernan MA (2008). For more information, see the
        following paper:
        
        Cole SR, Hernan MA. Constructing inverse probability weights for marginal structural models. 
        American Journal of Epidemiology 2008; 168(6):656–664.
        
        df:
            -dataframe that contains the weights
        weight:
            -dataframe column name of the weights
        decimal:
            -number of decimal places to display. Default is three
        '''
        avg = np.mean(df[weight].dropna())
        mx = np.max(df[weight].dropna())
        mn = np.min(df[weight].dropna())
        sd = np.std(df[weight].dropna())
        print('----------------------------------------------------------------------')
        print('IPW Diagnostic for positivity')
        print('NOTE: This method is only valid for stabilized IPTW weights\n')
        print('If the mean of the weights is far from either the min or max, this may\n indicate the model is mispecified or positivity is violated')
        print('Standard deviation can help in IPTW model selection')
        print('----------------------------------------------------------------------')
        print('Mean stabilized weight:\t\t',round(avg,decimal))
        print('Standard Deviation:\t\t',round(sd,decimal))
        print('Minimum weight:\t\t\t',round(mn,decimal))
        print('Maximum weight:\t\t\t',round(mx,decimal))
        print('----------------------------------------------------------------------')
