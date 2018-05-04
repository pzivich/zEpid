import numpy as np 
import pandas as pd 

def data_generator(n=10000,seed=101):
    np.random.seed(seed=seed) #Setting seed
    df = pd.DataFrame(index=range(n)) #generating empty data frame with n rows
    #Creating baseline confounders/variables
    df['Z'] = np.random.binomial(n=1,p=0.7,size=len(df)) #Simulating baseline confounder
    df['var1'] = np.random.normal(loc=500,scale=50,size=n)
    df['var2'] = np.random.normal(loc=25,scale=1,size=n)
    df['var3'] = np.random.normal(loc=25,scale=1,size=n)
    #Setting up a confounded exposure
    df['pa0'] = 1 / (1 + np.exp( -(np.log(1) - 20 - 5*df['Z'] + 0.001*df['var1'] + 0.0001*(df['var1']**2) + 
                     + 0.05*df['var2'] - 0.07*df['var3']))) #Simulating probability for exposure
    df['X'] = np.random.binomial(n=1,p=df['pa0'],size=len(df)) #Simulating binary exposure based on probability 
    #Determining Outcomes
    df['predy'] = (30 + 5*df['X'] + 3*df['Z'] + 2*df['X']*df['Z'] - 0.0001*df['var1'] - 0.00005*df['var1']**2 + 
                    0.5*df['var2'] - 0.4*df['var3'])#Simulating probability of outcome
    #Determining some survival times
    df['t2'] = np.random.weibull(1.55,size=len(df)) * df['predy'] #Setting survival times from Weibull dist
    df['t1'] = np.where(df['t2']>10,10,df['t2']) #Administrative censoring time is 5
    df['D'] = np.where(df['t2']>10,0,1) #Event indicator if event happened before time 5
    
    #Sampling a bit of our generated data. If you want above as a base code, just delete below until 'return' 
    sf = df.sample(n=384)
    #Generating some missing data for the exposure
    sf['pXm'] = 1 / (np.exp(1 - 0.5*sf['Z'] + 0.01*sf['var3']))
    sf['Xm'] = np.random.binomial(n=1,p=sf['pXm'],size=len(sf))
    sf['X'] = np.where(sf['Xm']==1,np.nan,sf['X'])
    #Generating some censoring before the end of the study period 
    sf['t'] = np.where(sf['D']==0,sf['t1'] -0.09*sf['var2'] - 0.15*sf['var3'] - 0.005*(sf['var3']**2) +
                        np.random.uniform(low=2,high=8,size=sf.shape[0]),sf['t1'])
    return sf[['D','X','Z','var1','var2','var3','t']]


data = data_generator()

