import warnings
import math 
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

class MonteCarloRR:
    '''Monte Carlo simulation to assess the impact of an unmeasured binary confounder on the results
    of a study. Observed RR comes from the data analysis, while the RR between the unmeasured confounder
    and the outcome should be obtained from prior literature or constitute an reasonable guess.
    Probability of exposure between the groups should also be reasonable numbers. 
    
    observed_RR:
        -observed RR from the data, not accounting for some binary unmeasured confounder
    sample:
        -number of MC simulations to run. It is important that the specified size of later distributions
         matches this number of samples
    
    Example)
    >>>from zepid.sensitivity_analysis import MonteCarloRR
    >>>mcrr = MonteCarloRR(observed_RR=)
    >>>
    >>>
    >>>
    >>>mcrr.fit(sample=10000)
    '''
    def __init__(self,observed_RR,sample):
        self.RRo = observed_RR
        self._rrdist = False
        self._p1dist = False
        self._p0dist = False
        self.sample = sample

    def confounder_RR_distribution(self,dist,seed=None):
        '''Distribution of the risk ratio between the unmeasured confounder and the outcome. This
        value should come from prior literature or a reasonable guess. Any numpy random distribution
        can be based to this function. Alternatively, the trapezoid distribution within this library
        can also be used

        dist:
            -distribution from which the confounder-outcome Risk Ratio is pulled from. Input should be
             something like numpy.random.triangular(left=0.9,mode=1.2,right=1.6)
        '''
        if seed != None:
            np.random.seed(seed)
        self.RRc_dist = dist
        self._rrdist = True

    def prop_confounder_exposed(self,dist,seed=None):
        '''Distribution of the proportion of the unmeasured confounder in the exposed group. This
        value should come from prior literature or a reasonable guess. Any numpy random distribution
        can be based to this function. Alternatively, the trapezoid distribution within this library
        can also be used

        dist:
            -distribution from which the confounder-outcome Risk Ratio is pulled from. Input should be
             something like numpy.random.triangular(left=0.1,mode=0.2,right=0.3)
        '''
        if seed != None:
            np.random.seed(seed)
        self.pc1 = dist
        self._p1dist = True

    def prop_confounder_unexposed(self,dist,seed=None):
        '''Distribution of the proportion of the unmeasured confounder in the unexposed group. This
        value should come from prior literature or a reasonable guess. Any numpy random distribution
        can be based to this function. Alternatively, the trapezoid distribution within this library
        can also be used

        dist:
            -distribution from which the confounder-outcome Risk Ratio is pulled from. Input should be
             something like numpy.random.triangular(left=0.2,mode=0.3,right=0.4)
        '''
        if seed != None:
            np.random.seed(seed)
        self.pc0 = dist
        self._p0dist = True

    def fit(self):
        '''After the observed Risk Ratio, distribution of the confounder-outcome Risk Ratio, proportion
        of the unmeasured confounder in exposed, proportion of the unmeasured confounder in the unexposed.

        sample:
            -number of samples to pull from each distribution and recalculate the corrected risk ratio. The
             default is 10,000. A high number is generally recommended
        seed:
            -set the numpy seed. Default is None

        Formula:
                    RR* = RR / d
                      d = (p1 * (RRc-1) + 1) / (p0 * (RRc - 1) + 1)
        RR* : corrected risk ratio
        RR  : observed risk ratio in the data set
        RRc : risk ratio between unmeasured confounder and outcome
        p1  : probability/proportion of unmeasured confounder in exposed
        p0  : probability/proportion of unmeasured confounder in unexposed
        '''
        if self._rrdist == False:
            raise ValueError('"confounder_RR_distribution()" has not been specified')
        if self._p1dist == False:
            raise ValueError('"prop_confounder_exposed()" has not been specified')
        if self._p0dist == False:
            raise ValueError('"prop_confounder_unexposed()" has not been specified')
        
        #Monte Carlo
        sf = pd.DataFrame(index = [i for i in range(self.sample)])
        sf['RR_cnf'] = self.RRc_dist
        sf['p1'] = self.pc1
        sf['p0'] = self.pc0
        sf['RR_obs'] = self.RRo
        sf['RR_cor'] = sf['RR_obs'] / ((sf['p1']*(sf['RR_cnf']-1)+1) / ((sf['p0']*(sf['RR_cnf']-1)+1)))

        #setting new attribute
        self.corrected_RR = list(sf['RR_cor'])

    def summary(self,decimal=4):
        '''Generate the summary information after the corrected risk ratio distribution is
        generated. fit() must be run before this

        decimal:
            -number of decimal places to display in output. Default is 4
        '''
        print('----------------------------------------------------------------------')
        print('Median corrected Risk Ratio: ',round(np.median(self.corrected_RR),decimal))
        print('Mean corrected Risk Ratio: ',round(np.mean(self.corrected_RR),decimal))
        print('25th & 75th Percentiles: ',np.round_(np.percentile(self.corrected_RR,q=[25,75]),decimals=decimal))     
        print('2.5th & 97.5th Percentiles: ',np.round_(np.percentile(self.corrected_RR,q=[2.5,97.5]),decimals=decimal))     
        print('----------------------------------------------------------------------')

    def plot(self,bw_method='scott',fill=True,color='b'):
        '''Generate a Gaussian kernel density plot of the corrected risk ratio distribution. The
        kernel density used is SciPy's Gaussian kernel. Either Scott's Rule or Silverman's Rule can
        be implemented

        bw_method:
            -method used to estimate the bandwidth. Following SciPy, either 'scott' or 'silverman' are valid options
        fill:
            -whether to color the area under the density curves. Default is true
        color:
            -color of the line/area for the treated group. Default is Blue
        '''
        x = np.linspace(np.min(self.corrected_RR),np.max(self.corrected_RR),100)
        density = stats.kde.gaussian_kde(self.corrected_RR,bw_method=bw_method)

        #Creating density plot
        ax = plt.gca()
        if fill == True:
            ax.fill_between(x,density(x),color=color,alpha=0.2)
        ax.plot(x, density(x),color=color)
        ax.set_xlabel('Corrected Risk Ratio')
        ax.set_ylabel('Density')
        return ax
