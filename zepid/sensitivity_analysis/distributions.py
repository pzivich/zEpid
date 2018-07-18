import warnings
import math 
import numpy as np

def trapezoidal(mini, mode1, mode2, maxi, size=None):
    '''Creates trapezoidal distribution based on Fox & Lash 2005. This function 
    can be used to generate distributions of probabilities and effect measures for
    sensitivity analyses. It is particularly useful when used in conjunction with 
    rr_corr to determine a distribution of potential results due to a single unadjusted
    confounder
    
    mini:
        -minimum value of trapezoidal distribution
    mode1:
        -Start of uniform distribution
    mode2:
        -End of uniform distribution
    maxi:
        -maximum value of trapezoidal distribution
    size:
        -number of observations to generate. Default is None, which returns a single draw
    
    Example)
    >>>zepid.sens_analysis.trapezoidal(mini=0.2,mode1=0.3,mode2=0.5,maxi=0.6)
    '''
    if size == None:
        p = np.random.uniform()
        v = (p*(maxi+mode2-mini-mode1)+(mini+mode1)) / 2
        if v < mode1:
            v = mini + np.sqrt((mode1-mini)*(2*v-mini-mode1))
        elif v > mode2:
            v = maxi - np.sqrt(2*(maxi-mode2)*(v-mode2))
        else:
            pass   
        return v
    
    #draws the number specified by size
    elif type(size) == int:
        va = []
        for i in range(size):
            va.append(trapezoidal(mini=mini,mode1=mode1,mode2=mode2,maxi=maxi,size=None))
        return va
    
    #ValueError is size is not an integer        
    else:
        raise ValueError('"size" must be an integer')
        
