import numpy as np


def trapezoidal(mini, mode1, mode2, maxi, size=None):
    """Generates random data following a trapezoidal distribution. This function can be used to generate distributions
    of probabilities and effect measures for sensitivity analyses. It is particularly useful when used in conjunction
    with `rr_corr` to determine a distribution of potential results due to a single unadjusted confounder

    Parameters
    --------------
    mini : float
        Minimum value of trapezoidal distribution
    mode1 : float
        Start of uniform distribution
    mode2 : float
        End of uniform distribution
    maxi : float
        Maximum value of trapezoidal distribution
    size : int, optional
        Number of observations to generate. Default is None, which returns a single draw

    Returns
    --------------
    float or array
        Returns either a single float from the distribution or an array of floats

    Examples
    --------------
    Single draw from a trapezoidal distribution

    >>>from zepid.sensitivity_analysis import trapezoidal
    >>>trapezoidal(mini=0.2, mode1=0.3, mode2=0.5, maxi=0.6)

    100 draws from a trapezoidal distribution

    >>>trapezoidal(mini=0.2, mode1=0.3, mode2=0.5, maxi=0.6, size=100)

    References
    ----------
    Fox MP, Lash TL, Hamer DH. (2005). A sensitivity analysis of a randomized controlled trial of zinc in treatment
    of falciparum malaria in children. Contemporary clinical trials, 26(3), 281-289.

    Fox MP, Lash TL Greenland S. (2005). A method to automate probabilistic sensitivity analyses of misclassified
    binary variables. International journal of epidemiology, 34(6), 1370-1376.
    """
    if size is None:
        p = np.random.uniform()
        v = (p*(maxi+mode2-mini-mode1)+(mini+mode1)) / 2
        if v < mode1:
            v = mini + np.sqrt((mode1-mini)*(2*v-mini-mode1))
        elif v > mode2:
            v = maxi - np.sqrt(2*(maxi-mode2)*(v-mode2))
        else:
            pass   
        return v
    
    # Draws the number specified by size
    elif type(size) is int:
        va = []
        for i in range(size):
            va.append(trapezoidal(mini=mini, mode1=mode1, mode2=mode2, maxi=maxi, size=None))
        return np.array(va)
    
    # ValueError is size is not an integer
    else:
        raise ValueError('"size" must be an integer')

