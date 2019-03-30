import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


class MonteCarloRR:
    r"""Monte Carlo simulation to assess the impact of an unmeasured binary confounder on the results
    of a study. Observed RR comes from the data analysis, while the RR between the unmeasured confounder
    and the outcome should be obtained from prior literature or constitute an reasonable guess.
    Probability of exposure between the groups should also be reasonable numbers.

    The Monte Carlo corrected Risk Ratio is calculated in each iteration by

    .. math::

        RR_{MC} = \frac{RR_{obs}}{\frac{p_1(RR_{c} - 1) + 1}{p_0(RR_{c} - 1) + 1}}

    Parameters
    ------------
    observed_RR : float
        Observed RR from the data, not accounting for some binary unmeasured confounder
    sd : float, optional
        Standard deviation of the observed log(risk ratio). This parameter is optional. If specified, then random
        error is incorporated into the bias analysis estimates
    sample : integer, optional
        Number of MC simulations to run. It is important that the specified size of later distributions
        matches this number of samples

    Examples
    -------------
    Monte Carlo bias analysis with trapezoidal distributions

    >>> from zepid.sensitivity_analysis import MonteCarloRR, trapezoidal
    >>> mcrr = MonteCarloRR(observed_RR=0.73322, sample=10000)
    >>> mcrr.confounder_RR_distribution(trapezoidal(mini=0.9, mode1=1.1, mode2=1.7, maxi=1.8, size=10000))
    >>> mcrr.prop_confounder_exposed(trapezoidal(mini=0.25, mode1=0.28, mode2=0.32, maxi=0.35, size=10000))
    >>> mcrr.prop_confounder_unexposed(trapezoidal(mini=0.55, mode1=0.58, mode2=0.62, maxi=0.65, size=10000))
    >>> mcrr.fit()

    Printing a summarization of the bias analysis to the console

    >>> mcrr.summary()

    Creating a density plot of the bias analysis results

    >>> import matplotlib.pyplot as plt
    >>> mcrr.plot()
    >>> plt.show()
    """
    def __init__(self, observed_RR, sd=None, sample=10000):
        self.RRo = observed_RR
        self._sd = sd
        self.sample = sample

        self.pc0 = None
        self.pc1 = None
        self.RRc_dist = None
        self.corrected_RR = None

    def confounder_RR_distribution(self, dist, seed=None):
        """Distribution of the risk ratio between the unmeasured confounder and the outcome. This
        value should come from prior literature or a reasonable guess. Any numpy random distribution
        can be based to this function. Alternatively, the trapezoid distribution within this library
        can also be used

        Parameters
        ------------
        dist :
            Distribution from which the confounder-outcome Risk Ratio is pulled from. Input should be something like
            `numpy.random.triangular(left=0.9,mode=1.2,right=1.6)` or `zepid.sensitivity_analysis.trapezoidal`
        seed : int, optional
            NumPy seed for the generated distribution. Default is None
        """
        if seed is not None:
            np.random.seed(seed)
        self.RRc_dist = dist

    def prop_confounder_exposed(self, dist, seed=None):
        """Distribution of the proportion of the unmeasured confounder in the exposed group. This value should come
        from prior literature or a reasonable guess. Any numpy random distribution can be based to this function.
        Alternatively, the trapezoid distribution within this library can also be used

        Parameters
        ------------
        dist :
            Distribution from which the confounder-exposure probability is pulled from. Input should be something like
            `numpy.random.triangular(left=0.9,mode=1.2,right=1.6)` or `zepid.sensitivity_analysis.trapezoidal`
        seed : int, optional
            NumPy seed for the generated distribution. Default is None
        """
        if seed is not None:
            np.random.seed(seed)
        self.pc1 = dist

    def prop_confounder_unexposed(self, dist, seed=None):
        """Distribution of the proportion of the unmeasured confounder in the unexposed group. This
        value should come from prior literature or a reasonable guess. Any numpy random distribution
        can be based to this function. Alternatively, the trapezoid distribution within this library
        can also be used

        Parameters
        ------------
        dist :
            Distribution from which the confounder-no exposure probability is pulled from. Input should be something
            like `numpy.random.triangular(left=0.9,mode=1.2,right=1.6)` or `zepid.sensitivity_analysis.trapezoidal`
        seed : int, optional
            NumPy seed for the generated distribution. Default is None
        """
        if seed is not None:
            np.random.seed(seed)
        self.pc0 = dist

    def fit(self):
        """After the observed Risk Ratio, distribution of the confounder-outcome Risk Ratio, proportion
        of the unmeasured confounder in exposed, proportion of the unmeasured confounder in the unexposed.

        .. math::
            RR* = RR / d
            d = (p1 * (RRc-1) + 1) / (p0 * (RRc - 1) + 1)

        Where RR* is the corrected risk ratio, RR is the observed risk ratio in the data set, RRc is the risk ratio
        between unmeasured confounder and outcome, p1 is the probability/proportion of unmeasured confounder in
        exposed, and p0 is the probability/proportion of unmeasured confounder in unexposed
        """
        if self.RRc_dist is None:
            raise ValueError('"confounder_RR_distribution()" has not been specified')
        if self.pc1 is None:
            raise ValueError('"prop_confounder_exposed()" has not been specified')
        if self.pc0 is None:
            raise ValueError('"prop_confounder_unexposed()" has not been specified')

        # Monte Carlo
        sf = pd.DataFrame(index=[i for i in range(self.sample)])
        sf['RR_cnf'] = self.RRc_dist
        sf['p1'] = self.pc1
        sf['p0'] = self.pc0
        sf['RR_obs'] = self.RRo
        sf['RR_cor'] = sf['RR_obs'] / ((sf['p1']*(sf['RR_cnf']-1)+1) / (sf['p0'] * (sf['RR_cnf'] - 1) + 1))

        if self._sd is not None:
            sf['RR_cor'] = np.exp(np.log(sf['RR_cor']) - np.random.normal(size=self.sample)*self._sd)

        # Setting new attribute
        self.corrected_RR = np.array(sf['RR_cor'])

    def summary(self, decimal=3):
        """Generate the summary information after the corrected risk ratio distribution is
        generated. fit() must be run before this

        Parameters
        -------------
        decimal : int, optional
            Decimal places to display in output. Default is 3
        """
        print('----------------------------------------------------------------------')
        print('Median corrected Risk Ratio: ', np.round(np.median(self.corrected_RR),decimal))
        print('Mean corrected Risk Ratio: ', np.round(np.mean(self.corrected_RR),decimal))
        print('25th & 75th Percentiles: ', np.round_(np.percentile(self.corrected_RR,q=[25, 75]), decimals=decimal))
        print('2.5th & 97.5th Percentiles: ', np.round_(np.percentile(self.corrected_RR,q=[2.5, 97.5]),
                                                        decimals=decimal))
        print('----------------------------------------------------------------------')

    def plot(self, bw_method='scott', fill=True, color='b'):
        """Generate a Gaussian kernel density plot of the corrected risk ratio distribution. The
        kernel density used is SciPy's Gaussian kernel. Either Scott's Rule or Silverman's Rule can
        be implemented

        Parameters
        -------------
        bw_method : str, optional
            Method used to estimate the bandwidth. Following SciPy, either 'scott' or 'silverman' are valid options
        fill : bool, optional
            Whether to color the area under the density curves. Default is true
        color : str, optional
            Color of the line/area for the treated group. Default is Blue

        Returns
        ------------
        matplotlib axes
        """
        x = np.linspace(np.min(self.corrected_RR),np.max(self.corrected_RR),100)
        density = stats.kde.gaussian_kde(self.corrected_RR,bw_method=bw_method)

        # Creating density plot
        ax = plt.gca()
        if fill:
            ax.fill_between(x, density(x), color=color, alpha=0.2)
        ax.plot(x, density(x), color=color)
        ax.set_xlabel('Corrected Risk Ratio')
        ax.set_ylabel('Density')
        return ax
