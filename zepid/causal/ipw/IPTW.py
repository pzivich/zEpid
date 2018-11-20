import warnings
import math
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from .utils import propensity_score


class IPTW:
    def __init__(self, df, treatment, stabilized=True, standardize='population'):
        """
        Calculates the weight for inverse probability of treatment weights through logistic regression.
        Both stabilized or unstabilized weights are implemented. Default is just to calculate the prevalence
        of the treatment in the population.

        The formula for stabilized weights is

        .. math::

            \pi_i = \frac{\Pr(A=a)}{\Pr(A=a|L=l)}

        For unstabilized weights

        .. math::

            \pi_i = \frac{1}{\Pr(A=a|L=l)}

        SMR unstabilized weights for weighting to exposed (A=1)

        .. math::

            \pi_i &= 1 if A = 1 \\
                  &= \frac{\Pr(A=1|L=l)}{\Pr(A=0|L=l)} if A = 0

        For SMR weighted to the unexposed (A=0) the equation becomes

        .. math::

            \pi_i &= \frac{\Pr(A=0|L=l)}{\Pr(A=1|L=l)} if A=1 \\
                  &= 1 if A = 0

        Parameters
        ----------
        df : DataFrame
            Pandas dataframe object containing all variables of interest
        treatment : str
            Variable name of treatment variable of interest. Must be coded as binary. 1 should indicate treatment, while 0
            indicates no treatment
        stabilized : bool, optional
            Whether to return stabilized or unstabilized weights. Default is stabilized weights (True)
        standardize : str, optional
            Who to standardize the estimate to. Options are the entire population, the exposed, or the unexposed. See Sato
            & Matsuyama Epidemiology (2003) for details on weighting to exposed/unexposed. Weighting to the exposed or
            unexposed is also referred to as SMR weighting. Options for standardization are:
            * 'population'    :   weight to entire population
            * 'exposed'       :   weight to exposed individuals
            * 'unexposed'     :   weight to unexposed individuals

        Example
        -------
        Stabilized IPTW weights
        >>>import zepid as ze
        >>>from zepid.causal.ipw import IPTW
        >>>df = ze.load_sample_data(False)
        >>>ipt = IPTW(df, treatment='art', stabilized=True)
        >>>ipt.regression_models('male + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
        >>>ipt.fit()

        Unstabilized IPTW weights
        >>>ipt = IPTW(df, treatment='art', stabilized=False)
        >>>ipt.regression_models('male + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
        >>>ipt.fit()

        SMR weight to the exposed population
        >>>ipt = IPTW(df, treatment='art', stabilized=False, standardize='exposed')
        >>>ipt.regression_models('male + age_rs1 + age_rs2 + cd40 + cd4_rs1 + cd4_rs2 + dvl0')
        >>>ipt.fit()

        Diagnostics
        1) Standard differences
        >>>ipt.StandardizedDifference('male', var_type='binary')
        >>>ipt.StandardizedDifference('age0', var_type='continuous')

        2) Positivity
        >>>ipt.positivity()

        3) Plots
        >>>ipt.plot_boxplot()
        >>>plt.show()
        >>>ipt.plot_kde()
        >>>plt.show()
        """
        self.denominator_model = None
        self.numerator_model = None

        self.Weight = None
        self.ProbabilityNumerator = None
        self.ProbabilityDenominator = None

        self.df = df.copy()
        self.ex = treatment
        self.stabilized = stabilized
        if standardize in ['population', 'exposed', 'unexposed']:
            self.standardize = standardize
        else:
            raise ValueError('Please specify one of the currently supported weighting schemes: ' +
                             'population, exposed, unexposed')

    def regression_models(self, model_denominator, model_numerator='1', print_results=True):
        """Logistic regression model(s) for propensity score models. The model denominator must be specified for both
        stabilized and unstabilized weights. The optional argument 'model_numerator' allows specification of the
        stabilization factor for the weight numerator. By default model results are returned

        Parameters
        ------------
        model_denominator : str
            String listing variables to predict the exposure, separated by +. For example, 'var1 + var2 + var3'. This
            is for the predicted probabilities of the denominator
        model_numerator : str, optional
            Optional string listing variables to predict the exposure, separated by +. Only used to calculate the
            numerator. Default ('1') calculates the overall probability of exposure. In general this is recommended. If
            confounding variables are included in the numerator, they would later need to be adjusted for. Argument is
            also only used when calculating stabilized weights
        print_results : bool, optional
            Whether to print the model results from the regression models. Default is True
        """
        self.denominator_model = propensity_score(self.df, self.ex + ' ~ ' + model_denominator,
                                                  print_results=print_results)
        if self.stabilized is True:
            self.numerator_model = propensity_score(self.df, self.ex + ' ~ ' + model_numerator,
                                                    print_results=print_results)
        else:
            if model_numerator != '1':
                warnings.warn('Argument for model_numerator is only used for stabilized=True')

    def fit(self):
        """Uses the specified regression models from 'regression_models' to generate the corresponding inverse
        probability of treatment weights

        Returns
        ------------
        IPTW class gains the Weight, ProbabilityDenominator, and ProbabilityNumerator attributed. Weights is a pandas
        Series containing the calculated IPTW.
        """
        if self.denominator_model is None:
            raise ValueError('No model has been fit to generated predicted probabilities')

        self.df['__denom__'] = self.denominator_model.predict(self.df)
        if self.stabilized:
            n = self.numerator_model.predict(self.df)
        else:
            n = 1
        self.df['__numer__'] = n
        self.Weight = self._weight_calculator(self.df, denominator='__denom__', numerator='__numer__')
        self.ProbabilityDenominator = self.df['__denom__']
        self.ProbabilityNumerator = self.df['__numer__']

    def plot_kde(self, bw_method='scott', fill=True, color_e='b', color_u='r'):
        """Generates a density plot that can be used to check whether positivity may be violated qualitatively. The
        kernel density used is SciPy's Gaussian kernel. Either Scott's Rule or Silverman's Rule can be implemented.
        Alternative option to the boxplot of probabilities

        Parameters
        ------------
        bw_method : str, optional
            Method used to estimate the bandwidth. Following SciPy, either 'scott' or 'silverman' are valid options
        fill : bool, optional
            Whether to color the area under the density curves. Default is true
        color_e : str, optional
            Color of the line/area for the treated group. Default is Blue
        color_u : str, optional
            Color of the line/area for the treated group. Default is Red

        Returns
        ---------------
        matplotlib axes
        """
        x = np.linspace(0, 1, 10000)
        density_t = stats.kde.gaussian_kde(self.df.loc[self.df[self.ex] == 1]['__denom__'].dropna(),
                                           bw_method=bw_method)
        density_u = stats.kde.gaussian_kde(self.df.loc[self.df[self.ex] == 0]['__denom__'].dropna(),
                                           bw_method=bw_method)
        ax = plt.gca()
        if fill:
            ax.fill_between(x, density_t(x), color=color_e, alpha=0.2, label=None)
            ax.fill_between(x, density_u(x), color=color_u, alpha=0.2, label=None)
        ax.plot(x, density_t(x), color=color_e, label='Treat = 1')
        ax.plot(x, density_u(x), color=color_u, label='Treat = 0')
        ax.set_xlabel('Probability')
        ax.set_ylabel('Density')
        ax.legend()
        return ax

    def plot_boxplot(self):
        """Generates a stratified boxplot that can be used to visually check whether positivity may be violated,
        qualitatively. Alternative option to the kernel density plot.

        Returns
        -------------
        matplotlib axes
        """
        boxes = (self.df.loc[self.df[self.ex] == 1]['__denom__'].dropna(),
                 self.df.loc[self.df[self.ex] == 0]['__denom__'].dropna())
        labs = ['Treat = 1', 'Treat = 0']
        meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='black')
        ax = plt.gca()
        ax.boxplot(boxes, labels=labs, meanprops=meanpointprops, showmeans=True)
        ax.set_ylabel('Probability')
        return ax

    def positivity(self, decimal=3):
        """Use this to assess whether positivity is a valid assumption. Note that this should only be used for
        stabilized weights generated from IPTW. This diagnostic method is based on recommendations from
        Cole SR & Hernan MA (2008). For more information, see the following paper:
        Cole SR, Hernan MA. Constructing inverse probability weights for marginal structural models.
        American Journal of Epidemiology 2008; 168(6):656–664.

        Parameters
        --------------
        decimal : int, optional
            Number of decimal places to display. Default is three

        Returns
        --------------
        None
            Prints the positivity results to the console but does not return any objects
        """
        self.df['iptw'] = self.Weight
        if not self.stabilized:
            warnings.warn('Positivity should only be used for stabilized IPTW')
        avg = float(np.mean(self.df['iptw'].dropna()))
        mx = np.max(self.df['iptw'].dropna())
        mn = np.min(self.df['iptw'].dropna())
        sd = float(np.std(self.df['iptw'].dropna()))
        print('----------------------------------------------------------------------')
        print('IPW Diagnostic for positivity')
        print('''If the mean of the weights is far from either the min or max, this may\n indicate the model is
                incorrect or positivity is violated''')
        print('Standard deviation can help in IPTW model selection')
        print('----------------------------------------------------------------------')
        print('Mean weight:\t\t\t', round(avg, decimal))
        print('Standard Deviation:\t\t', round(sd, decimal))
        print('Minimum weight:\t\t\t', round(mn, decimal))
        print('Maximum weight:\t\t\t', round(mx, decimal))
        print('----------------------------------------------------------------------')

    def StandardizedDifference(self, variable, var_type, decimal=3):
        """Calculates the standardized mean difference between the treat/exposed and untreated/unexposed for a
        specified variable. Useful for checking whether a confounder was balanced between the two treatment groups
        by the specified IPTW model SMD based on: Austin PC 2011; https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3144483/

        Parameters
        ---------------
        variable : str
            Label for variable to calculate the standardized difference
        var_type : str
            Variable type. Options are 'binary' or 'continuous'
        decimal : int, optional
            Decimal places to display in results. Default is 3

        Returns
        --------------
        None
            Prints the positivity results to the console but does not return any objects
        """
        self.df['iptw'] = self.Weight
        if var_type == 'binary':
            wt1 = np.sum(self.df.loc[((self.df[variable] == 1) & (self.df[self.ex] == 1))]['iptw'].dropna())
            wt2 = np.sum(self.df.loc[(self.df[self.ex] == 1)].dropna()['iptw'])
            wt = wt1 / wt2
            wn1 = np.sum(self.df.loc[(self.df[variable] == 1) & (self.df[self.ex] == 0)]['iptw'].dropna())
            wn2 = np.sum(self.df.loc[self.df[self.ex] == 0]['iptw'].dropna())
            wn = wn1 / wn2
            wsmd = (wt - wn) / math.sqrt((wt*(1 - wt) + wn*(1 - wn))/2)
        elif var_type == 'continuous':
            wmn, wmt = self._weighted_avg(self.df, v=variable, w='iptw', t=self.ex)
            wsn, wst = self._weighted_std(self.df, v=variable, w='iptw', t=self.ex, xbar=[wmn, wmt])
            wsmd = (wmt - wmn) / math.sqrt((wst**2 + wsn**2)/2)
        else:
            raise ValueError('The only variable types currently supported are binary and continuous')
        print('----------------------------------------------------------------------')
        print('IPW Diagnostic for balance: Standardized Differences')
        if var_type == 'binary':
            print('\tBinary variable: ' + variable)
        if var_type == 'continuous':
            print('\tContinuous variable: ' + variable)
        print('----------------------------------------------------------------------')
        print('Weighted SMD: \t', round(wsmd, decimal))
        print('----------------------------------------------------------------------')

    def _weight_calculator(self, df, denominator, numerator):
        """Calculates the IPTW based on the predicted probabilities and the specified group to standardize to in the
        background for the fit() function. Not intended to be used by users

        df is the dataframe, denominator is the string indicating the column of Pr, numerator is the string indicating
        the column of Pr
        """
        if self.stabilized:  # Stabilized weights
            if self.standardize == 'population':
                df['w'] = np.where(df[self.ex] == 1, (df[numerator] / df[denominator]),
                                   ((1 - df[numerator]) / (1 - df[denominator])))
                df.loc[(df[self.ex] != 1) & (df[self.ex] != 0), 'w'] = np.nan
            # Stabilizing to exposed (compares all exposed if they were exposed versus unexposed)
            elif self.standardize == 'exposed':
                df['w'] = np.where(df[self.ex] == 1, 1,
                                   ((df[denominator] / (1 - df[denominator])) * ((1 - df[numerator]) /
                                                                                 df[numerator])))
                df.loc[(df[self.ex] != 1) & (df[self.ex] != 0), 'w'] = np.nan
            # Stabilizing to unexposed (compares all unexposed if they were exposed versus unexposed)
            else:
                df['w'] = np.where(df[self.ex] == 1,
                                   (((1 - df[denominator]) / df[denominator]) * (df[numerator] /
                                                                                 (1 - df[numerator]))),
                                   1)
                df.loc[(df[self.ex] != 1) & (df[self.ex] != 0), 'w'] = np.nan

        else:  # Unstabilized weights
            if self.standardize == 'population':
                df['w'] = np.where(df[self.ex] == 1, 1 / df[denominator], 1 / (1 - df[denominator]))
                df.loc[(df[self.ex] != 1) & (df[self.ex] != 0), 'w'] = np.nan
            # Stabilizing to exposed (compares all exposed if they were exposed versus unexposed)
            elif self.standardize == 'exposed':
                df['w'] = np.where(df[self.ex] == 1, 1, (df[denominator] / (1 - df[denominator])))
                df.loc[(df[self.ex] != 1) & (df[self.ex] != 0), 'w'] = np.nan
            # Stabilizing to unexposed (compares all unexposed if they were exposed versus unexposed)
            else:
                df['w'] = np.where(df[self.ex] == 1, ((1 - df[denominator]) / df[denominator]), 1)
                df.loc[(df[self.ex] != 1) & (df[self.ex] != 0), 'w'] = np.nan
        return df['w']

    @staticmethod
    def _weighted_avg(df, v, w, t):
        """Calculates the weighted mean for continuous variables. Used by StandardizedDifferences
        """
        l = []
        for i in [0, 1]:
            n = sum(df.loc[df[t] == i][v] * df.loc[df[t] == i][w])
            d = sum(df.loc[df[t] == i][w])
            a = n / d
            l.append(a)
        return l[0], l[1]

    @staticmethod
    def _weighted_std(df, v, w, t, xbar):
        """Calculates the weighted standard deviation for continuous variables. Used by StandardizedDifferences
        """
        l = []
        for i in [0, 1]:
            n1 = sum(df.loc[df[t] == i][w])
            d1 = sum(df.loc[df[t] == i][w]) ** 2
            d2 = sum(df.loc[df[t] == i][w] ** 2)
            n2 = sum(df.loc[df[t] == i][w] * ((df.loc[df[t] == i][v] - xbar[i]) ** 2))
            a = ((n1 / (d1 - d2)) * n2)
            l.append(a)
        return l[0], l[1]
