import warnings
import math
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from .utils import propensity_score


class IPTW:
    """
    Calculates the weight for inverse probability of treatment weights through logistic regression.
    Both stabilized or unstabilized weights are implemented. Default is just to calculate the prevalence
    of the treatment in the population.

    df:
        -pandas dataframe object containing all variables of interest
    treatment:
        -Variable name of treatment variable of interest. Must be coded as binary.
         1 should indicate treatment, while 0 indicates no treatment
    stabilized:
        -Whether to return stabilized or unstabilized weights. Input is True/False. Default is
         stabilized weights (True)
    standardize:
        -who to standardize the estimate to. Options are the entire population, the exposed, or
         the unexposed. See Sato & Matsuyama Epidemiology (2003) for details on weighting to exposed/unexposed
         Options for standardization are:
            'population'    :   weight to entire population
            'exposed'       :   weight to exposed individuals
            'unexposed'     :   weight to unexposed individuals
    """
    def __init__(self, df, treatment, stabilized=True, standardize='population'):
        self.df = df.copy()
        self.ex = treatment
        self.stabilized = stabilized
        if standardize in ['population', 'exposed', 'unexposed']:
            self.standardize = standardize
        else:
            raise ValueError('Please specify one of the currently supported weighting schemes: ' +
                             'population, exposed, unexposed')

    def regression_models(self, model_denominator, model_numerator='1', print_results=True):
        """
        Logistic regression model(s) for propensity score models. The model denominator must be specified for both
        stabilized and unstabilized weights. The optional argument 'model_numerator' allows specification of the
        stabilization factor for the weight numerator. By default model results are returned

        model_denominator:
            -statsmodels glm format for modeling data. Only includes predictor variables
             Example) 'var1 + var2 + var3'
        model_numerator:
            - statsmodels glm format for modeling data. Only includes predictor variables for the numerator.
             Default ('1') calculates the overall probability. In general this is recommended. If confounding
             variables are included in the numerator, they would later need to be adjusted for.
             Example) 'var1'
        print_results:
            -whether to print the model results from the regression models. Default is True
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
        """
        Uses the specified regression models from 'regression_models' to generate the corresponding
        inverse probability of treatment weights

        IPTW will have the Weight attribute which contains a pandas Series of the calculated IPTW
        """
        self.df['propscore'] = self.denominator_model.predict(self.df)
        if self.stabilized:
            n = self.numerator_model.predict(self.df)
        else:
            n = 1
        self.df['denom'] = self.df['propscore']
        self.df['numer'] = n
        self.df['iptw'] = self._weight_calculator(self.df, denominator='denom', numerator='numer')
        self.Weight = self.df['iptw']
        self.ProbabilityDenominator = self.df['denom']
        self.ProbabilityNumerator = self.df['numer']

    def plot_kde(self, bw_method='scott', fill=True, color_e='b', color_u='r'):
        """
        Generates a density plot that can be used to check whether positivity may be violated qualitatively. Note
        input probability variable, not the weight! The kernel density used is SciPy's Gaussian kernel. Either Scott's
        Rule or Silverman's Rule can be implemented.

        bw_method:
            -method used to estimate the bandwidth. Following SciPy, either 'scott' or 'silverman' are
             valid options
        fill:
            -whether to color the area under the density curves. Default is true
        color_e:
            -color of the line/area for the treated group. Default is Blue
        color_u:
            -color of the line/area for the treated group. Default is Red

        Returns matplotlib axes
        """
        x = np.linspace(0, 1, 10000)
        density_t = stats.kde.gaussian_kde(self.df.loc[self.df[self.ex] == 1]['propscore'].dropna(),
                                           bw_method=bw_method)
        density_u = stats.kde.gaussian_kde(self.df.loc[self.df[self.ex] == 0]['propscore'].dropna(),
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
        """
        Generates a stratified boxplot that can be used to visually check whether positivity may be violated,
        qualitatively.

        Returns matplotlib axes
        """
        boxes = (self.df.loc[self.df[self.ex] == 1]['propscore'].dropna(),
                 self.df.loc[self.df[self.ex] == 0]['propscore'].dropna())
        labs = ['Treat = 1', 'Treat = 0']
        meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='black')
        ax = plt.gca()
        ax.boxplot(boxes, labels=labs, meanprops=meanpointprops, showmeans=True)
        ax.set_ylabel('Probability')
        return ax

    def positivity(self, decimal=3):
        '''Use this to assess whether positivity is a valid assumption. Note that this should only be used for
        stabilized weights generated from IPTW. This diagnostic method is based on recommendations from
        Cole SR & Hernan MA (2008). For more information, see the following paper:
        Cole SR, Hernan MA. Constructing inverse probability weights for marginal structural models.
        American Journal of Epidemiology 2008; 168(6):656–664.

        decimal:
            -number of decimal places to display. Default is three
        '''
        if not self.stabilized:
            warnings.warn('Positivity should only be used for stabilized IPTW', UserWarning)
        self._pos_avg = float(np.mean(self.df['iptw'].dropna()))
        self._pos_max = np.max(self.df['iptw'].dropna())
        self._pos_min = np.min(self.df['iptw'].dropna())
        self._pos_sd = float(np.std(self.df['iptw'].dropna()))
        print('----------------------------------------------------------------------')
        print('IPW Diagnostic for positivity')
        print('''If the mean of the weights is far from either the min or max, this may\n indicate the model is
                incorrect or positivity is violated''')
        print('Standard deviation can help in IPTW model selection')
        print('----------------------------------------------------------------------')
        print('Mean weight:\t\t\t', round(self._pos_avg, decimal))
        print('Standard Deviation:\t\t', round(self._pos_sd, decimal))
        print('Minimum weight:\t\t\t', round(self._pos_min, decimal))
        print('Maximum weight:\t\t\t', round(self._pos_max, decimal))
        print('----------------------------------------------------------------------')

    def standardized_difference(self, variable, var_type, decimal=3):
        """Calculates the standardized mean difference between the treat/exposed and untreated/unexposed for a
        specified variable. Useful for checking whether a confounder was balanced between the two treatment groups
        by the specified IPTW model

        SMD based on: Austin PC 2011; https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3144483/

        variable:
            -label for variable to calculate the standardized difference
        var_type:
            -variable type. Options are 'binary' or 'continuous'
        decimal:
            -decimal places to display in results. Default is 3
        """
        # TODO add plotter of this... (needs variable type detector, patsy extractor, loop to calculate, plot df)
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
        """
        Calculates the IPTW based on the predicted probabilities and the specified group to standardize to
        """
        if self.stabilized:  # Stabilized weights
            if self.standardize == 'population':
                df['w'] = np.where(df[self.ex] == 1, (df[numerator] / df[denominator]),
                                   ((1 - df[numerator]) / (1 - df[denominator])))
                df['w'] = np.where(df[self.ex].isna(), np.nan, df['w'])
            # Stabilizing to exposed (compares all exposed if they were exposed versus unexposed)
            elif self.standardize == 'exposed':
                df['w'] = np.where(df[self.ex] == 1, 1,
                                   ((df[denominator] / (1 - df[denominator])) * ((1 - df[numerator]) /
                                                                                 df[numerator])))
                df['w'] = np.where(df[self.ex].isna(), np.nan, df['w'])
            # Stabilizing to unexposed (compares all unexposed if they were exposed versus unexposed)
            else:
                df['w'] = np.where(df[self.ex] == 1,
                                   (((1 - df[denominator]) / df[denominator]) * (df[numerator] /
                                                                                 (1 - df[numerator]))),
                                   1)
                df['w'] = np.where(df[self.ex].isna(), np.nan, df['w'])

        else:  # Unstabilized weights
            if self.standardize == 'population':
                df['w'] = np.where(df[self.ex] == 1, 1 / df[denominator], 1 / (1 - df[denominator]))
                df['w'] = np.where(df[self.ex].isna(), np.nan, df['w'])
            # Stabilizing to exposed (compares all exposed if they were exposed versus unexposed)
            elif self.standardize == 'exposed':
                df['w'] = np.where(df[self.ex] == 1, 1, (df[denominator] / (1 - df[denominator])))
                df['w'] = np.where(df[self.ex].isna(), np.nan, df['w'])
            # Stabilizing to unexposed (compares all unexposed if they were exposed versus unexposed)
            else:
                df['w'] = np.where(df[self.ex] == 1, ((1 - df[denominator]) / df[denominator]), 1)
                df['w'] = np.where(df[self.ex].isna(), np.nan, df['w'])
        return df['w']

    @staticmethod
    def _weighted_avg(df, v, w, t):
        """
        Calculates the weighted mean for continuous variables
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
        """
        Calculates the weighted standard deviation for continuous variables
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
